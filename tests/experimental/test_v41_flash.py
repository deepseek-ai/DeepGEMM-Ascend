"""V41 Flash experimental kernel correctness tests and cold-L2 benchmarks."""

from argparse import ArgumentParser
from functools import partial

import torch
import torch_npu  # noqa: F401

import deep_gemm
from deep_gemm.testing import bench_msprof
from deep_gemm.testing.par_compile import par_compile


QKV_SHAPES = (
    (1, 32768),
    (96, 32768),
    (128, 36864),
    (160, 36864),
    (192, 36864),
    (256, 32768),
    (256, 36864),
    (513, 32768),
    (513, 36864),
)
O_PROJ_SHAPES = (1, 96, 128, 192, 224, 240, 256, 513)


def calc_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a, b = actual.float(), expected.float()
    denominator = (a.square().sum() + b.square().sum()).clamp_min(1e-30)
    return ((a - b).square().sum() / denominator).item()


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = calc_diff(actual, expected)
    assert torch.isfinite(actual.float()).all() and diff < 1e-4, f'{diff=}'


def assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_bytes = actual.cpu().contiguous().view(torch.uint8)
    expected_bytes = expected.cpu().contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


def quantize_mx_fp8(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # CPU reference: BF16 values, MX32 groups, two UE8M0 exponents packed per int16.
    values = values.cpu().to(torch.bfloat16)
    groups = values.float().reshape(*values.shape[:-1], values.size(-1) // 32, 32)
    scales = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)))
    quant = (groups / scales.unsqueeze(-1)).reshape(values.shape).to(torch.float8_e4m3fn)
    exp = (scales.view(torch.int32) >> 23) & 255
    packed_sf = (exp[..., ::2] | (exp[..., 1::2] << 8)).to(torch.int16)
    return quant, packed_sf


def make_fp8_pair(shape: tuple[int, ...], device: str = 'npu') -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = shape[-2], shape[-1] // 64
    data = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    sf = torch.empty((*shape[:-2], cols, rows), dtype=torch.int16, device=device).mT
    return data, sf


def fill_fp8_pair(
    pair: tuple[torch.Tensor, torch.Tensor],
    seed: int,
    cache: dict | None = None,
) -> None:
    key = (tuple(pair[0].shape), seed)
    if cache is None or key not in cache:
        gen = torch.Generator().manual_seed(seed)
        values = torch.randn(pair[0].shape, generator=gen, dtype=torch.bfloat16)
        quantized = quantize_mx_fp8(values)
        if cache is not None:
            cache[key] = quantized
    else:
        quantized = cache[key]
    pair[0].copy_(quantized[0])
    pair[1].copy_(quantized[1])


def offset_view(tensor: torch.Tensor) -> torch.Tensor:
    backing = torch.empty((tensor.numel() + 1,), dtype=tensor.dtype, device=tensor.device)
    view = backing.as_strided(tensor.shape, tensor.stride(), storage_offset=1)
    view.copy_(tensor)
    return view


def make_qkv_case(m: int, n: int, device: str = 'npu') -> dict:
    num_pages = (m + 127) // 128 + 1
    return dict(
        x=make_fp8_pair((m, 5120), device),
        qkv_weight=make_fp8_pair((1792, 5120), device),
        q_b_weight=make_fp8_pair((n, 1280), device),
        q_norm_weight=torch.empty((1280,), dtype=torch.bfloat16, device=device),
        kv_norm_weight=torch.empty((512,), dtype=torch.bfloat16, device=device),
        positions=torch.empty((m,), dtype=torch.int32, device=device),
        cos_sin=torch.empty((512, 64), dtype=torch.float32, device=device),
        slots=torch.empty((m,), dtype=torch.int32, device=device),
        q=torch.empty((m, n), dtype=torch.bfloat16, device=device),
        cache=torch.empty((num_pages, 128, 1, 528), dtype=torch.uint8, device=device),
        eps=1e-6,
    )


def init_qkv_case(case: dict, seed: int = 2026, weight_cache: dict | None = None) -> None:
    for idx, key in enumerate(('x', 'qkv_weight', 'q_b_weight')):
        fill_fp8_pair(case[key], seed + idx, weight_cache if key != 'x' else None)

    gen = torch.Generator().manual_seed(seed)
    for key in ('q_norm_weight', 'kv_norm_weight'):
        case[key].copy_(torch.randn(case[key].shape, generator=gen, dtype=torch.bfloat16))

    angles = torch.randn((512, 32), generator=gen)
    case['cos_sin'].copy_(torch.cat((angles.cos(), angles.sin()), dim=1))

    m = case['q'].size(0)
    case['positions'].copy_(torch.arange(m, dtype=torch.int32) % 512)
    slots = torch.randperm(m, generator=gen).to(torch.int32)
    slots[::17] = -1
    slots[1::19] = case['cache'].numel() // 528
    case['slots'].copy_(slots)

    # Invalid slots must not require a valid position or write any cache entry.
    case['positions'][::17] = -12345
    case['positions'][1::19] = 12345
    case['cache'].fill_(0x35)


def make_o_proj_case(m: int, device: str = 'npu') -> dict:
    x = (
        torch.empty((m, 8, 4096), dtype=torch.float8_e4m3fn, device=device),
        torch.empty_strided((m, 8, 64), (1, 64 * m, m), dtype=torch.int16, device=device),
    )
    return dict(
        x=x,
        wv_weight=make_fp8_pair((8, 1024, 4096), device),
        o_weight=make_fp8_pair((5120, 8192), device),
        y=torch.empty((m, 5120), dtype=torch.bfloat16, device=device),
    )


def init_o_proj_case(case: dict, seed: int = 2026, weight_cache: dict | None = None) -> None:
    for idx, key in enumerate(('x', 'wv_weight', 'o_weight')):
        fill_fp8_pair(case[key], seed + idx, weight_cache if key != 'x' else None)


def qkv_reference(case: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = case['q'].shape
    device = case['q'].device
    qkv = torch.empty((m, 1792), dtype=torch.bfloat16, device=device)
    deep_gemm.fp8_gemm_nt(case['x'], case['qkv_weight'], qkv, recipe=(1, 1, 32))
    qkv = qkv.cpu()

    def rms_norm(values: torch.Tensor, weight: torch.Tensor, out_scale: float = 1.0) -> torch.Tensor:
        rstd = torch.rsqrt(values.double().square().mean(-1, keepdim=True) + case['eps']).float()
        return (values.float() * rstd * (weight.cpu().float() * out_scale)).bfloat16()

    qa, qa_sf = quantize_mx_fp8(rms_norm(qkv[:, :1280], case['q_norm_weight'], out_scale=2.0))
    sf = torch.empty((20, m), dtype=torch.int16, device=device).mT
    sf.copy_(qa_sf)
    q = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    deep_gemm.fp8_gemm_nt((qa.to(device), sf), case['q_b_weight'], q, recipe=(1, 1, 32))

    cache = torch.full(case['cache'].shape, 0x35, dtype=torch.uint8)
    kv = rms_norm(qkv[:, 1280:], case['kv_norm_weight'])
    slots = case['slots'].cpu()
    positions = case['positions'].cpu()
    cos_sin = case['cos_sin'].cpu()
    valid = (slots >= 0) & (slots < cache.numel() // 528)

    rows = kv[valid].float()
    angles = cos_sin[positions[valid].long()]
    even, odd = rows[:, 448::2].clone(), rows[:, 449::2].clone()
    rows[:, 448::2] = (even.double() * angles[:, :32].double() - (odd * angles[:, 32:]).double()).float()
    rows[:, 449::2] = (even.double() * angles[:, 32:].double() + (odd * angles[:, :32]).double()).float()

    data, kv_sf = quantize_mx_fp8(rows.bfloat16())
    cache.view(-1, 528)[slots[valid].long()] = torch.cat((data.view(torch.uint8), kv_sf.view(torch.uint8)), dim=-1)
    return q.cpu(), cache, valid


def check_qkv_reference(case: dict) -> None:
    q_ref, cache_ref, valid = qkv_reference(case)
    assert_close(case['q'].cpu(), q_ref)

    num_slots = cache_ref.numel() // 528
    written = torch.zeros(num_slots, dtype=torch.bool)
    written[case['slots'].cpu()[valid].long()] = True

    actual = case['cache'].cpu().contiguous().view(-1, 528)
    expected = cache_ref.view(-1, 528)
    assert_bitwise_equal(actual[~written], expected[~written])
    for values in (actual, expected):
        assert (values[written, 512:] != 255).all()

    def decode_kv_slot(slot_bytes: torch.Tensor) -> torch.Tensor:
        fp8_data = slot_bytes[:, :512].contiguous().view(torch.float8_e4m3fn).float()
        scales = torch.exp2(slot_bytes[:, 512:].float() - 127.0).repeat_interleave(32, dim=-1)
        return fp8_data * scales

    assert_close(decode_kv_slot(actual[written]), decode_kv_slot(expected[written]))


def check_o_proj_reference(case: dict) -> None:
    m, device = case['y'].size(0), case['y'].device
    mid = torch.empty((m, 8, 1024), dtype=torch.float8_e4m3fn, device=device)
    sf = torch.empty((128, m), dtype=torch.int16, device=device).mT
    y_ref = torch.empty_like(case['y'])
    deep_gemm.fp8_einsum('bhr,hdr->bhd', case['x'], case['wv_weight'], (mid, sf), recipe=(1, 1, 32))
    deep_gemm.fp8_gemm_nt((mid.flatten(1), sf), case['o_weight'], y_ref, recipe=(1, 1, 32))
    assert_close(case['y'], y_ref)


def check_replay(
    case: dict,
    api,
    check_fn,
    output_keys: tuple[str, ...],
    label: str,
    rounds: int,
) -> None:
    api(**case)
    check_fn(case)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        api(**case)

    saved_x = case['x'][0].clone()
    for idx in range(rounds):
        case['x'][0].copy_(saved_x)
        if idx % 3 == 1:
            case['x'][0].zero_()
        elif idx % 3 == 2:
            case['x'][0].view(torch.uint8).bitwise_xor_(128)
        if 'cache' in case:
            case['cache'].fill_(0x35)

        api(*case.values())
        expected = {key: case[key].cpu() for key in output_keys}
        assert torch.isfinite(expected[output_keys[0]].float()).all()

        for key in output_keys:
            case[key].fill_(0x35 if key == 'cache' else float('nan'))
        graph.replay()
        for key in output_keys:
            assert_bitwise_equal(case[key], expected[key])

    check_fn(case)
    m = case[output_keys[0]].size(0)
    print(f' > {label}: M={m} reference, positional/keyword API and {rounds} updated graph replays passed', flush=True)


def check_qkv_alignment(case: dict) -> None:
    local = {
        key: (offset_view(val[0]), offset_view(val[1])) if isinstance(val, tuple) else
             (offset_view(val) if isinstance(val, torch.Tensor) and key not in ('q', 'cache') else val)
        for key, val in case.items()
    }
    m = case['q'].size(0)
    padded_sf = torch.full((1 + 80 * (m + 7),), 0x7F7F, dtype=torch.int16, device=case['q'].device).as_strided(
        (m, 80), (1, m + 7), storage_offset=1)
    padded_sf.copy_(case['x'][1])
    local['x'] = (local['x'][0], padded_sf)
    pages = (m + 2) // 3 + 1
    block_size = 3
    stride = 3 * 528 + 256
    backing = torch.full((4 + pages * stride + 32,), 0xA5, dtype=torch.uint8, device=case['q'].device)
    local['cache'] = backing.as_strided((pages, block_size, 1, 528), (stride, 528, 528, 1), storage_offset=4)
    local['cache'].fill_(0x35)
    local['slots'][1::19] = pages * block_size
    local['q'] = torch.empty_like(case['q'])

    compact = dict(
        case,
        q=torch.empty_like(case['q']),
        slots=local['slots'],
        cache=torch.full(local['cache'].shape, 0x35, dtype=torch.uint8, device=case['q'].device),
    )
    deep_gemm.experimental.v41f_qkv_lora(**compact)
    deep_gemm.experimental.v41f_qkv_lora(**local)
    assert_bitwise_equal(local['q'], compact['q'])

    expected = torch.full(backing.shape, 0xA5, dtype=torch.uint8)
    expected.as_strided(local['cache'].shape, local['cache'].stride(), storage_offset=4).copy_(compact['cache'].cpu())
    assert_bitwise_equal(backing, expected)
    print(' > qkv: offset views, output alignment, and padded scale/page strides passed', flush=True)


def check_o_proj_alignment(case: dict) -> None:
    local = {
        key: (offset_view(val[0]), offset_view(val[1])) if isinstance(val, tuple) else val
        for key, val in case.items()
    }
    m = case['y'].size(0)
    padded_sf = torch.full((1 + 8 * 64 * (m + 15),), 0x7F7F, dtype=torch.int16, device=case['y'].device).as_strided(
        (m, 8, 64), (1, 64 * (m + 15), m + 15), storage_offset=1)
    padded_sf.copy_(case['x'][1])
    local['x'] = (local['x'][0], padded_sf)
    local['y'] = torch.empty_like(case['y'])
    deep_gemm.experimental.v41f_wv_o_proj(**local)
    assert_bitwise_equal(local['y'], case['y'])
    print(' > o_proj: offset views, output alignment, and padded scale strides passed', flush=True)


def benchmark_qkv(case: dict) -> None:
    fill_fp8_pair(case['x'], 2026)
    prof = bench_msprof(
        partial(deep_gemm.experimental.v41f_qkv_lora, **case),
        kernel_names='v41f_qkv_lora_kernel',
        flush_l2=True,
        backend='persistent',
    )
    m, n = case['q'].shape
    flops = 2 * m * (5120 * 1792 + 1280 * n)
    check_qkv_reference(case)
    print(f' > v41f_qkv_lora: M={m}, N={n}: {prof.dur_us:.3f} us, '
          f'{prof.tflops(flops):.2f} TFLOPS (bench_msprof/persistent)', flush=True)


def benchmark_o_proj(case: dict) -> None:
    fill_fp8_pair(case['x'], 2026)
    prof = bench_msprof(
        partial(deep_gemm.experimental.v41f_wv_o_proj, **case),
        kernel_names='v41f_wv_o_proj_kernel',
        flush_l2=True,
        backend='persistent',
    )
    m, n = case['y'].shape
    flops = 2 * m * (8 * 4096 * 1024 + 8192 * 5120)
    check_o_proj_reference(case)
    print(f' > v41f_wv_o_proj: M={m}, N={n}: {prof.dur_us:.3f} us, '
          f'{prof.tflops(flops):.2f} TFLOPS (bench_msprof/persistent)', flush=True)


def compile_v41f_qkv_lora(shapes: tuple[tuple[int, int], ...]) -> None:
    torch.npu.current_stream()
    kernels = [
        partial(deep_gemm.experimental.v41f_qkv_lora, **make_qkv_case(m, n, device='meta'))
        for m, n in shapes
    ]
    for m, n in shapes:
        for out_dim, in_dim in ((1792, 5120), (n, 1280)):
            a = make_fp8_pair((m, in_dim), device='meta')
            b = make_fp8_pair((out_dim, in_dim), device='meta')
            d = torch.empty((m, out_dim), dtype=torch.bfloat16, device='meta')
            kernels.append(partial(deep_gemm.fp8_gemm_nt, a, b, d, recipe=(1, 1, 32)))
    par_compile(kernels)


def compile_v41f_wv_o_proj(shapes: tuple[int, ...]) -> None:
    torch.npu.current_stream()
    kernels = []
    for m in shapes:
        case = make_o_proj_case(m, device='meta')
        mid = torch.empty((m, 8, 1024), dtype=torch.float8_e4m3fn, device='meta')
        sf = torch.empty((128, m), dtype=torch.int16, device='meta').mT
        kernels.extend((
            partial(deep_gemm.experimental.v41f_wv_o_proj, **case),
            partial(deep_gemm.fp8_einsum, 'bhr,hdr->bhd', case['x'], case['wv_weight'], (mid, sf), recipe=(1, 1, 32)),
            partial(deep_gemm.fp8_gemm_nt, (mid.flatten(1), sf), case['o_weight'], case['y'], recipe=(1, 1, 32)),
        ))
    par_compile(kernels)


def test_v41f_qkv_lora(
    shapes: tuple[tuple[int, int], ...] = QKV_SHAPES,
    rounds: int = 20,
    skip_prof: bool = False,
) -> None:
    print('Testing V41F QKV LoRA:', flush=True)
    compile_v41f_qkv_lora(shapes)
    weight_cache = {}
    for m, n in shapes:
        case = make_qkv_case(m, n)
        init_qkv_case(case, weight_cache=weight_cache)
        check_replay(case, deep_gemm.experimental.v41f_qkv_lora, check_qkv_reference, ('q', 'cache'), 'qkv', rounds)
        if not skip_prof:
            benchmark_qkv(case)
        if (m, n) in ((1, 32768), (128, 36864), (256, 32768), (513, 32768)):
            check_qkv_alignment(case)
    print(flush=True)


def test_v41f_wv_o_proj(
    shapes: tuple[int, ...] = O_PROJ_SHAPES,
    rounds: int = 20,
    skip_prof: bool = False,
) -> None:
    print('Testing V41F Wv + O-proj:', flush=True)
    compile_v41f_wv_o_proj(shapes)
    weight_cache = {}
    for m in shapes:
        case = make_o_proj_case(m)
        init_o_proj_case(case, weight_cache=weight_cache)
        check_replay(case, deep_gemm.experimental.v41f_wv_o_proj, check_o_proj_reference, ('y',), 'o_proj', rounds)
        if not skip_prof:
            benchmark_o_proj(case)
        if m in (1, 128, 256, 513):
            check_o_proj_alignment(case)
    print(flush=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('select', nargs='?', default='all', choices=('all', 'qkv', 'o_proj'))
    parser.add_argument('--m', type=int, nargs='+')
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--skip-prof', '--no-prof', dest='skip_prof', action='store_true',
                        help='Skip performance measurements and just test correctness')
    args = parser.parse_args()
    assert args.rounds > 0

    torch.set_num_threads(4)
    torch.npu.set_device(0)

    if args.select != 'o_proj':
        qkv_shapes = tuple((m, n) for m in args.m for n in (32768, 36864)) if args.m else QKV_SHAPES
        test_v41f_qkv_lora(qkv_shapes, rounds=args.rounds, skip_prof=args.skip_prof)
    if args.select != 'qkv':
        o_proj_shapes = tuple(args.m) if args.m else O_PROJ_SHAPES
        test_v41f_wv_o_proj(o_proj_shapes, rounds=args.rounds, skip_prof=args.skip_prof)
    print('V41F checks passed', flush=True)


if __name__ == '__main__':
    main()
