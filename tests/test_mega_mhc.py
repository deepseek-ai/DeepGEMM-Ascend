"""Mega mHC correctness tests and kernel benchmarks."""

import time
from functools import partial

import torch

from deep_gemm.testing import bench_msprof
from deep_gemm.testing.par_compile import par_compile
from generators import ceil_to_ue8m0
from utils import calc_diff, count_bytes


def enumerate_cases():
    for sf_layout in ('col', 'row'):
        for hidden in (5120, 7168):
            for num_tokens in (1, 64, 65, 96, 97, 127, 128, 129, 144, 145, 160, 161, 191, 192, 193, 200, 256, 257, 1025, 4096, 32768):
                yield sf_layout, num_tokens, hidden


def make_inputs(num_tokens: int, hidden: int, seed: int, *, device: str = 'npu'):
    if device.split(':')[0] == 'npu':
        import torch_npu  # noqa: F401

    hc_mult = 4
    num_hc_outputs = hc_mult * (hc_mult + 2)
    generator = None if device == 'meta' else torch.Generator(device=device).manual_seed(seed)

    def randn(shape, dtype):
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    inputs = dict(
        x=randn((num_tokens, hidden), torch.bfloat16),
        residual=randn((num_tokens, hc_mult, hidden), torch.bfloat16),
        post_mix=randn((num_tokens, hc_mult, 1), torch.float).sigmoid(),
        comb_res_mix=torch.rand((num_tokens, hc_mult, hc_mult), device=device, generator=generator),
        shifted_prev_mix=randn((num_tokens, hc_mult, 1), torch.float).sigmoid(),
        fn=randn((num_hc_outputs, hc_mult * hidden), torch.float).mul_(1e-2),
        mix_scales=randn((3,), torch.float).mul_(0.1),
        mix_bases=randn((num_hc_outputs,), torch.float).mul_(0.1),
        rmsnorm_weight=randn((hidden,), torch.float).mul_(0.1).add_(1).bfloat16(),
        hc_mult=hc_mult,
        hc_norm_eps=2e-5,
        hc_pre_eps=3e-4,
        hc_post_scale=1.25,
        sinkhorn_eps=2e-6,
        num_sinkhorn_iters=20,
        rmsnorm_eps=7e-6,
        rmsnorm_scale=1.25,
    )
    for _ in range(3):
        inputs['comb_res_mix'] /= inputs['comb_res_mix'].sum(-1, keepdim=True)
        inputs['comb_res_mix'] /= inputs['comb_res_mix'].sum(-2, keepdim=True)
    duplicates = tuple(i for i in (63, 64, 511, 512, num_tokens - 1) if 0 < i < num_tokens)
    for name in ('x', 'residual', 'post_mix', 'comb_res_mix', 'shifted_prev_mix'):
        if duplicates:
            inputs[name][list(set(duplicates))] = inputs[name][0].clone()
    return inputs, duplicates


def make_outputs(inputs, sf_layout: str, output_mode: str = 'both'):
    assert sf_layout in ('col', 'row') and output_mode in ('both', 'bf16', 'fp8')
    num_tokens, hidden = inputs['x'].shape
    outputs = dict(
        new_residual=torch.empty_like(inputs['residual']),
        new_prev_mix=torch.empty_like(inputs['shifted_prev_mix']),
        new_post_mix=torch.empty_like(inputs['post_mix']),
        new_comb_res_mix=torch.empty_like(inputs['comb_res_mix']),
        y_bf16=torch.empty_like(inputs['x']) if output_mode != 'fp8' else None,
        y_fp8=torch.empty((num_tokens, hidden), dtype=torch.float8_e4m3fn, device=inputs['x'].device) if output_mode != 'bf16' else None,
    )
    if output_mode != 'bf16':
        outputs['y_routed_sf' if sf_layout == 'row' else 'y_gemm_sf'] = torch.empty_strided(
            (num_tokens, hidden // 64), (hidden // 64, 1) if sf_layout == 'row' else (1, (num_tokens + 3) // 4 * 4),
            dtype=torch.int16, device=inputs['x'].device)
    return outputs


def run_mega_mhc(inputs, outputs, **overrides):
    import deep_gemm

    return deep_gemm.mega_mhc(**{**inputs, **outputs, **overrides})


def compile_mega_mhc(device: str = 'meta') -> None:
    print('Compiling Mega mHC kernels', flush=True)
    kernels = []
    # Let the JIT cache deduplicate variants; token counts participate in dispatch.
    cases = [*enumerate_cases(), ('col', 65, 4096), ('row', 65, 4096)]
    for sf_layout, num_tokens, hidden in cases:
        for output_mode in ('both', 'bf16', 'fp8'):
            if output_mode == 'bf16' and sf_layout == 'row':
                continue
            inputs, _ = make_inputs(num_tokens, hidden, 2026, device=device)
            outputs = make_outputs(inputs, sf_layout, output_mode)
            kernels.append(partial(run_mega_mhc, inputs, outputs))
    par_compile(kernels)


def quantize_fp8(y: torch.Tensor):
    num_tokens, hidden = y.shape
    assert hidden % 128 == 0
    groups = y.float().reshape(num_tokens, hidden // 32, 32)
    sf = ceil_to_ue8m0(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)
    y_fp8 = (groups * sf.reciprocal().unsqueeze(-1)).to(torch.float8_e4m3fn).reshape_as(y)
    # Two adjacent H32 exponent bytes form one H64 int16 scale word.
    packed_sf = (sf.contiguous().view(torch.int32) // 2**23).to(torch.uint8).view(torch.int16)
    return y_fp8, packed_sf


def _sinkhorn_reference(value: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


@torch.no_grad()
def mhc_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mix_scales: torch.Tensor,
    mix_bases: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    hc_mult: int,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_eps: float,
    rmsnorm_scale: float,
    shifted_prev_mix: torch.Tensor,
    sf_layout: str = 'col',
) -> dict[str, torch.Tensor]:
    assert sf_layout in ('col', 'row')
    new_residual = x.float().unsqueeze(1) * post_mix
    new_residual += torch.einsum(
        'tij,tih->tjh', comb_res_mix, residual.float())
    new_residual = new_residual.bfloat16()
    norm_input = (
        new_residual.float() * shifted_prev_mix).sum(dim=1).bfloat16()

    if x.device.type == 'npu':
        old_allow_hf32 = torch.npu.matmul.allow_hf32
        torch.npu.matmul.allow_hf32 = False
    try:
        mixes = new_residual.flatten(1).float() @ fn.mT
    finally:
        if x.device.type == 'npu':
            torch.npu.matmul.allow_hf32 = old_allow_hf32
    hc_sqr_sum = new_residual.float().square().sum((1, 2))
    mixes *= torch.rsqrt(
        hc_sqr_sum / (hc_mult * x.size(1)) + hc_norm_eps).unsqueeze(1)
    scales = torch.cat((
        mix_scales[0].expand(hc_mult),
        mix_scales[1].expand(hc_mult),
        mix_scales[2].expand(hc_mult * hc_mult),
    ))
    mixes = mixes * scales + mix_bases
    new_prev_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(2) + hc_pre_eps
    new_post_mix = (
        mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_scale
    ).unsqueeze(2)
    new_comb_res_mix = _sinkhorn_reference(
        mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult),
        num_sinkhorn_iters,
        sinkhorn_eps,
    )

    norm_input_float = norm_input.float()
    y_fp32 = (
        norm_input_float
        * torch.rsqrt(norm_input_float.square().mean(1) + rmsnorm_eps).unsqueeze(1)
        * (rmsnorm_weight.float() * rmsnorm_scale)
    )
    y_bf16 = y_fp32.bfloat16()
    result = {
        'new_residual': new_residual,
        'new_prev_mix': new_prev_mix,
        'new_post_mix': new_post_mix,
        'new_comb_res_mix': new_comb_res_mix,
        'y_bf16': y_bf16,
    }
    y_fp8, y_fp8_sf = quantize_fp8(y_bf16)
    result.update(y_fp8=y_fp8, y_sf=y_fp8_sf)
    return result


def logical_outputs(outputs):
    return {'y_sf' if name in ('y_gemm_sf', 'y_routed_sf') else name: tensor
            for name, tensor in outputs.items() if tensor is not None}


def assert_bitwise_equal(actual, expected, label):
    expected = logical_outputs(expected)
    for name, tensor in logical_outputs(actual).items():
        assert torch.equal(tensor.contiguous().view(torch.uint8), expected[name].contiguous().view(torch.uint8)), (label, name)


def check_correctness(actual, reference, case):
    assert actual.keys() == reference.keys(), case
    diffs = []
    for name, tensor in actual.items():
        ref = reference[name]
        assert tensor.shape == ref.shape and tensor.dtype == ref.dtype, (case, name)
        if tensor.dtype == torch.int16:
            tensor, ref = tensor.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
        diff = float(calc_diff(tensor, ref))
        assert diff < (2e-4 if name == 'y_fp8' else 5e-5), (case, name, diff)
        diffs.append(diff)
    return diffs


@torch.no_grad()
def test_mega_mhc_api_contract(device: str = 'npu') -> None:
    import deep_gemm

    def expect_error(inputs, outputs, message, **overrides):
        try:
            run_mega_mhc(inputs, outputs, **overrides)
        except (RuntimeError, TypeError) as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError(f'Expected error: {message}')

    inputs, _ = make_inputs(65, 4096, 2026, device=device)
    outputs = make_outputs(inputs, 'col')
    for overrides, message in (
        ({'shifted_prev_mix': None}, 'incompatible function arguments'),
        ({'new_prev_mix': None}, 'incompatible function arguments'),
        ({'y_bf16': None, 'y_fp8': None, 'y_gemm_sf': None}, 'provide at least one'),
        ({'y_fp8': None}, 'y_fp8 requires exactly one'),
        ({'y_gemm_sf': None}, 'y_fp8 requires exactly one'),
        ({'y_routed_sf': outputs['y_gemm_sf']}, 'provide only one'),
        ({'y_shared_sf': outputs['y_gemm_sf']}, 'a separate y_shared_sf is unsupported'),
        ({'shared_sf_block_m': 128}, 'shared_sf_block_m must be 0'),
        ({'hc_mult': 3}, 'hc_mult must be 4'),
        ({'num_sinkhorn_iters': 0}, 'num_sinkhorn_iters must be positive'),
    ):
        expect_error(inputs, outputs, message, **overrides)
    for name, tensor in {**inputs, **outputs}.items():
        if isinstance(tensor, torch.Tensor):
            dtype = torch.float32 if tensor.dtype != torch.float32 else torch.bfloat16
            expect_error(inputs, outputs, f'{name} has an unexpected dtype', **{name: tensor.to(dtype)})
    for hidden in (0, 128, 1536):
        unsupported, _ = make_inputs(1, hidden, 2026, device=device)
        expect_error(unsupported, make_outputs(unsupported, 'col'), 'hidden must be a positive multiple of 1024')
    for strides in ((128, 2), (1, 64), (1, 65), (68, 1), (1, 1)):
        sf = torch.empty_strided((65, 64), strides, dtype=torch.int16, device=device)
        expect_error(inputs, outputs, 'y_gemm_sf must be column-major', y_gemm_sf=sf)
    row_outputs = make_outputs(inputs, 'row')
    expect_error(inputs, row_outputs, 'y_fp8 requires exactly one', y_routed_sf=None)
    expect_error(inputs, row_outputs, 'y_routed_sf has an unexpected dtype', y_routed_sf=row_outputs['y_routed_sf'].int())
    expect_error(inputs, row_outputs, 'y_routed_sf must be contiguous row-major', y_routed_sf=outputs['y_gemm_sf'])
    expect_error(inputs, outputs, 'y_gemm_sf must be column-major', y_gemm_sf=row_outputs['y_routed_sf'])
    old_num_sms, old_dry_run = deep_gemm.get_num_sms(), deep_gemm._C.get_dry_run()
    try:
        deep_gemm.set_num_sms(16)
        expect_error(inputs, outputs, 'num_ai_cores == kNumAICores')
        deep_gemm.set_num_sms(old_num_sms)
        for tensor in logical_outputs(outputs).values():
            (tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor).fill_(1)
        before = {name: tensor.clone() for name, tensor in logical_outputs(outputs).items()}
        deep_gemm._C.set_dry_run(True)
        run_mega_mhc(inputs, outputs)
        assert_bitwise_equal(outputs, before, 'dry-run')
        meta_inputs, _ = make_inputs(65, 4096, 2026, device='meta')
        for sf_layout in ('col', 'row'):
            for output_mode in ('both', 'bf16', 'fp8'):
                meta_outputs = make_outputs(meta_inputs, sf_layout, output_mode)
                run_mega_mhc(meta_inputs, meta_outputs)
        expect_error(meta_inputs, meta_outputs, 'x has an unexpected dtype', x=meta_inputs['x'].float())
        expect_error(meta_inputs, meta_outputs, 'same device as x', rmsnorm_weight=inputs['rmsnorm_weight'])
        deep_gemm._C.set_dry_run(False)
        expect_error(meta_inputs, meta_outputs, 'requires NPU tensors')
    finally:
        deep_gemm.set_num_sms(old_num_sms)
        deep_gemm._C.set_dry_run(old_dry_run)

    # Calls preserve inputs; graph replay and independent streams own their workspace.
    inputs, _ = make_inputs(65, 4096, 2026, device=device)
    for output_mode in ('both', 'bf16', 'fp8'):
        before = {name: tensor.clone() for name, tensor in inputs.items() if isinstance(tensor, torch.Tensor)}
        outputs = make_outputs(inputs, 'col', output_mode)
        assert run_mega_mhc(inputs, outputs, y_routed_sf=None, y_shared_sf=None, shared_sf_block_m=0) is None
        row_outputs = make_outputs(inputs, 'row', output_mode)
        run_mega_mhc(inputs, row_outputs, y_gemm_sf=None, y_shared_sf=None, shared_sf_block_m=0)
        assert_bitwise_equal(outputs, logical_outputs(row_outputs), 'row/column-major SF')
        assert_bitwise_equal(before, inputs, 'input preservation')
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        graph_outputs = make_outputs(inputs, 'col', output_mode)
        with torch.npu.graph(graph):
            run_mega_mhc(inputs, graph_outputs)
        for _ in range(3):
            inputs['x'].add_(0.125)
            graph.replay()
            run_mega_mhc(inputs, outputs)
            assert_bitwise_equal(graph_outputs, outputs, 'graph replay')
        pending = []
        for _ in range(2):
            stream = torch.npu.Stream(device=device)
            stream.wait_stream(torch.npu.current_stream(device))
            with torch.npu.stream(stream):
                stream_outputs = make_outputs(inputs, 'col', output_mode)
                run_mega_mhc(inputs, stream_outputs)
            pending.append((stream, stream_outputs))
        for stream, stream_outputs in pending:
            stream.synchronize()
            assert_bitwise_equal(stream_outputs, outputs, 'multi-stream')
    empty, _ = make_inputs(0, 4096, 2026, device=device)
    for sf_layout in ('col', 'row'):
        assert run_mega_mhc(empty, make_outputs(empty, sf_layout)) is None
    print('Mega mHC API contract: OK', flush=True)


@torch.no_grad()
def test_mega_mhc(device: str = 'npu', skip_prof: bool = False) -> None:
    print('Testing Mega shifted mHC:')
    for case_idx, (sf_layout, num_tokens, hidden) in enumerate(enumerate_cases()):
        inputs, duplicates = make_inputs(
            num_tokens, hidden, 2026 + case_idx, device=device)
        outputs = make_outputs(inputs, sf_layout)
        reference = mhc_reference(**inputs, sf_layout=sf_layout)
        run_mega_mhc(inputs, outputs)
        actual = logical_outputs(outputs)
        diffs = check_correctness(actual, reference, (sf_layout, num_tokens, hidden))
        expected = {name: tensor.clone() for name, tensor in actual.items()}
        for _ in range(10):
            run_mega_mhc(inputs, outputs)
            assert_bitwise_equal(actual, expected, 'repeated invocation')
        for token_idx in duplicates:
            assert_bitwise_equal({name: tensor[token_idx] for name, tensor in actual.items()},
                                 {name: tensor[0] for name, tensor in actual.items()}, f'token {token_idx}')
        for output_mode in ('bf16', 'fp8'):
            optional_outputs = make_outputs(inputs, sf_layout, output_mode)
            for _ in range(2):
                run_mega_mhc(inputs, optional_outputs)
                assert_bitwise_equal(optional_outputs, expected, output_mode)
        if skip_prof:
            print(f' > {sf_layout:5} T={num_tokens:5}, H={hidden:4}: diff {max(diffs):.1e}', flush=True)
            continue
        # Report the median of three cold-L2 short batches.
        profiles = []
        for _ in range(3):
            torch.npu.synchronize(device)
            time.sleep(0.5)
            profiles.append(bench_msprof(
                lambda: run_mega_mhc(inputs, outputs),
                kernel_names='mega_mhc_kernel', num_warmups=2, num_tests=7,
                flush_l2=True, backend='fast'))
        prof = sorted(profiles, key=lambda profile: profile.dur_us)[1]
        logical_inputs = [tensor for tensor in inputs.values() if isinstance(tensor, torch.Tensor)]
        logical_bytes = count_bytes(*logical_inputs, *actual.values())
        intermediate_io_bytes = 2 * count_bytes(outputs['y_bf16'])
        print(f' > {sf_layout:5} T={num_tokens:5}, H={hidden:4}: {prof.dur_us:6.1f} us, '
              f'{prof.gbps(logical_bytes):4.0f} GB/s '
              f'({prof.gbps(logical_bytes + intermediate_io_bytes):4.0f} incl. Scratch I/O) | '
              f'AIC[mte2={prof.aic_mte2 * 100:4.1f}%, mad={prof.aic_mad * 100:4.1f}%] | '
              f'AIV[mte2={prof.aiv_mte2 * 100:4.1f}%, vec={prof.aiv_vec * 100:4.1f}%] | '
              f'diff {max(diffs):.1e}', flush=True)
    print()


if __name__ == '__main__':
    compile_mega_mhc()
    test_mega_mhc_api_contract()
    test_mega_mhc()
