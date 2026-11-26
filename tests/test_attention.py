import enum
from functools import partial
import gc
import itertools
import os
import random
from statistics import median
import time
import torch

import deep_gemm
from deep_gemm.testing.bench import bench_msprof, close_persistent_profiler, flush_l2_cache
from deep_gemm.testing.par_compile import par_compile, dry_run
from utils import calc_diff, count_bytes, empty_kv_cache


class Mode(enum.Enum):
    """How each row's valid KV range [ks, ke) is generated."""
    FullRange = enum.auto()  # ks=0, ke=seq_len_kv (whole sequence)
    Causal    = enum.auto()  # single-machine causal (ke = arange + offset)
    Cp        = enum.auto()  # context-parallel chunked causal
    Random    = enum.auto()  # per-row random interval [min(a, b), max(a, b))

    def __str__(self) -> str: return self.name


DEFAULT_MODES = (Mode.Cp, Mode.Causal)

SPARSE_MQA_PREFILL_KV_LENS = (64 * 1024, 512 * 1024)
SPARSE_MQA_PREFILL_REQUEST_TOKENS = (3157, 5035)
SPARSE_MQA_PREFILL_OVERLAPS = (0.8, 0.9)
MQA_PERF_NUM_HEADS = (8, 12, 16, 20, 32, 64)


def format_pipe_profile(prof):
    cycles = (f'AIC:{prof.aic_total_cycles}/AIV:{prof.aiv_total_cycles}'
              if prof.aic_total_cycles or prof.aiv_total_cycles else 'unavailable')
    return (f'cycles={cycles} | '
            f'AIC[mad={prof.aic_mad*100:4.1f}%, scalar={prof.aic_scalar*100:4.1f}%, '
            f'mte1={prof.aic_mte1*100:4.1f}%, mte2={prof.aic_mte2*100:4.1f}%, '
            f'mte3={prof.aic_mte3*100:4.1f}%, fix={prof.aic_fixpipe*100:4.1f}%] | '
            f'AIV(avg2)[vec={prof.aiv_vec*100:4.1f}%, scalar={prof.aiv_scalar*100:4.1f}%, '
            f'mte2={prof.aiv_mte2*100:4.1f}%, mte3={prof.aiv_mte3*100:4.1f}%]')


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def expect_error(error_type, action, text=''):
    try:
        action()
    except error_type as error:
        assert text in str(error)
    else:
        raise AssertionError(f'{error_type.__name__} was not raised')


def ceil_to_ue8m0_with_byte(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bits = x.abs().float().view(torch.int)
    exp = (((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()).clamp(1, 254)
    return (exp << 23).view(torch.float), exp.to(torch.uint8)


def quantize_to_fp4_raw(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ax = x.abs()
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += (ax > boundary).to(torch.uint8)
    sign = (x < 0) & (code != 0)
    values = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=x.device, dtype=torch.float32)
    repr_f32 = values[code.long()]
    repr_f32 = torch.where(sign, -repr_f32, repr_f32)
    code = code | (sign.to(torch.uint8) * 8)
    packed = (code[..., ::2] & 0x0f) | ((code[..., 1::2] & 0x0f) << 4)
    return packed.contiguous().view(torch.int8), repr_f32


def quantize_mqa_tensor(x: torch.Tensor, use_fp4: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sf_div = 6.0 if use_fp4 else 448.0
    sf_shape = (*x.shape[:-1], x.shape[-1] // 32, 32)
    sf, sf_byte = ceil_to_ue8m0_with_byte(x.float().reshape(sf_shape).abs().amax(dim=-1).clamp(1e-4) / sf_div)
    x_scaled = (x.float().reshape(sf_shape) / sf[..., None]).reshape_as(x)
    if use_fp4:
        x_quant, x_repr = quantize_to_fp4_raw(x_scaled)
    else:
        x_quant = x_scaled.to(torch.float8_e4m3fn)
        x_repr = x_quant.float()
    x_ref = (x_repr.reshape(sf_shape) * sf[..., None]).reshape_as(x)
    sf_packed = sf_byte[..., ::2].to(torch.int16) | (sf_byte[..., 1::2].to(torch.int16) << 8)
    return x_quant, sf_packed.contiguous(), x_ref


def ref_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor, cu_seq_len_k_end: torch.Tensor) -> torch.Tensor:
    seq_len, num_heads, _ = q.shape
    seq_len_kv = kv.shape[0]
    q_f32 = q.float()
    kv_f32 = kv.float()
    weights_f32 = weights.float()

    logits = torch.empty((seq_len, seq_len_kv), dtype=torch.float32, device=q.device)
    kv_positions = torch.arange(seq_len_kv, device=q.device)
    split_kv = int(os.getenv('DG_MQA_SPLIT_KV', '256'))
    for kv_base in range(0, seq_len_kv, split_kv):
        kv_end = min(kv_base + split_kv, seq_len_kv)
        score = torch.einsum('qhd,kd->qhk', q_f32, kv_f32[kv_base:kv_end])
        out = (score.relu() * weights_f32[:, :, None]).sum(dim=1)
        mask = ((kv_positions[kv_base:kv_end][None, :] >= cu_seq_len_k_start[:, None]) &
                (kv_positions[kv_base:kv_end][None, :] < cu_seq_len_k_end[:, None]))
        logits[:, kv_base:kv_end] = out.masked_fill(~mask, float('-inf'))
    return logits


def generate_ks_ke(seq_len: int, seq_len_kv: int, mode: Mode) -> tuple[torch.Tensor, torch.Tensor, int]:
    ks = torch.zeros((seq_len,), device='npu', dtype=torch.int32)
    if mode == Mode.Random:
        a = torch.randint(0, seq_len_kv + 1, (seq_len,), device='npu', dtype=torch.int32)
        b = torch.randint(0, seq_len_kv + 1, (seq_len,), device='npu', dtype=torch.int32)
        ks = torch.minimum(a, b)
        ke = torch.maximum(a, b)
    elif mode == Mode.FullRange:
        ke = torch.full((seq_len,), seq_len_kv, device='npu', dtype=torch.int32)
    elif mode == Mode.Causal:
        ke = torch.arange(seq_len, device='npu', dtype=torch.int32) + (seq_len_kv - seq_len)
    else:  # Mode.Cp
        assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
        cp_size = seq_len_kv // seq_len
        chunk_size = seq_len // 2
        cp_id = cp_size // 3
        ke = torch.empty((seq_len,), device='npu', dtype=torch.int32)
        offsets = torch.arange(chunk_size, device='npu', dtype=torch.int32)
        ke[:chunk_size] = cp_id * chunk_size + offsets
        ke[chunk_size:] = (cp_size * 2 - 1 - cp_id) * chunk_size + offsets
    return ks, ke, int((ke - ks).max().item())


def run_mqa_logits_case(seq_len: int, seq_len_kv: int, num_heads: int, head_dim: int,
                        skip_prof: bool, use_fp4: bool = False,
                        mode: Mode = Mode.FullRange, collect: list = None) -> None:
    dtype_name = 'FP4' if use_fp4 else 'FP8'
    case_tag = f'SQ={seq_len:4}, SK={seq_len_kv:5}, H={num_heads:2}, D={head_dim:3}, {str(mode):9}'

    fill = torch.ones if os.getenv('DG_MQA_TEST_PATTERN') == 'ones' else torch.randn
    q_bf16 = fill((seq_len, num_heads, head_dim), device='npu', dtype=torch.bfloat16)
    kv_bf16 = fill((seq_len_kv, head_dim), device='npu', dtype=torch.bfloat16)
    weights = fill((seq_len, num_heads), device='npu', dtype=torch.bfloat16)
    cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k = generate_ks_ke(seq_len, seq_len_kv, mode)
    valid_len = cu_seq_len_k_end - cu_seq_len_k_start

    q, q_sf, q_ref = quantize_mqa_tensor(q_bf16, use_fp4)
    kv, kv_sf, kv_ref = quantize_mqa_tensor(kv_bf16, use_fp4)

    def run_kernel():
        return deep_gemm.fp8_fp4_mqa_logits(
            (q, q_sf), (kv, kv_sf), weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k)

    if collect is not None:
        collect.append(run_kernel)
        return

    ref_bf16 = ref_mqa_logits(q_bf16, kv_bf16, weights, cu_seq_len_k_start, cu_seq_len_k_end).to(torch.bfloat16)
    ref = ref_mqa_logits(q_ref, kv_ref, weights, cu_seq_len_k_start, cu_seq_len_k_end).to(torch.bfloat16)
    kernel_out = run_kernel()
    assert kernel_out.dtype == torch.bfloat16
    assert kernel_out.stride(0) * kernel_out.element_size() % 512 == 0 and kernel_out.stride(1) == 1
    assert kernel_out.shape == (seq_len, max_seqlen_k)
    out = torch.full((seq_len, seq_len_kv), float('-inf'), device='npu', dtype=torch.bfloat16)
    starts = cu_seq_len_k_start.cpu().tolist()
    ends = cu_seq_len_k_end.cpu().tolist()
    for i, (start, end) in enumerate(zip(starts, ends)):
        out[i, start:end] = kernel_out[i, :end - start]

    # Compressed rows define only their valid prefixes; compare those via a dense masked view.
    ref_f, out_f = ref.float(), out.float()
    ref_neginf = torch.isinf(ref_f) & (ref_f < 0)
    assert torch.equal(torch.isinf(out_f) & (out_f < 0), ref_neginf), 'logits -inf mask mismatch'
    out = out.masked_fill(ref_neginf, 0)
    ref = ref.masked_fill(ref_neginf, 0)
    ref_bf16 = ref_bf16.masked_fill(ref_neginf, 0)
    if os.getenv('DG_MQA_TEST_PATTERN') == 'ones':
        out_f = out.float()
        ref_ones = ref.float()
        assert torch.isfinite(out_f).all(), 'ones MQA logits produced non-finite output'
        max_err = (out_f - ref_ones).abs().max().item()
        if max_err >= 1e-5:
            print(' > Ones full-reduce failure:')
            print(f'   shape={tuple(out.shape)}, dtype={out.dtype}, heads={num_heads}, D={head_dim}, max_err={max_err:.3e}')
            print(f'   out min/max={out_f.min().item():.6e}/{out_f.max().item():.6e}')
            print(f'   out rows[0:8,0:16]=\n{out_f[:8, :16]}')
        assert max_err < 1e-5, f'ones full reduce max_err {max_err:.3e}'
        print(f' > {dtype_name} ones full reduce ({case_tag}): OK')
        return

    orig_diff = calc_diff(out.float(), ref_bf16.float())
    diff = calc_diff(out.float(), ref.float())
    orig_tol = 2e-2 if use_fp4 else 1e-3
    tol = 5e-4
    if (torch.isnan(out).any() or torch.isinf(out).any() or torch.isnan(torch.tensor(diff)) or
            torch.isnan(torch.tensor(orig_diff)) or diff >= tol or orig_diff >= orig_tol):
        out_f = out.float()
        ref_f = ref.float()
        finite = torch.isfinite(out_f)
        print(' > Debug MQA logits failure:')
        print(f'   {case_tag}, shape={tuple(out.shape)}, dtype={out.dtype}, '
              f'sim_diff={diff:.3e}/{tol:.1e}, orig_diff={orig_diff:.3e}/{orig_tol:.1e}')
        print(f'   out nan={torch.isnan(out_f).sum().item()}, inf={torch.isinf(out_f).sum().item()}, finite={finite.sum().item()}/{out.numel()}')
        print(f'   ref nan={torch.isnan(ref_f).sum().item()}, inf={torch.isinf(ref_f).sum().item()}')
        if finite.any():
            print(f'   out finite min/max={out_f[finite].min().item():.6e}/{out_f[finite].max().item():.6e}')
            print(f'   ref finite min/max={ref_f[torch.isfinite(ref_f)].min().item():.6e}/{ref_f[torch.isfinite(ref_f)].max().item():.6e}')
            err = (out_f - ref_f).abs()
            print(f'   abs err max/mean={err.max().item():.6e}/{err.mean().item():.6e}')
        bad = ~finite
        if bad.any():
            bad_idx = bad.nonzero()[0].tolist()
            q_idx, kv_idx = bad_idx
            q0, q1 = max(0, q_idx - 1), min(seq_len, q_idx + 2)
            k0, k1 = max(0, kv_idx - 8), min(seq_len_kv, kv_idx + 8)
            print(f'   first bad index={bad_idx}')
            print(f'   out window=\n{out_f[q0:q1, k0:k1]}')
            print(f'   ref window=\n{ref_f[q0:q1, k0:k1]}')
        else:
            print(f'   out first row first 16={out_f[0, :16]}')
            print(f'   ref first row first 16={ref_f[0, :16]}')
            print(f'   out first row first 64={out_f[0, :64]}')
            print(f'   ref first row first 64={ref_f[0, :64]}')
            print(f'   out first 4 rows first 16=\n{out_f[:4, :16]}')
            print(f'   ref first 4 rows first 16=\n{ref_f[:4, :16]}')
    assert orig_diff < orig_tol, f'MQA logits original-bf16 diff {orig_diff:.3e} exceeds {orig_tol:.1e}'
    assert diff < tol, f'MQA logits simulated diff {diff:.3e} exceeds {tol:.1e}'

    if not skip_prof:
        prof = bench_msprof(run_kernel, kernel_names='mqa_logits')
        valid_sum = valid_len.sum().item()
        flops = 2.0 * valid_sum * num_heads * head_dim
        bytes_moved = count_bytes(q, kv, q_sf, kv_sf, weights, kernel_out,
                                  cu_seq_len_k_start, cu_seq_len_k_end)
        print(f' > {dtype_name} Perf ({case_tag}) {prof.dur_us:7.1f} us | {prof.tflops(flops):4.0f} TFLOPS | '
              f'{prof.gbps(bytes_moved):5.0f} GB/s | {format_pipe_profile(prof)}')
    else:
        print(f' > {dtype_name} ({case_tag}): OK')


def compile_mqa_kernels(run_all) -> None:
    kernels = []
    run_all(kernels)
    par_compile(kernels, max_workers=128)
    del kernels
    gc.collect()
    torch.npu.empty_cache()


def test_mqa_logits(case_specs, skip_prof: bool, *, num_heads, use_fp4, head_dims) -> None:
    seq_len = int(os.getenv('DG_MQA_SQ', '4096'))
    seq_len_kv = int(os.getenv('DG_MQA_SK', '8192'))
    cases = tuple(
        (seq_len, seq_len_kv, heads, head_dim, skip_prof, fp4, mode)
        for mode, fp4, head_dim, heads in itertools.product(case_specs, use_fp4, head_dims, num_heads)
    )

    def run_all(collect=None):
        for case in cases:
            run_mqa_logits_case(*case, collect=collect)

    print(f'Testing MQA logits ({len(cases)} cases):')
    compile_mqa_kernels(run_all)

    run_all()
    print()


def test_mqa_logits_boundaries() -> None:
    full_seq_len, head_dim = 32, 128
    cases = []
    split_kv = 512
    for delta in (-1, 1):
        # FullRange forces the split+1 case to schedule and store its final one-token split.
        modes = (Mode.FullRange, Mode.Causal) if delta < 0 else (Mode.FullRange,)
        for mode in modes:
            cases.append((full_seq_len, split_kv + delta, 32, head_dim, True, delta > 0, mode))
    cases.append((512 // 32 + 1, split_kv, 32, head_dim, True, False, Mode.FullRange))
    for heads in (12, 64):
        cases.append((full_seq_len, split_kv, heads, head_dim, True, False, Mode.FullRange))
    cases.append((full_seq_len, split_kv, 8, head_dim, True, True, Mode.Random))

    def run_all(collect=None):
        for case in cases:
            run_mqa_logits_case(*case, collect=collect)

    print('Testing MQA logits boundaries:')
    compile_mqa_kernels(run_all)
    run_all()
    print()


def ref_paged_mqa_logits(q, kv, weights, context_lens, block_table, max_context_len):
    num_q_tokens = q.size(0)
    page_size = kv.size(1)
    out = torch.zeros((num_q_tokens, max_context_len), device=q.device)
    for q_idx, (context_len,) in enumerate(context_lens.cpu().tolist()):
        pages = (context_len + page_size - 1) // page_size
        keys = kv[block_table[q_idx, :pages].long()].flatten(0, 1).float()
        scores = torch.einsum('hd,kd->hk', q[q_idx, 0].float(), keys)
        logits = (scores.relu() * weights[q_idx, :, None]).sum(0)
        out[q_idx, :context_len] = logits[:context_len]
    return out


def reorder_paged_mqa_pages(pages, order):
    assert order in ('random', 'forward', 'reverse', 'pair_swap')
    if order == 'reverse':
        return pages.flip(0)
    if order == 'pair_swap':
        paired = pages.numel() // 2 * 2
        return torch.cat((pages[:paired].view(-1, 2).flip(1).flatten(),
                          pages[paired:]))
    return pages


def run_paged_mqa_logits_case(batch_size, num_heads, head_dim, avg_kv, use_fp4, request_tokens,
                              page_size=64, use_ascend_kv_layout=False, *, skip_prof=False,
                              random_lens=False, request_context_lens=None,
                              page_order='random', collect: list = None):
    # Drop the resident profiler so this case's reference traffic stays out of the PMU channel
    close_persistent_profiler()
    assert len(request_tokens) == batch_size
    request_tokens = torch.tensor(request_tokens, device='npu', dtype=torch.int32)
    min_tokens, max_tokens = int(request_tokens.min()), int(request_tokens.max())
    assert min_tokens >= 1
    request_tag = (f'TPR={min_tokens}' if min_tokens == max_tokens else
                   f'TPR={min_tokens}-{max_tokens}')
    indices = torch.arange(batch_size, device='npu', dtype=torch.int32).repeat_interleave(request_tokens)
    num_q_tokens = indices.numel()

    if request_context_lens is None:
        request_context_lens = (torch.randint(
            int(.7 * avg_kv), int(1.3 * avg_kv),
            (batch_size,), device='npu', dtype=torch.int32)
            if random_lens else
            torch.full((batch_size,), avg_kv, device='npu', dtype=torch.int32))
    else:
        assert len(request_context_lens) == batch_size
        request_context_lens = torch.tensor(request_context_lens, device='npu', dtype=torch.int32)
    offsets = torch.cat([torch.arange(int(n), device='npu', dtype=torch.int32)
                         for n in request_tokens.cpu()])
    context_lens = (request_context_lens.repeat_interleave(request_tokens) + offsets).view(-1, 1)
    request_context_lens += request_tokens - 1

    max_context_len = int(context_lens.max())
    pages_per_request = (request_context_lens + page_size - 1) // page_size
    max_pages, num_pages = int(pages_per_request.max()), int(pages_per_request.sum())
    q_bf16 = torch.randn((num_q_tokens, 1, num_heads, head_dim), device='npu', dtype=torch.bfloat16)
    kv_bf16 = torch.randn((num_pages, page_size, head_dim), device='npu', dtype=torch.bfloat16)
    weights = torch.randn((num_q_tokens, num_heads), device='npu', dtype=torch.bfloat16)
    q, q_sf, q_ref = quantize_mqa_tensor(q_bf16, use_fp4)
    kv, kv_sf, kv_ref = quantize_mqa_tensor(kv_bf16, use_fp4)
    if use_ascend_kv_layout:
        kv = kv.reshape(num_pages, page_size, -1, 32).permute(0, 2, 1, 3).contiguous().view_as(kv)
        kv_sf = kv_sf.reshape(num_pages, page_size // 16, 16, -1).permute(0, 1, 3, 2).contiguous().view(
            num_pages, -1, page_size)
    # Fused page layout: [page_size * data_bytes | page_size * sf_bytes] per page.
    data_bytes = kv.contiguous().view(torch.uint8).numel() // (num_pages * page_size)
    sf_bytes = kv_sf.contiguous().view(torch.uint8).numel() // (num_pages * page_size)
    fused = torch.cat([
        kv.contiguous().view(torch.uint8).view(num_pages, -1),
        kv_sf.contiguous().view(torch.uint8).view(num_pages, -1),
    ], dim=1)
    fused = fused.view(num_pages, page_size, 1, data_bytes + sf_bytes)

    pages = (torch.randperm(num_pages, device='npu', dtype=torch.int32)
             if page_order == 'random' else
             torch.arange(num_pages, device='npu', dtype=torch.int32))
    block_table = torch.zeros((batch_size, max_pages), device='npu', dtype=torch.int32)
    offset = 0
    for request, count in enumerate(pages_per_request.cpu().tolist()):
        block_table[request, :count] = reorder_paged_mqa_pages(
            pages[offset:offset + count], page_order)
        offset += count
    block_table = block_table.repeat_interleave(request_tokens, 0)

    def make_metadata():
        return deep_gemm.get_paged_mqa_logits_metadata(context_lens, num_heads, indices)

    def kernel(metadata):
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            (q, q_sf), fused, weights, context_lens, block_table, metadata,
            max_context_len, indices, use_ascend_kv_layout)

    if collect is not None:
        collect.append(lambda: kernel(make_metadata()))
        return

    reference = ref_paged_mqa_logits(q_ref, kv_ref, weights, context_lens, block_table, max_context_len)
    metadata = make_metadata()
    out = kernel(metadata)
    assert out.stride(0) * out.element_size() % 512 == 0 and out.stride(1) == 1
    valid = torch.arange(max_context_len, device='npu')[None] < context_lens.view(-1, 1)
    diff = calc_diff(out.float().masked_fill(~valid, 0), reference.masked_fill(~valid, 0))
    tolerance = 5e-4
    assert out.shape == reference.shape and out.dtype == torch.bfloat16
    assert diff < tolerance, f'paged MQA simulated diff {diff:.3e} exceeds {tolerance:.1e}'
    assert torch.equal(kernel(metadata).masked_fill(~valid, 0), out.masked_fill(~valid, 0))

    profile = 'random' if random_lens else 'exact'
    tag = (f'Profile={profile:6}, Fmt={"fp4" if use_fp4 else "fp8":3}, Page={page_size:3}, '
           f'AscendKV={str(use_ascend_kv_layout):5}, Order={page_order:9}, BSZ={batch_size:3}, '
           f'{request_tag:8}, H={num_heads:2}, D={head_dim:3}')
    if skip_prof:
        print(f' > {tag}: OK')
        return
    prof = bench_msprof(lambda: kernel(metadata), kernel_names='paged_mqa_logits_impl')
    meta_prof = bench_msprof(make_metadata, kernel_names='paged_mqa_logits_metadata_simt_impl', flush_l2=False)
    accessed = int(context_lens.sum())
    unique_kv = int(request_context_lens.sum())
    flops = 2.0 * accessed * num_heads * head_dim
    # KV traffic is the unique rows per request, output writes one row per valid (query, context) pair
    kv_row_bytes = data_bytes + sf_bytes
    bytes_moved = (count_bytes(q, q_sf, weights, context_lens, block_table, indices) +
                   unique_kv * kv_row_bytes + accessed * out.element_size())
    print(f' > Dev={torch.npu.current_device():1}, {tag}, L={avg_kv:4}: '
          f'{prof.tflops(flops):6.1f} TFLOPS, {prof.dur_us:6.1f} us, '
          f'{prof.gbps(bytes_moved):5.0f} GB/s, '
          f'Metadata={meta_prof.dur_us:4.1f} us | {format_pipe_profile(prof)}',
          flush=True)


def test_paged_mqa_logits(skip_prof, *, num_heads, head_dims, use_fp4,
                          min_tpr, max_tpr, page_sizes, use_ascend_kv_layout):
    assert 1 <= min_tpr <= max_tpr
    use_ascend_kv_layout = use_ascend_kv_layout if isinstance(use_ascend_kv_layout, tuple) else (use_ascend_kv_layout,)
    cases = []
    for batch_size, heads, head_dim, avg_kv, fp4, page_size, ascend_kv_layout in itertools.product(
            (256,), num_heads, head_dims, (8192,), use_fp4, page_sizes, use_ascend_kv_layout):
        request_tokens = tuple(torch.randint(min_tpr, max_tpr + 1, (batch_size,)).tolist())
        cases.append((batch_size, heads, head_dim, avg_kv, fp4, request_tokens, page_size, ascend_kv_layout))
    limit = int(os.getenv('DG_MQA_NUM_CASES', '0'))
    if limit:
        cases = random.Random(100000).sample(cases, min(limit, len(cases)))

    def run_all(collect=None):
        for case in cases:
            run_paged_mqa_logits_case(*case, skip_prof=skip_prof, random_lens=True, collect=collect)

    print(f'Testing MX Paged MQA Logits ({len(cases)} cases):')
    compile_mqa_kernels(run_all)
    run_all()
    print()


def run_paged_sparse_mqa_logits_perf_case(kv_len, skip_prof):
    num_requests, num_heads, head_dim = 256, 32, 128
    num_sparse_blocks, sparse_block_kv, page_kv, num_layers = 2048, 8, 128, 4
    num_candidates = num_sparse_blocks * sparse_block_kv
    rng = random.Random(torch.initial_seed())
    request_sizes = tuple(rng.randint(3, 7) for _ in range(num_requests))
    num_q_tokens = sum(request_sizes)
    num_pages_per_request = ceil_div(kv_len + max(request_sizes) - 1, page_kv)
    num_pages = num_requests * num_pages_per_request

    q_bf16 = torch.zeros((num_q_tokens, 1, num_heads, head_dim), dtype=torch.bfloat16, device='npu')
    weights = torch.zeros((num_q_tokens, num_heads), dtype=torch.bfloat16, device='npu')
    for head_idx in range(8):
        q_bf16[:, 0, head_idx, head_idx // 2 * 32 + head_idx % 2] = 1
        weights[:, head_idx] = 1 << head_idx
    q, sf_q, q_ref = quantize_mqa_tensor(q_bf16, True)

    physical_token = (torch.arange(num_pages, dtype=torch.int32, device='npu')[:, None] * page_kv +
                      torch.arange(page_kv, dtype=torch.int32, device='npu')[None])
    kv_data = torch.zeros((num_pages, page_kv, head_dim // 2), dtype=torch.uint8, device='npu')
    kv_sf_bytes = torch.empty((num_pages, page_kv, head_dim // 32), dtype=torch.uint8, device='npu')
    for group_idx in range(head_dim // 32):
        low = 1 + (physical_token + group_idx * 3) % 7
        high = 1 + (physical_token * 3 + group_idx * 5) % 7
        kv_data[..., group_idx * 16] = (low | (high << 4)).to(torch.uint8)
        kv_sf_bytes[..., group_idx] = (123 + (physical_token * 3 + group_idx) % 4).to(torch.uint8)
    fused_kv = empty_kv_cache((num_pages, page_kv, head_dim // 2 + head_dim // 32))
    fused_pages = fused_kv.view(num_pages, -1)
    fused_pages[:, :page_kv * (head_dim // 2)].copy_(kv_data.view(num_pages, -1))
    fused_pages[:, page_kv * (head_dim // 2):].copy_(kv_sf_bytes.view(num_pages, -1))
    fused_kv = fused_kv.unsqueeze(2)
    del q_bf16, kv_data, kv_sf_bytes, physical_token

    request_ends = list(itertools.accumulate(request_sizes))
    request_starts = [0, *request_ends[:-1]]
    indices_host = [request_idx for request_idx, size in enumerate(request_sizes) for _ in range(size)]
    context_lens_host = [kv_len + offset for size in request_sizes for offset in range(size)]
    context_lens = torch.tensor(context_lens_host, dtype=torch.int32, device='npu')
    indices = torch.tensor(indices_host, dtype=torch.int32, device='npu')
    page_order = torch.randperm(num_pages, dtype=torch.int32, device='npu')
    request_table = page_order.view(num_requests, num_pages_per_request)
    block_table = request_table[indices.long()]
    sparse_indices, _ = make_sparse_kv_block_indices(
        context_lens_host, indices_host, sparse_block_kv, num_sparse_blocks, seed=0)

    def make_metadata():
        return deep_gemm.get_paged_sparse_mqa_logits_metadata(
            context_lens, block_table, indices, page_kv, sparse_indices, q.dtype, sparse_block_kv)

    def run_sparse(metadata):
        return deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
            (q, sf_q), fused_kv, weights, metadata, num_sparse_blocks, sparse_block_kv)

    metadata = make_metadata()
    logits = run_sparse(metadata)
    assert logits.shape == (num_q_tokens, num_candidates) and logits.dtype == torch.bfloat16

    probe_q = torch.tensor((request_starts[0], request_ends[0] - 1,
                            request_starts[1], request_ends[1] - 1,
                            request_starts[-1], request_ends[-1] - 1),
                           dtype=torch.long, device='npu')
    probe_pos = torch.tensor((0, 1, 7, 8, 15, 16, 127, 128, 2047, 2048, 16383),
                             dtype=torch.long, device='npu')
    block_slot = torch.div(probe_pos, sparse_block_kv, rounding_mode='floor')
    row_in_block = probe_pos % sparse_block_kv
    logical_token = sparse_indices[probe_q[:, None], block_slot[None]] * sparse_block_kv + row_in_block
    logical_page = torch.div(logical_token, page_kv, rounding_mode='floor')
    physical_page = block_table[probe_q[:, None], logical_page]
    physical_row = logical_token % page_kv
    physical_token = physical_page * page_kv + physical_row

    fp4_values = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device='npu')
    expected = torch.zeros_like(physical_token, dtype=torch.float32)
    for head_idx in range(8):
        group_idx, half_idx = head_idx // 2, head_idx % 2
        if half_idx == 0:
            code = 1 + (physical_token + group_idx * 3) % 7
        else:
            code = 1 + (physical_token * 3 + group_idx * 5) % 7
        sf_exp = 123 + (physical_token * 3 + group_idx) % 4
        sf = (sf_exp.to(torch.int32) << 23).view(torch.float32)
        q_value = q_ref[probe_q, 0, head_idx, group_idx * 32 + half_idx].float()[:, None]
        expected += q_value * fp4_values[code.long()] * sf * weights[probe_q, head_idx].float()[:, None]
    actual = logits[probe_q[:, None], probe_pos[None]]
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)

    target_page = physical_page[0, 0].long()
    target_offset = physical_row[0, 0].long() * (head_dim // 2) + 3 * 16
    fused_pages[target_page, target_offset] = 0
    stale_logits = run_sparse(metadata)
    assert stale_logits[probe_q[0], probe_pos[0]] == actual[0, 0], 'materialized cache changed after source KV mutation'
    fresh_metadata = make_metadata()
    fresh_logits = run_sparse(fresh_metadata)
    assert fresh_logits[probe_q[0], probe_pos[0]] != actual[0, 0], 'refreshed metadata ignored source KV mutation'
    metadata = fresh_metadata
    del logits, stale_logits, fresh_logits

    if skip_prof:
        print(f' > Random, Requests={num_requests}, T/R=3-7, Q={num_q_tokens}, '
              f'K={kv_len}, H={num_heads}, D={head_dim}, S={num_sparse_blocks}, '
              f'B={sparse_block_kv}, Retain=80%, R={num_layers}: OK')
        return

    metadata_prof = bench_msprof(make_metadata, kernel_names='paged_sparse_mqa_prepare_impl')
    assert metadata_prof is not None, 'missing paged sparse prepare kernel'
    first_prof = bench_msprof(lambda: run_sparse(make_metadata()), kernel_names='paged_sparse_mqa_logits_impl')
    reuse_prof = bench_msprof(lambda: run_sparse(metadata), kernel_names='paged_sparse_mqa_logits_impl')
    for name, prof in zip(('prepare', 'first-main'), (metadata_prof, first_prof)):
        print(f' > K={kv_len}, {name}: {prof.dur_us:.1f} us | {format_pipe_profile(prof)}', flush=True)
    close_persistent_profiler()

    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    # Flush once before each prepare + FIRST + 3 REUSE chain.
    chain_us = []
    for iteration in range(40):
        flush_l2_cache()
        start.record()
        chain_metadata = make_metadata()
        for _ in range(num_layers):
            run_sparse(chain_metadata)
        end.record()
        end.synchronize()
        if iteration >= 10:
            chain_us.append(start.elapsed_time(end) * 1000)
        del chain_metadata

    flops = 2.0 * num_q_tokens * num_heads * head_dim * num_candidates
    r4_us = median(chain_us)
    print(f' > Random, Requests={num_requests}, T/R=3-7, Q={num_q_tokens}, '
          f'K={kv_len}, H={num_heads}, D={head_dim}, S={num_sparse_blocks}, '
          f'B={sparse_block_kv}, Retain=80%, R={num_layers}: '
          f'metadata={metadata_prof.dur_us:.1f} us, '
          f'first-main={first_prof.dur_us:.1f} us ({first_prof.tflops(flops):.1f} TFLOPS), '
          f'reuse-main={reuse_prof.dur_us:.1f} us ({reuse_prof.tflops(flops):.1f} TFLOPS), '
          f'R4={r4_us:.1f} us ({num_layers * flops / r4_us / 1e6:.1f} TFLOPS), '
          f'{format_pipe_profile(reuse_prof)}', flush=True)


def test_paged_sparse_mqa_logits_perf(skip_prof):
    print('Testing Paged sparse MQA logits production case:')
    run_paged_sparse_mqa_logits_perf_case(64 * 1024, skip_prof)
    gc.collect()
    torch.npu.empty_cache()
    print()


def enumerate_cuda_sparse_mqa_logits():
    avg_kv_lens = (4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024)
    num_sms = deep_gemm.get_num_sms()
    for use_fp4 in (True, False):
        for capacity, kv in itertools.product((2048, 1024, 512), avg_kv_lens):
            for unaligned in (False, True):
                yield use_fp4, False, 8192, kv, 8, capacity, unaligned
            yield use_fp4, True, 512, kv, 8, capacity, False
        for block, unaligned in itertools.product((8, 16), (False, True)):
            yield use_fp4, False, 9, 640 if use_fp4 else 512, block, 128, unaligned

    # B16 capacity is capped at 4095 by Ascend's 16-bit output offsets.
    for case in (
        (1, 1, 16, 4), (2, 639, 16, 1024), (2, 0, 16, 4),
        (3, 640, 16, 1024), (5, 641, 16, 1024),
        (num_sms - 1, 16 * 1024 + 3, 16, 1024),
        (num_sms, 16 * 1024 + 3, 16, 1024),
        (num_sms + 1, 16 * 1024 + 3, 16, 1024),
        (2 * num_sms + 1, 64 * 1024 + 7, 16, 4095),
    ):
        for unaligned in (False, True):
            yield True, False, *case, unaligned
    for case in (
        (1, 64, 16, 8), (3, 64, 16, 8),
        (num_sms - 1, 16 * 1024 + 3, 16, 1024),
        (num_sms + 1, 16 * 1024 + 3, 16, 1024),
        (2 * num_sms + 1, 16 * 1024 + 3, 16, 1024),
        (3, 1024 * 1024, 8, 2048),
    ):
        yield True, True, *case, False
    for unaligned in (False, True):
        yield False, False, 3, 640, 16, 1024, unaligned
    yield False, True, 3, 64, 16, 8, False


def make_sparse_kv_block_indices(context_lens, indices, sparse_block_kv,
                                 num_max_sparse_blocks, seed, context_starts=None, overlap=0.8):
    assert 0 <= overlap <= 1
    rng = random.Random(seed)
    indices = [0] * len(context_lens) if indices is None else indices
    context_starts = [0] * len(context_lens) if context_starts is None else context_starts
    sparse_indices, num_blocks_per_q = [], []
    previous_blocks, previous_request_idx = None, None
    for context_start, context_len, request_idx in zip(context_starts, context_lens, indices):
        first_block = context_start // sparse_block_kv
        num_available_blocks = ceil_div(max(0, context_len - context_start), sparse_block_kv)
        block_end = first_block + num_available_blocks
        num_sparse_blocks = min(num_available_blocks, num_max_sparse_blocks)
        if num_sparse_blocks == num_available_blocks:
            blocks = list(range(first_block, block_end))
        elif request_idx != previous_request_idx:
            blocks = rng.sample(range(first_block, block_end), num_sparse_blocks)
        else:
            previous = [block_idx for block_idx in previous_blocks if first_block <= block_idx < block_end]
            retained = rng.sample(previous, min(round(num_sparse_blocks * overlap), len(previous)))
            retained_set = set(retained)
            replacements = set()
            while len(retained) + len(replacements) < num_sparse_blocks:
                block_idx = rng.randrange(first_block, block_end)
                if block_idx not in retained_set:
                    replacements.add(block_idx)
            blocks = retained + list(replacements)
        blocks.sort()
        sparse_indices.append(blocks + [blocks[-1] if blocks else 0] * (num_max_sparse_blocks - num_sparse_blocks))
        num_blocks_per_q.append(num_sparse_blocks)
        previous_blocks, previous_request_idx = blocks, request_idx
    return torch.tensor(sparse_indices, device='npu', dtype=torch.int32), num_blocks_per_q


def make_random_paged_mqa_kv(num_pages, page_kv, head_dim, use_fp4):
    """Build MQA pages with DATA followed by SF."""
    data_bytes = head_dim // 2 if use_fp4 else head_dim
    token_bytes = data_bytes + head_dim // 32
    page_bytes = page_kv * token_bytes
    page_stride_bytes = ceil_div(page_bytes, 512) * 512
    storage = empty_kv_cache((num_pages, page_stride_bytes))
    for page_begin in range(0, num_pages, 16 * 1024):
        page_end = min(page_begin + 16 * 1024, num_pages)
        kv, kv_sf, kv_ref = quantize_mqa_tensor(torch.randn(
            (page_end - page_begin, page_kv, 1, head_dim), device='npu', dtype=torch.bfloat16), use_fp4)
        pages = storage[page_begin:page_end, :page_bytes]
        pages[:, :page_kv * data_bytes].copy_(kv.contiguous().view(torch.uint8).view(page_end - page_begin, -1))
        pages[:, page_kv * data_bytes:].copy_(kv_sf.contiguous().view(torch.uint8).view(page_end - page_begin, -1))
        del kv, kv_sf, kv_ref
    return storage.as_strided((num_pages, page_kv, 1, token_bytes),
                              (page_stride_bytes, token_bytes, token_bytes, 1))


def run_sparse_mqa_logits_case(num_heads, use_fp4, is_paged, num_q_tokens, avg_kv_len,
                               sparse_block_kv, num_max_sparse_blocks, skip_prof,
                               num_repeats=2, page_kv=64, request_sizes=None, overlap=0.8,
                               compile_only=False, check_snapshot=False, use_unaligned_ks=False, weights_padding=0):
    assert not (is_paged and use_unaligned_ks)
    head_dim = 128
    rng = random.Random(num_q_tokens * 1000000 + avg_kv_len + sparse_block_kv)
    if request_sizes is not None:
        assert sum(request_sizes) == num_q_tokens
    elif is_paged:
        request_sizes = []
        remaining_q_tokens = num_q_tokens
        while remaining_q_tokens > 0:
            request_size = min(rng.randint(2, 6), remaining_q_tokens)
            request_sizes.append(request_size)
            remaining_q_tokens -= request_size
    else:
        num_requests = min(rng.randint(2, 4), num_q_tokens)
        request_ends = sorted(rng.sample(range(1, num_q_tokens), num_requests - 1)) + [num_q_tokens]
        request_sizes = [request_end - request_begin
                         for request_begin, request_end in zip([0] + request_ends, request_ends)]
    indices_host = [request_idx for request_idx, request_size in enumerate(request_sizes)
                    for _ in range(request_size)]

    if is_paged:
        context_starts = [0] * num_q_tokens
        batch_size = len(request_sizes)
        request_context_lens = [rng.randint(int(0.7 * avg_kv_len) // sparse_block_kv,
                                            int(1.3 * avg_kv_len) // sparse_block_kv) * sparse_block_kv
                                for _ in range(batch_size)]
        context_lens = [context_len + q_offset
                        for context_len, request_size in zip(request_context_lens, request_sizes)
                        for q_offset in range(request_size)]
    else:
        context_starts, context_lengths = [], []
        num_kv_tokens = 0
        for request_size in request_sizes:
            start = ceil_div(num_kv_tokens, sparse_block_kv) * sparse_block_kv
            if use_unaligned_ks:
                start += rng.randrange(1, sparse_block_kv)
            context_starts.extend([start] * request_size)
            context_lengths.extend(avg_kv_len + q_offset for q_offset in range(request_size))
            num_kv_tokens = start + context_lengths[-1]
        context_lens = [start + length for start, length in zip(context_starts, context_lengths)]
        num_kv_tokens = max(1, num_kv_tokens)

    if is_paged:
        max_context_lens = [context_len + request_size - 1
                            for context_len, request_size in zip(request_context_lens, request_sizes)]
        num_pages_per_request = [ceil_div(context_len, page_kv) for context_len in max_context_lens]
        max_num_pages, num_pages = max(num_pages_per_request), sum(num_pages_per_request)

    if compile_only:
        assert deep_gemm._C.get_dry_run(), 'compile_only requires dry_run'
        dtype = torch.int8 if use_fp4 else torch.float8_e4m3fn
        data_bytes = head_dim // 2 if use_fp4 else head_dim
        q_shape = (num_q_tokens, 1, num_heads) if is_paged else (num_q_tokens, num_heads)
        q = (torch.empty((*q_shape, data_bytes), dtype=dtype, device='meta'),
             torch.empty((*q_shape, 2), dtype=torch.int16, device='meta'))
        weights = torch.empty_strided((num_q_tokens, num_heads), (num_heads + weights_padding, 1),
                                      dtype=torch.bfloat16, device='meta')
        bounds = torch.empty(num_q_tokens, dtype=torch.int32, device='meta')
        sparse_indices = torch.empty((num_q_tokens, num_max_sparse_blocks), dtype=torch.int32, device='meta')
        if is_paged:
            token_bytes = data_bytes + head_dim // 32
            page_stride = ceil_div(page_kv * token_bytes, 512) * 512
            kv = torch.empty_strided((num_pages, page_kv, 1, token_bytes),
                                     (page_stride, token_bytes, token_bytes, 1), dtype=torch.uint8, device='meta')
            block_table = torch.empty((num_q_tokens, max_num_pages), dtype=torch.int32, device='meta')
            metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
                bounds, block_table, bounds, page_kv, sparse_indices, dtype, sparse_block_kv)
            deep_gemm.fp8_fp4_paged_sparse_mqa_logits(q, kv, weights, metadata, num_max_sparse_blocks, sparse_block_kv)
            contexts = bounds.view(-1, 1)
            schedule = deep_gemm.get_paged_mqa_logits_metadata(contexts, num_heads, bounds)
            deep_gemm.fp8_fp4_paged_mqa_logits(
                q, kv, weights.contiguous(), contexts, block_table, schedule, max(context_lens), bounds)
        else:
            kv = (torch.empty((num_kv_tokens, data_bytes), dtype=dtype, device='meta'),
                  torch.empty((num_kv_tokens, 2), dtype=torch.int16, device='meta'))
            metadata = deep_gemm.get_sparse_mqa_logits_metadata(
                bounds, bounds, num_kv_tokens, sparse_indices, dtype, sparse_block_kv)
            deep_gemm.fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, num_max_sparse_blocks, sparse_block_kv)
            deep_gemm.fp8_fp4_mqa_logits(q, kv, weights, bounds, bounds,
                max(1, max(end - start for start, end in zip(context_starts, context_lens))))
        return

    close_persistent_profiler()
    q_shape = (num_q_tokens, 1, num_heads, head_dim) if is_paged else (num_q_tokens, num_heads, head_dim)
    q, q_sf, q_ref = quantize_mqa_tensor(torch.randn(q_shape, device='npu', dtype=torch.bfloat16), use_fp4)
    del q_ref
    weights = torch.randn((num_q_tokens, num_heads + weights_padding), device='npu', dtype=torch.bfloat16)[:, :num_heads]
    sparse_indices, num_sparse_blocks = make_sparse_kv_block_indices(
        context_lens, indices_host, sparse_block_kv, num_max_sparse_blocks,
        avg_kv_len, context_starts, overlap)

    if is_paged:
        kv_cache = make_random_paged_mqa_kv(num_pages, page_kv, head_dim, use_fp4)
        indices = torch.tensor(indices_host, device='npu', dtype=torch.int32)
        context_lens_tensor = torch.tensor(context_lens, device='npu', dtype=torch.int32)
        page_pool = torch.randperm(num_pages, device='npu', dtype=torch.int32)
        request_block_table = torch.zeros((batch_size, max_num_pages), device='npu', dtype=torch.int32)
        page_begin = 0
        for request_idx, num_request_pages in enumerate(num_pages_per_request):
            page_end = page_begin + num_request_pages
            request_block_table[request_idx, :num_request_pages] = page_pool[page_begin:page_end]
            page_begin = page_end
        block_table = request_block_table[indices.long()].contiguous()
        metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
            context_lens_tensor, block_table, indices, page_kv, sparse_indices,
            q.dtype, sparse_block_kv)
        sparse_kwargs = dict(q=(q, q_sf), kv_cache=kv_cache, weights=weights, metadata=metadata,
                             num_max_sparse_blocks=num_max_sparse_blocks, sparse_block_kv=sparse_block_kv)
        context_lens_2d = context_lens_tensor.view(-1, 1)
        full_kwargs = dict(
            q=(q, q_sf), kv_cache=kv_cache, weights=weights.contiguous(), context_lens=context_lens_2d,
            block_table=block_table,
            schedule_meta=deep_gemm.get_paged_mqa_logits_metadata(context_lens_2d, num_heads, indices),
            max_context_len=max(context_lens), indices=indices)
        run_sparse = lambda: deep_gemm.fp8_fp4_paged_sparse_mqa_logits(**sparse_kwargs)
        run_full = lambda: deep_gemm.fp8_fp4_paged_mqa_logits(**full_kwargs)
        sparse_kernel_name = 'paged_sparse_mqa_logits_impl'
        full_kernel_name = 'paged_mqa_logits_impl'
        case = f'Paged BSZ={batch_size:3}, SQ={num_q_tokens:4}'
    else:
        kv, kv_sf, kv_ref = quantize_mqa_tensor(
            torch.randn((num_kv_tokens, head_dim), device='npu', dtype=torch.bfloat16), use_fp4)
        kv, kv_sf = [empty_kv_cache(t.shape, dtype=t.dtype).copy_(t) for t in (kv, kv_sf)]
        del kv_ref
        starts = torch.tensor(context_starts, device='npu', dtype=torch.int32)
        ends = torch.tensor(context_lens, device='npu', dtype=torch.int32)
        metadata = deep_gemm.get_sparse_mqa_logits_metadata(
            starts, ends, num_kv_tokens, sparse_indices, q.dtype, sparse_block_kv)
        sparse_kwargs = dict(q=(q, q_sf), kv=(kv, kv_sf), weights=weights, metadata=metadata,
                             num_max_sparse_blocks=num_max_sparse_blocks, sparse_block_kv=sparse_block_kv)
        full_kwargs = dict(q=(q, q_sf), kv=(kv, kv_sf), weights=weights,
                           cu_seq_len_k_start=starts, cu_seq_len_k_end=ends,
                           max_seqlen_k=max(1, max(end - start for start, end in zip(context_starts, context_lens))))
        run_sparse = lambda: deep_gemm.fp8_fp4_sparse_mqa_logits(**sparse_kwargs)
        run_full = lambda: deep_gemm.fp8_fp4_mqa_logits(**full_kwargs)
        sparse_kernel_name, full_kernel_name = 'sparse_mqa_logits_impl', 'mqa_logits_impl'
        case = f'SQ={num_q_tokens:4}, Overlap={overlap:.0%}, UnalignedKS={use_unaligned_ks}'

    if check_snapshot:
        assert is_paged and skip_prof
        full_logits = run_full()
        inputs = (context_lens_tensor, block_table, indices, sparse_indices)
        saved = [value.clone() for value in inputs]
        for value in inputs:
            value.fill_(-1)
        with dry_run():
            run_sparse()
        sparse_logits = run_sparse()
        for value, original in zip(inputs, saved):
            value.copy_(original)
        kv_cache.zero_()
    else:
        sparse_logits, full_logits = run_sparse(), run_full()
    assert sparse_logits.shape == (num_q_tokens, num_max_sparse_blocks * sparse_block_kv)
    assert sparse_logits.dtype == torch.bfloat16
    assert sparse_logits.stride(0) * sparse_logits.element_size() % 512 == 0 and sparse_logits.stride(1) == 1
    kv_offsets = torch.arange(sparse_block_kv, device='npu', dtype=torch.int32)
    context_starts_tensor = torch.tensor(context_starts, device='npu', dtype=torch.int32)
    token_indices = (sparse_indices.unsqueeze(-1) * sparse_block_kv +
                     (context_starts_tensor % sparse_block_kv)[:, None, None] + kv_offsets).flatten(1).long()
    num_sparse_blocks_tensor = torch.tensor(num_sparse_blocks, device='npu')
    num_sparse_tokens = num_sparse_blocks_tensor * sparse_block_kv
    valid_mask = torch.arange(token_indices.size(1), device='npu')[None] < num_sparse_tokens[:, None]
    valid_mask &= token_indices >= context_starts_tensor[:, None]
    valid_mask &= token_indices < torch.tensor(context_lens, device='npu')[:, None]
    sparse_valid = sparse_logits[valid_mask]
    full_token_indices = token_indices - context_starts_tensor[:, None]
    full_valid = full_logits.gather(
        1, full_token_indices.clamp(min=0, max=full_logits.size(1) - 1))[valid_mask]
    if sparse_valid.numel():
        if is_paged or not use_fp4:
            assert torch.equal(sparse_valid, full_valid), 'sparse MQA changed BF16 logits bits'
        else:
            diff = calc_diff(sparse_valid.float(), full_valid.float())
            assert diff < 0.001, f'sparse MQA logits mismatch: {diff=:.5f}'
    for _ in range(num_repeats):
        assert torch.equal(run_sparse()[valid_mask], sparse_valid), 'sparse MQA logits self-consistency mismatch'

    fmt = 'mxfp4' if use_fp4 else 'mxfp8'
    tag = (f'Fmt={fmt:5}, H={num_heads:2}, {case}, KV={avg_kv_len:7}, '
           f'SPARSE_BLOCK_KV={sparse_block_kv:2}, MAX_BLOCKS={num_max_sparse_blocks:4}')
    full_selection = all(count == ceil_div(end - start, sparse_block_kv)
                         for count, start, end in zip(num_sparse_blocks, context_starts, context_lens))
    if skip_prof:
        print(f' > {tag}: OK')
    elif full_selection:
        print(f' > {tag}: OK (full selection; correctness only)')
    else:
        sparse_prof = bench_msprof(run_sparse, kernel_names=sparse_kernel_name)
        full_prof = bench_msprof(run_full, kernel_names=full_kernel_name)
        num_sparse_tokens_total = int(num_sparse_blocks_tensor.sum().item()) * sparse_block_kv
        flops = 2.0 * num_sparse_tokens_total * num_heads * head_dim
        print(f' > {tag}: sparse {sparse_prof.dur_us:6.1f} us, '
              f'{sparse_prof.tflops(flops):6.1f} TFLOPS, '
              f'full {full_prof.dur_us:6.1f} us, '
              f'{full_prof.dur_ns / sparse_prof.dur_ns:5.2f}x | '
              f'{format_pipe_profile(sparse_prof)}', flush=True)
    del sparse_logits, full_logits
    gc.collect()
    torch.npu.empty_cache()


def test_sparse_mqa_logits_perf(skip_prof):
    print('Testing sparse MQA logits production cases:')
    for overlap in SPARSE_MQA_PREFILL_OVERLAPS:
        for avg_kv_len in SPARSE_MQA_PREFILL_KV_LENS:
            run_sparse_mqa_logits_case(
                32, True, False, sum(SPARSE_MQA_PREFILL_REQUEST_TOKENS), avg_kv_len,
                8, 2048, skip_prof, num_repeats=1,
                request_sizes=SPARSE_MQA_PREFILL_REQUEST_TOKENS, overlap=overlap)
    print()


def test_paged_sparse_mqa_metadata_contract(num_q_tokens=3):
    num_heads, num_max_sparse_blocks, sparse_block_kv, page_kv = 4, 32, 8, 64
    q = torch.full((num_q_tokens, 1, num_heads, 64), 0x22, dtype=torch.int8, device='npu')
    sf_q = torch.full((num_q_tokens, 1, num_heads, 2), 0x7f7f, dtype=torch.int16, device='npu')
    weights = torch.ones((num_q_tokens, num_heads), dtype=torch.bfloat16, device='npu')
    kv = empty_kv_cache((2, page_kv, 1, 68)).zero_()
    kv.view(2, -1)[:, page_kv * 64:] = 127
    contexts = torch.full((num_q_tokens,), 128, dtype=torch.int32, device='npu')
    requests = torch.zeros(num_q_tokens, dtype=torch.int32, device='npu')
    table = torch.tensor([[0, 1]] * num_q_tokens, dtype=torch.int32, device='npu')
    rows = torch.arange(num_max_sparse_blocks, dtype=torch.int32, device='npu').repeat(num_q_tokens, 1)
    expect_error(TypeError, lambda: deep_gemm.get_paged_sparse_mqa_logits_metadata(
        contexts, table, requests, page_kv, rows, q.dtype, sparse_block_kv, use_mqa_kv_layout=True),
        'use_mqa_kv_layout')
    expect_error(RuntimeError, lambda: deep_gemm.get_paged_sparse_mqa_logits_metadata(
        contexts[:0], table[:0], requests[:0], page_kv, rows[:0], q.dtype, sparse_block_kv), 'size(0) > 0')
    make_metadata = lambda: deep_gemm.get_paged_sparse_mqa_logits_metadata(
        contexts, table, requests, page_kv, rows, q.dtype, sparse_block_kv)
    num_ai_cores = deep_gemm.get_num_sms()
    metadata = make_metadata()
    assert isinstance(metadata, torch.Tensor)
    metadata_bytes = metadata.numel()
    full_rows = torch.arange(16384 // sparse_block_kv, dtype=torch.int32, device='npu').repeat(num_q_tokens, 1)
    full_metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
        contexts, table, requests, page_kv, full_rows, q.dtype, sparse_block_kv)
    # Per-core boundary reserves are fixed, so small Q counts need not scale proportionally.
    assert metadata_bytes < full_metadata.numel(), 'small sparse capacity retained full-size KV storage'
    del full_metadata, full_rows
    kwargs = dict(q=(q, sf_q), kv_cache=kv, weights=weights, metadata=metadata,
                  num_max_sparse_blocks=num_max_sparse_blocks, sparse_block_kv=sparse_block_kv)

    def run(**overrides):
        return deep_gemm.fp8_fp4_paged_sparse_mqa_logits(**(kwargs | overrides))

    invalid = [
        dict(num_max_sparse_blocks=16),
        dict(sparse_block_kv=16),
        dict(kv_cache=kv.view(1, 128, 1, 68)),
        dict(q=(q[:2], sf_q[:2]), weights=weights[:2]),
        dict(q=(torch.zeros((num_q_tokens, 1, num_heads, 128), dtype=torch.uint8, device='npu').view(torch.float8_e4m3fn), sf_q),
             kv_cache=empty_kv_cache((2, page_kv, 1, 132)).zero_()),
    ]
    for override in invalid:
        expect_error(RuntimeError, partial(run, **override), 'metadata')
    expect_error(TypeError, partial(run, metadata=torch.empty_like(metadata)), 'metadata')
    try:
        deep_gemm.set_num_sms(1 if num_ai_cores != 1 else 2)
        expect_error(RuntimeError, run, 'metadata')
    finally:
        deep_gemm.set_num_sms(num_ai_cores)
    with dry_run():
        run()
    kv.view(2, -1)[:, :page_kv * 64] = 0x22
    first = run()
    assert torch.all(first[:, :128] == 512)
    with dry_run():
        run()
    expect_error(RuntimeError, partial(run, num_max_sparse_blocks=16))
    kv.view(2, -1)[:, :page_kv * 64] = 0
    weights.mul_(2)
    reused = run()
    assert torch.all(reused[:, :128] == 1024), 'REUSE reread KV or cached weights'
    fresh = run(metadata=make_metadata())
    assert torch.all(fresh[:, :128] == 0), 'new metadata did not refresh KV'

    # Check allocation arithmetic above 4 GiB without launching on uninitialized inputs.
    num_large_queries, capacity, block_kv = 2048, 2048, 8
    large_contexts = torch.empty(num_large_queries, dtype=torch.int32, device='meta')
    large_table = torch.empty((num_large_queries, 1), dtype=torch.int32, device='meta')
    large_rows = torch.empty((num_large_queries, capacity), dtype=torch.int32, device='meta')
    with dry_run():
        large_metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
            large_contexts, large_table, large_contexts, page_kv, large_rows, torch.float8_e4m3fn, block_kv)
    assert large_metadata.is_meta
    assert large_metadata.numel() >= num_large_queries * capacity * block_kv * (128 + 4), 'compact KV allocation overflow'


def test_sparse_mqa_capacity_boundaries(use_fp4=True):
    num_queries = 4
    q, q_sf, _ = quantize_mqa_tensor(torch.ones((num_queries, 32, 128), device='npu', dtype=torch.bfloat16), use_fp4)
    weights = torch.ones((num_queries, 32), device='npu', dtype=torch.bfloat16)
    for block_kv, capacity in ((8, 4096), (16, 4095)):
        num_tokens = capacity * block_kv
        num_kv_tokens = num_queries * num_tokens
        values = (torch.arange(num_kv_tokens, device='npu') % 3 + 1).to(torch.bfloat16)
        kv, kv_sf, kv_ref = quantize_mqa_tensor(values[:, None].expand(-1, 128).contiguous(), use_fp4)
        kv, kv_sf = [empty_kv_cache(t.shape, dtype=t.dtype).copy_(t) for t in (kv, kv_sf)]
        # Disjoint queries leave empty KV splits after each earlier query's final output.
        rows = torch.arange(num_queries * capacity, dtype=torch.int32, device='npu').view(num_queries, capacity)
        starts = torch.zeros(num_queries, dtype=torch.int32, device='npu')
        ends = torch.full_like(starts, num_kv_tokens)
        metadata = deep_gemm.get_sparse_mqa_logits_metadata(starts, ends, num_kv_tokens, rows, q.dtype, block_kv)
        expected = (kv_ref[:, 0] * (32 * 128)).to(torch.bfloat16).view(num_queries, num_tokens)
        for _ in range(2):
            logits = deep_gemm.fp8_fp4_sparse_mqa_logits((q, q_sf), (kv, kv_sf), weights, metadata, capacity, block_kv)
            assert torch.equal(logits, expected), (use_fp4, block_kv, capacity)
        if block_kv == 16:
            invalid_rows = torch.full((num_queries, 4096), -1, dtype=torch.int32, device='npu')
            expect_error(RuntimeError, lambda: deep_gemm.get_sparse_mqa_logits_metadata(
                starts, ends, num_kv_tokens, invalid_rows, q.dtype, block_kv), '16-bit output offsets')
            expect_error(RuntimeError, lambda: deep_gemm.fp8_fp4_sparse_mqa_logits(
                (q, q_sf), (kv, kv_sf), weights, metadata, 4096, block_kv), '16-bit output offsets')
        print(f' > Sparse capacity fp4={use_fp4}, block={block_kv}, capacity={capacity}: OK')


def test_sparse_mqa_unaligned_ks():
    for block_kv in (8, 16):
        num_queries, capacity = 4 * block_kv + 3, 65
        num_kv_tokens = 129 * block_kv + 5
        starts = [q_idx % block_kv + (q_idx // block_kv % 2) * block_kv for q_idx in range(num_queries)]
        lengths = (0, 1, block_kv - 1, block_kv, block_kv + 1, 65 * block_kv - 1, 129 * block_kv)
        ends = [min(start + lengths[q_idx % len(lengths)], num_kv_tokens) for q_idx, start in enumerate(starts)]
        ends[1] = starts[1] - 1
        starts[-1], ends[-1] = num_kv_tokens - 2, num_kv_tokens + block_kv
        rows, tokens = [], []
        for q_idx, (start, end) in enumerate(zip(starts, ends)):
            end = min(end, num_kv_tokens)
            num_blocks = ceil_div(max(0, end - start), block_kv)
            available = list(range(start // block_kv, start // block_kv + num_blocks))
            selected = available if num_blocks <= capacity else sorted(
                random.Random(q_idx).sample(available[:-1], capacity - 1) + available[-1:])
            rows.append(selected + [-1] * (capacity - len(selected)))
            tokens.append([block * block_kv + start % block_kv + offset
                           for block in selected for offset in range(block_kv)
                           if block * block_kv + start % block_kv + offset < end])
        for use_fp4 in (True, False):
            q, q_sf, _ = quantize_mqa_tensor(torch.ones((num_queries, 32, 128), device='npu', dtype=torch.bfloat16), use_fp4)
            idx = torch.arange(num_kv_tokens, device='npu')
            values = torch.pow(2.0, idx % 9 - 4) * torch.where(idx % 3 == 0, -1, 1)
            kv, kv_sf, kv_ref = quantize_mqa_tensor(values[:, None].expand(-1, 128).to(torch.bfloat16).contiguous(), use_fp4)
            kv, kv_sf = [empty_kv_cache(t.shape, dtype=t.dtype).copy_(t) for t in (kv, kv_sf)]
            weights = torch.ones((num_queries, 32), device='npu', dtype=torch.bfloat16)
            metadata = deep_gemm.get_sparse_mqa_logits_metadata(
                torch.tensor(starts, device='npu', dtype=torch.int32),
                torch.tensor(ends, device='npu', dtype=torch.int32), num_kv_tokens,
                torch.tensor(rows, device='npu', dtype=torch.int32), q.dtype, block_kv)
            expected = [(kv_ref[torch.tensor(row, device='npu', dtype=torch.int64), 0].clamp_min(0) * (32 * 128)).to(torch.bfloat16)
                        for row in tokens]
            for _ in range(3):
                logits = deep_gemm.fp8_fp4_sparse_mqa_logits((q, q_sf), (kv, kv_sf), weights, metadata, capacity, block_kv)
                for q_idx, row in enumerate(expected):
                    assert torch.equal(logits[q_idx, :row.numel()], row), (use_fp4, block_kv, q_idx, starts[q_idx], ends[q_idx])
            print(f' > Unaligned KS fp4={use_fp4}, block={block_kv}, queries={num_queries}: OK')


def test_sparse_mqa_metadata_tensor():
    q, q_sf, _ = quantize_mqa_tensor(torch.ones((1, 32, 128), device='npu', dtype=torch.bfloat16), True)
    kv, kv_sf, _ = quantize_mqa_tensor(torch.ones((64, 128), device='npu', dtype=torch.bfloat16), True)
    weights = torch.ones((1, 32), dtype=torch.bfloat16, device='npu')
    starts = torch.zeros(1, dtype=torch.int32, device='npu')
    ends = torch.full_like(starts, 64)
    rows = torch.arange(8, dtype=torch.int32, device='npu')[None]
    expect_error(RuntimeError, lambda: deep_gemm.get_sparse_mqa_logits_metadata(
        starts, ends, 0, rows, q.dtype, 8), 'num_kv_tokens > 0')
    expect_error(RuntimeError, lambda: deep_gemm.get_sparse_mqa_logits_metadata(
        starts[:0], ends[:0], 64, rows[:0], q.dtype, 8), 'size(0) > 0')
    metadata = deep_gemm.get_sparse_mqa_logits_metadata(starts, ends, 64, rows, q.dtype, 8)
    # Metadata is an ordinary tensor; a copy with the same contents works across head counts.
    for plan in (metadata, metadata.clone()):
        for heads in (4, 32):
            call = partial(deep_gemm.fp8_fp4_sparse_mqa_logits,
                (q[:, :heads].contiguous(), q_sf[:, :heads].contiguous()), (kv, kv_sf),
                weights[:, :heads], plan, 8, 8)
            with dry_run():
                call()
            logits = call()
            assert torch.equal(logits, torch.full_like(logits, heads * 128))
    print(' > Sparse prefill metadata tensor: OK')


def test_sparse_mqa_meta_dry_run(run_contiguous=True, run_paged=True):
    runtime_calls = []

    def compile_case(dtype, block_kv, page_kv):
        num_queries, num_heads, capacity, num_kv_tokens = 3, 4, 32, 128
        data_bytes = 64 if dtype == torch.int8 else 128
        q_shape = (num_queries, 1, num_heads) if page_kv else (num_queries, num_heads)
        q = torch.empty((*q_shape, data_bytes), dtype=dtype, device='meta')
        sf_q = torch.empty((*q_shape, 2), dtype=torch.int16, device='meta')
        weights = torch.empty_strided((num_queries, num_heads), (8, 1), dtype=torch.bfloat16, device='meta')
        bounds = torch.empty(num_queries, dtype=torch.int32, device='meta')
        rows = torch.empty((num_queries, capacity), dtype=torch.int32, device='meta')
        if page_kv:
            table = torch.empty_strided((num_queries, 2), (4, 1), dtype=torch.int32, device='meta')
            prepare = partial(deep_gemm.get_paged_sparse_mqa_logits_metadata,
                              bounds, table, bounds, page_kv, qk_dtype=dtype, sparse_block_kv=block_kv)
            kv = torch.empty((2, page_kv, 1, data_bytes + 4), dtype=torch.uint8, device='meta')
            main = partial(deep_gemm.fp8_fp4_paged_sparse_mqa_logits, kv_cache=kv)
        else:
            prepare = partial(deep_gemm.get_sparse_mqa_logits_metadata,
                              bounds, bounds, num_kv_tokens, qk_dtype=dtype, sparse_block_kv=block_kv)
            kv = torch.empty((num_kv_tokens, data_bytes), dtype=dtype, device='meta')
            sf_kv = torch.empty((num_kv_tokens, 2), dtype=torch.int16, device='meta')
            main = partial(deep_gemm.fp8_fp4_sparse_mqa_logits, kv=(kv, sf_kv))
        metadata = prepare(sparse_kv_block_indices=rows)
        run = partial(main, q=(q, sf_q), weights=weights, metadata=metadata,
                      num_max_sparse_blocks=capacity, sparse_block_kv=block_kv)
        logits = run()
        assert metadata.is_meta and logits.is_meta and logits.dtype == torch.bfloat16
        assert logits.shape == (num_queries, capacity * block_kv)
        assert logits.stride() == (ceil_div(capacity * block_kv, 256) * 256, 1)
        if page_kv:
            config, is_first = metadata._paged_sparse_mqa_state
            assert is_first, 'dry-run consumed FIRST'
            metadata._paged_sparse_mqa_state = (config, False)
            run()
            assert metadata._paged_sparse_mqa_state == (config, False)

        # Invalid contracts must still fail before compilation, even without storage.
        for invalid_rows in (rows.to(torch.int64), rows[:, :-1], rows.view(-1)):
            expect_error(RuntimeError, partial(prepare, sparse_kv_block_indices=invalid_rows))
        expect_error(RuntimeError, partial(run, q=(q[..., :-1], sf_q)))
        expect_error(RuntimeError, partial(run, q=(q, sf_q.to(torch.int32))))
        invalid_metadata = metadata[:-1]
        if page_kv:
            invalid_metadata._paged_sparse_mqa_state = metadata._paged_sparse_mqa_state
        expect_error(RuntimeError, partial(run, metadata=invalid_metadata))
        invalid_weights = torch.empty_strided(weights.shape, (16, 2), dtype=weights.dtype, device='meta')
        expect_error(RuntimeError, partial(run, weights=invalid_weights))
        runtime_calls.extend((partial(prepare, sparse_kv_block_indices=rows), run))

    # Warm up the runtime before checking that compilation allocates no NPU tensor storage.
    torch.npu.current_stream()
    deep_gemm.get_num_sms()
    before_compile = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    pages = ([0] if run_contiguous else []) + ([64, 128] if run_paged else [])
    calls = [partial(compile_case, dtype, block, page)
             for dtype, block, page in itertools.product((torch.int8, torch.float8_e4m3fn), (8, 16), pages)]
    # Include the largest dense-reference outputs used by sparse_cuda/paged_sparse_cuda.
    calls += [partial(run_sparse_mqa_logits_case, 32, fp4, bool(page), 512 if page else 8192,
                      256 * 1024, 8, 2048, True, page_kv=page or 64, compile_only=True)
              for fp4, page in itertools.product((True, False), pages)]
    par_compile(calls, max_workers=128)
    assert torch.npu.max_memory_allocated() == before_compile
    assert torch.npu.memory_allocated() == before_compile
    assert not deep_gemm._C.get_dry_run()
    for call in runtime_calls:
        expect_error(RuntimeError, call, 'PrivateUse1')
    print(' > Sparse MQA meta dry-run, validation and device rejection: OK')


def test_sparse_mqa_logits_boundaries(run_contiguous=True, run_paged=False):
    test_sparse_mqa_meta_dry_run(run_contiguous, run_paged)
    # CUDA-style small-Q, empty-context, split-tail and block-size coverage.
    # Paged cases also check Ascend's FIRST/REUSE snapshot and padded weights.
    cases = (
        (32, True, False, 2, 0, 8, 4, False),
        (32, True, False, 1, 1, 16, 4, False),
        (12, True, False, 9, 640, 8, 128, True),
        (20, False, False, 9, 512, 16, 128, True),
        (8, True, False, 33, 1025, 16, 65, False),
        (32, False, False, 3, 640, 16, 1024, False),
        (32, True, True, 1, 64, 16, 8, False),
        (12, True, True, 7, 513, 8, 64, False),
        (20, False, True, 9, 512, 8, 128, False),
        (8, False, True, 3, 64, 16, 8, False),
    )
    calls = [partial(run_sparse_mqa_logits_case, heads, fp4, paged, nq, kv, block, capacity,
                     True, use_unaligned_ks=unaligned, check_snapshot=paged,
                     page_kv=128 if fp4 else 64, weights_padding=4 if heads == 12 else 0)
             for heads, fp4, paged, nq, kv, block, capacity, unaligned in cases
             if (run_paged if paged else run_contiguous)]
    par_compile([partial(call, compile_only=True) for call in calls], max_workers=128)
    for call in calls:
        call()
    if run_contiguous:
        test_sparse_mqa_metadata_tensor()
        test_sparse_mqa_unaligned_ks()
        test_sparse_mqa_capacity_boundaries()
    if run_paged:
        test_paged_sparse_mqa_metadata_contract()


def test_cuda_sparse_mqa_logits(skip_prof, run_contiguous, run_paged, num_heads, use_fp4):
    torch.manual_seed(0)
    torch.empty(0, device='npu')
    cases = [case for case in enumerate_cuda_sparse_mqa_logits()
             if (run_paged if case[1] else run_contiguous) and case[0] in use_fp4]
    num_cases = os.getenv('DG_MQA_NUM_CASES')
    if num_cases is not None:
        cases = random.Random(200000).sample(cases, min(int(num_cases), len(cases)))
    calls = [partial(run_sparse_mqa_logits_case, heads, *case[:-1], skip_prof, use_unaligned_ks=case[-1])
             for case in cases for heads in (8, 12, 20, 32) if heads in num_heads]
    print(f'Testing CUDA-parity sparse MQA logits ({len(calls)} cases):')
    par_compile([partial(call, compile_only=True) for call in calls], max_workers=128)
    for call in calls:
        call()
    print()


def test_paged_mqa_logits_boundaries():
    context_lens = (0, 1, 63, 64, 65, 127, 128, 129, 511, 512, 513)
    request_tokens = (1, 8, 2, 7, 3, 6, 4, 5, 1, 8, 2)
    # Pair formats, page sizes and layouts instead of a full cross product.
    cases = (
        (8, 128, False, 64, False),
        (12, 128, True, 128, False),
        (20, 128, False, 128, True),
        (64, 64, True, 64, True),
    )

    def run_all(collect=None):
        for heads, head_dim, fp4, page_size, ascend_layout in cases:
            run_paged_mqa_logits_case(
                len(context_lens), heads, head_dim, 512, fp4, request_tokens, page_size, ascend_layout,
                skip_prof=True, request_context_lens=context_lens, collect=collect)

    print('Testing MX Paged MQA Logits boundaries:')
    compile_mqa_kernels(run_all)
    run_all()
    print()



if __name__ == '__main__':
    from argparse import ArgumentParser

    def make_lookup(mapping):
        def lookup(s):
            try:
                return mapping[s.lower()]
            except KeyError:
                raise ValueError(s)  # argparse turns this into a friendly 'invalid choice'
        return lookup

    to_mode = make_lookup({mode.name.lower(): mode for mode in Mode})
    _formats = {'fp8': False, 'fp4': True}
    to_format = make_lookup(_formats)

    parser = ArgumentParser()
    parser.add_argument(
        'select', nargs='*',
        choices=('all', 'prefill', 'paged', 'sparse', 'paged_sparse', 'sparse_layout',
                 'sparse_cuda', 'paged_sparse_cuda', 'sparse_boundary', 'sparse_requests',
                 'boundary', 'paged_boundary', 'paged_sparse_boundary',
                 ),
        help='Tests to run (default: all; CUDA sparse parity is opt-in'
             ')')
    parser.add_argument('--mode', nargs='+', type=to_mode, choices=list(Mode), default=None,
                        help='ks/ke modes to test, e.g. --mode Causal Cp (default: built-in point set)')
    parser.add_argument('--formats', nargs='+', type=to_format,
                        choices=list(_formats.values()), default=list(_formats.values()),
                        metavar='{fp8,fp4}',
                        help='Quantized input formats to test (default: both)')
    parser.add_argument('--num_heads', nargs='+', type=int, default=MQA_PERF_NUM_HEADS,
                        help='Head counts to test, e.g. --num_heads 4 8 (default: 8 12 16 20 32 64)')
    parser.add_argument('--head_dims', nargs='+', type=int, default=(128,),
                        help='Head dims to test, e.g. --head_dims 64 128 (default: 128)')
    parser.add_argument('--page_sizes', '--page-sizes', nargs='+', type=int, choices=(64, 128), default=(64, 128),
                        help='Page sizes to test for paged MQA logits (default: 64 128)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--min-tpr', type=int, default=3,
                        help='Minimum random tokens per request for paged MQA logits (default: 3)')
    parser.add_argument('--max-tpr', type=int, default=7,
                        help='Maximum random tokens per request for paged MQA logits (default: 7)')
    parser.add_argument('--use-ascend-kv-layout', action='store_true',
                        help='Use the Ascend-optimized KV cache layout (default: disabled)')
    parser.add_argument('--skip-prof', '--no-prof', dest='skip_prof', action='store_true')

    args = parser.parse_args()
    if args.min_tpr < 1 or args.max_tpr < args.min_tpr:
        parser.error('--min-tpr and --max-tpr must define a positive nonempty range')
    torch.manual_seed(args.seed)

    selected = set(args.select or ('all',))
    if 'all' in selected:
        selected = {'prefill', 'paged', 'boundary', 'paged_boundary', 'sparse_boundary', 'paged_sparse_boundary'}
    if 'prefill' in selected:
        case_specs = args.mode if args.mode is not None else DEFAULT_MODES
        test_mqa_logits(case_specs, args.skip_prof, use_fp4=args.formats,
                        num_heads=args.num_heads, head_dims=args.head_dims)

    if 'paged' in selected:
        test_paged_mqa_logits(args.skip_prof, num_heads=args.num_heads,
                              head_dims=args.head_dims, use_fp4=args.formats,
                              min_tpr=args.min_tpr, max_tpr=args.max_tpr,
                              page_sizes=args.page_sizes,
                              use_ascend_kv_layout=args.use_ascend_kv_layout)

    if 'sparse' in selected:
        test_sparse_mqa_logits_perf(args.skip_prof)
    if 'paged_sparse' in selected:
        test_paged_sparse_mqa_logits_perf(args.skip_prof)

    if selected & {'sparse_cuda', 'paged_sparse_cuda'}:
        test_cuda_sparse_mqa_logits(
            args.skip_prof, 'sparse_cuda' in selected, 'paged_sparse_cuda' in selected,
            args.num_heads, args.formats)

    if 'boundary' in selected:
        test_mqa_logits_boundaries()
    if 'paged_boundary' in selected:
        test_paged_mqa_logits_boundaries()
    run_contiguous = bool(selected & {'sparse_boundary', 'sparse_layout', 'sparse_requests'})
    run_paged = bool(selected & {'paged_sparse_boundary', 'sparse_layout'})
    if run_contiguous or run_paged:
        test_sparse_mqa_logits_boundaries(run_contiguous, run_paged)
