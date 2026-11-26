"""HyperConnection prenorm forward GEMM, TileLang implementation for Ascend."""

import math

import torch
import tilelang
from tilelang.ascend import language as T
from tilelang.layout import make_ascend_compact_nz_layout

from .. import _C


def get_config(num_tokens: int, num_k_splits: int, x_dtype: T.dtype = T.bfloat16) -> dict:
    """Return launch parameters for a given token count and K split count."""
    assert num_tokens > 0, f'HC prenorm fwd requires positive num_tokens, got {num_tokens}'
    assert x_dtype in (T.bfloat16, T.float32), f'HC prenorm fwd requires bfloat16 or float32 x, got {x_dtype}'
    block_num_tokens = 64 if (num_tokens <= 128 or x_dtype == T.float32) else 128
    num_aic_cores = _C.get_num_sms()
    num_token_blocks = math.ceil(num_tokens / block_num_tokens)

    assert 1 <= num_k_splits <= num_aic_cores, (
        f'HC prenorm fwd requires 1 <= num_k_splits <= {num_aic_cores}, got {num_k_splits}'
    )
    cores_per_split = min(num_aic_cores // num_k_splits, num_token_blocks)

    return {
        'block_num_tokens': block_num_tokens,
        'num_aic_cores': num_k_splits * cores_per_split,
        'x_dtype': x_dtype,
    }


@tilelang.jit
def tf32_hc_prenorm_gemm_kernel(
    x,
    fn,
    out_mul,
    sqrsum,
    block_num_tokens: int,
    num_aic_cores: int,
    x_dtype: T.dtype = T.bfloat16,
):
    block_hidden = 256
    mad_hidden = 64
    num_aivs = 2
    ub_size_bytes = 248 * 1024
    l1_size_bytes = 512 * 1024
    l0b_size_bytes = 64 * 1024
    l0c_size_bytes = 256 * 1024
    num_cast_stages = 2
    num_mad_stages = 2

    num_tokens = T.dynamic('num_tokens')
    num_k_splits = T.const('num_k_splits')
    mhc_mult3 = T.const('mhc_mult3')
    mhc_hidden_size = T.const('mhc_hidden_size')

    assert x_dtype in (T.bfloat16, T.float32), f'HC prenorm fwd requires bfloat16 or float32 x, got {x_dtype}'
    assert block_num_tokens in (64, 128), f'HC prenorm fwd requires block_num_tokens in (64, 128), got {block_num_tokens}'
    assert block_num_tokens == 64 or x_dtype == T.bfloat16, 'FP32 HC prenorm fwd requires block_num_tokens == 64'
    assert 0 < mhc_mult3 <= 64 and mhc_mult3 % 8 == 0, (
        f'HC prenorm fwd requires mhc_mult3 to be 8-aligned and no larger than 64, got {mhc_mult3}'
    )
    assert mhc_hidden_size > 0 and mhc_hidden_size % block_hidden == 0, (
        f'HC prenorm fwd requires mhc_hidden_size to be a positive multiple of {block_hidden}, got {mhc_hidden_size}'
    )
    num_hidden_blocks = mhc_hidden_size // block_hidden
    assert 1 <= num_k_splits <= num_hidden_blocks, (
        f'HC prenorm fwd requires 1 <= num_k_splits <= {num_hidden_blocks}, got {num_k_splits}'
    )
    assert num_aic_cores % num_k_splits == 0

    half_block_num_tokens = block_num_tokens // num_aivs
    num_mads_per_block = block_hidden // mad_hidden
    block_mhc_mult3 = (mhc_mult3 + 15) // 16 * 16
    cores_per_split = num_aic_cores // num_k_splits
    num_hidden_blocks_per_split = num_hidden_blocks // num_k_splits
    num_extra_hidden_blocks = num_hidden_blocks % num_k_splits
    num_l0b_stages = l0b_size_bytes // (block_mhc_mult3 * block_hidden * 4)
    num_l0c_stages = l0c_size_bytes // (block_num_tokens * block_mhc_mult3 * 4)
    num_token_blocks_per_chunk = cores_per_split * num_l0c_stages
    nz_stage_rows = half_block_num_tokens + 1

    # Derive UB and L1 stage counts from tile byte sizes and hardware capacities.
    x_ub_stage_bytes = half_block_num_tokens * block_hidden * x_dtype.bytes
    nz_ub_bytes = num_cast_stages * nz_stage_rows * block_hidden * 4
    sqrsum_ub_bytes = num_l0c_stages * max(64, half_block_num_tokens) * 4
    num_load_stages = 4 if x_ub_stage_bytes <= 16 * 1024 else 3
    assert num_load_stages * x_ub_stage_bytes + nz_ub_bytes + sqrsum_ub_bytes <= ub_size_bytes

    x_l1_stage_bytes = block_num_tokens * block_hidden * 4
    fn_l1_stage_bytes = block_mhc_mult3 * block_hidden * 4
    num_l1_x_stages = 4 if block_num_tokens == 64 else 3
    num_l1_fn_stages = (2 if block_num_tokens == 64 else 1) * num_l0b_stages
    assert num_l1_x_stages * x_l1_stage_bytes + num_l1_fn_stages * fn_l1_stage_bytes <= l1_size_bytes

    x: T.Tensor[(num_tokens, mhc_hidden_size), x_dtype]
    fn: T.Tensor[(mhc_mult3, mhc_hidden_size), T.float32]
    out_mul: T.Tensor[(num_k_splits, num_tokens, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_k_splits, num_tokens), T.float32]

    with T.MixedKernel(num_aic_cores) as (core_id, aiv_id):
        split_id = core_id // cores_per_split
        core_in_split = core_id % cores_per_split
        hidden_block_offset = split_id * num_hidden_blocks_per_split + T.min(split_id, num_extra_hidden_blocks)
        num_split_hidden_blocks = num_hidden_blocks_per_split + T.if_then_else(split_id < num_extra_hidden_blocks, 1, 0)
        num_token_blocks = T.ceildiv(num_tokens, block_num_tokens)
        num_token_chunks = T.ceildiv(num_token_blocks, num_token_blocks_per_chunk)

        x_ub = T.alloc_shared((half_block_num_tokens, block_hidden), x_dtype)
        x_nz_ub = T.alloc_shared((nz_stage_rows, block_hidden), T.float32)
        T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})
        # SIMD row sums use one full 64-lane register even when half_block_num_tokens < 64.
        sqrsum_ub = T.alloc_shared((num_l0c_stages, max(64, half_block_num_tokens)), T.float32)

        x_l1 = T.alloc_l1((block_num_tokens, block_hidden), T.float32)
        fn_l1 = T.alloc_l1((block_mhc_mult3, block_hidden), T.float32)
        x_l0 = T.alloc_l0a((block_num_tokens, mad_hidden), T.float32)
        fn_l0 = T.alloc_l0b((num_mads_per_block, block_mhc_mult3, mad_hidden), T.float32)
        out_mul_l0 = T.alloc_l0c((num_l0c_stages, block_num_tokens, block_mhc_mult3), T.float32)

        T.annotate_buffer_versions(
            {
                x_ub: num_load_stages,
                x_nz_ub: num_cast_stages,
                x_l1: num_l1_x_stages,
                fn_l1: num_l1_fn_stages,
                x_l0: num_mad_stages,
                fn_l0: num_l0b_stages,
            }
        )

        T.set_hf32_mode('nearest_even')
        for chunk_idx in T.Serial(num_token_chunks):
            token_block_base = chunk_idx * num_token_blocks_per_chunk + core_in_split
            num_valid_token_stages = T.min(
                num_l0c_stages,
                T.max(T.ceildiv(num_token_blocks - token_block_base, cores_per_split), 0),
            )

            for local_hidden_block in T.Pipelined(num_split_hidden_blocks, num_stages=2):
                global_hidden_block = hidden_block_offset + local_hidden_block
                hidden_begin = global_hidden_block * block_hidden
                T.assume(hidden_begin >= 0)
                T.assume(hidden_begin + block_hidden <= mhc_hidden_size)

                # Keep physical buffers block_mhc_mult3-aligned, but restrict
                # data movement and MAD regions to the logical mhc_mult3.
                T.copy(
                    fn[0:mhc_mult3, hidden_begin : hidden_begin + block_hidden],
                    fn_l1[0:mhc_mult3, :],
                )
                for hidden_mad in T.Serial(num_mads_per_block):
                    T.copy(
                        fn_l1[0:mhc_mult3, hidden_mad * mad_hidden : (hidden_mad + 1) * mad_hidden],
                        fn_l0[hidden_mad, 0:mhc_mult3, :],
                    )

                for token_stage in T.Serial(
                    num_valid_token_stages,
                    annotations={'multi_buffer_eligible': [x_l1]},
                ):
                    token_block = token_block_base + token_stage * cores_per_split
                    token_begin = token_block * block_num_tokens
                    actual_num_tokens = T.min(num_tokens - token_begin, block_num_tokens)
                    aiv_token_begin = token_begin + aiv_id * half_block_num_tokens

                    if aiv_token_begin < num_tokens:
                        actual_aiv_num_tokens = T.min(
                            num_tokens - aiv_token_begin,
                            half_block_num_tokens,
                        )
                        T.copy(
                            x[
                                aiv_token_begin : aiv_token_begin + actual_aiv_num_tokens,
                                hidden_begin : hidden_begin + block_hidden,
                            ],
                            x_ub[0:actual_aiv_num_tokens, :],
                            l2_cache_ctrl='notalloc_keep',
                            pad_value=0,
                        )

                        # Widen one AIV half-tile, pack padded NZ, and update row sums.
                        with T.SimdVF(latency=640):
                            bf16_mask = T.simd.pset(16)
                            f32_mask = T.simd.pset(32)
                            nz_stride = T.int32((nz_stage_rows << 16) | (8 * nz_stage_rows))
                            zero_bf16 = T.simd.vdup(T.bfloat16(0), 'bfloat16', bf16_mask)
                            row_indices = T.simd.vci(T.float32(0), 'float32')
                            current_row = T.simd.alloc_var('float32')
                            current_row = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                            one = T.simd.vdup(T.float32(1), 'float32', f32_mask)
                            sqr_sums = T.simd.alloc_var('float32')
                            sqr_sums = T.simd.vdup(T.float32(0), 'float32', f32_mask)

                            for row in T.Serial(half_block_num_tokens):
                                row_acc = T.simd.alloc_var('float32')
                                row_acc = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                                nz_ptr = T.simd.make_ubuf_ptr(
                                    T.access_ptr(
                                        x_nz_ub[row, 0],
                                        'w',
                                        1,
                                        block_hidden,
                                    ),
                                    'float32',
                                )

                                for pass_id in T.Serial(block_hidden // 128):
                                    if x_dtype == T.bfloat16:
                                        packed = T.simd.vld(x_ub[row, pass_id * 128])
                                        lo_bits, hi_bits = T.simd.vintlv(zero_bf16, packed)
                                        lo = T.reinterpret(lo_bits, 'float32x64')
                                        hi = T.reinterpret(hi_bits, 'float32x64')
                                    else:
                                        lo = T.simd.vld(x_ub[row, pass_id * 128])
                                        hi = T.simd.vld(x_ub[row, pass_id * 128 + 64])

                                    row_acc = T.simd.vadd(
                                        row_acc,
                                        T.simd.vmul(lo, lo, f32_mask),
                                        f32_mask,
                                    )
                                    row_acc = T.simd.vadd(
                                        row_acc,
                                        T.simd.vmul(hi, hi, f32_mask),
                                        f32_mask,
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        lo,
                                        nz_ptr,
                                        nz_stride,
                                        f32_mask,
                                        update=True,
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        hi,
                                        nz_ptr,
                                        nz_stride,
                                        f32_mask,
                                        update=True,
                                    )

                                row_sum = T.simd.alloc_var('float32')
                                row_sum = T.simd.vcadd(row_acc, f32_mask)
                                row_sum_broadcast = T.simd.vdupv(row_sum, f32_mask)
                                row_mask = T.simd.vcmp(
                                    row_indices,
                                    current_row,
                                    f32_mask,
                                    'eq',
                                )
                                sqr_sums = T.simd.vsel(row_sum_broadcast, sqr_sums, row_mask)
                                current_row = T.simd.vadd(current_row, one, f32_mask)

                            if local_hidden_block != 0:
                                previous = T.simd.vld(sqrsum_ub[token_stage, 0])
                                sqr_sums = T.simd.vadd(sqr_sums, previous, f32_mask)
                            T.simd.vsts(sqrsum_ub[token_stage, 0], sqr_sums, f32_mask)

                        T.dual_copy(
                            x_nz_ub[0:half_block_num_tokens, 0:block_hidden],
                            x_l1[0:block_num_tokens, 0:block_hidden],
                        )

                    for hidden_mad in T.Serial(num_mads_per_block):
                        T.copy(
                            x_l1[
                                0:actual_num_tokens,
                                hidden_mad * mad_hidden : (hidden_mad + 1) * mad_hidden,
                            ],
                            x_l0[0:actual_num_tokens, :],
                        )
                        T.gemm(
                            x_l0[0:actual_num_tokens, :],
                            fn_l0[hidden_mad, 0:mhc_mult3, :],
                            out_mul_l0[token_stage, 0:actual_num_tokens, 0:mhc_mult3],
                            transpose_B=True,
                            clear_accum=(local_hidden_block == 0 and hidden_mad == 0),
                            unit_flag_ctrl=T.Select(
                                local_hidden_block == num_split_hidden_blocks - 1 and hidden_mad == num_mads_per_block - 1,
                                3,
                                2,
                            ),
                        )

            for token_stage in T.Serial(num_valid_token_stages):
                token_block = token_block_base + token_stage * cores_per_split
                token_begin = token_block * block_num_tokens
                actual_num_tokens = T.min(num_tokens - token_begin, block_num_tokens)
                T.copy(
                    out_mul_l0[token_stage, 0:actual_num_tokens, 0:mhc_mult3],
                    out_mul[
                        split_id,
                        token_begin : token_begin + actual_num_tokens,
                        0:mhc_mult3,
                    ],
                    unit_flag_ctrl=3,
                )
                aiv_token_begin = token_begin + aiv_id * half_block_num_tokens
                if aiv_token_begin < num_tokens:
                    actual_aiv_num_tokens = T.min(num_tokens - aiv_token_begin, half_block_num_tokens)
                    T.copy(
                        sqrsum_ub[token_stage, 0:actual_aiv_num_tokens],
                        sqrsum[
                            split_id,
                            aiv_token_begin : aiv_token_begin + actual_aiv_num_tokens,
                        ],
                        l2_cache_ctrl='normal_fv',
                    )


def tf32_hc_prenorm_gemm(a: torch.Tensor,
                         b: torch.Tensor,
                         d: torch.Tensor,
                         sqr_sum: torch.Tensor,
                         num_splits: int | None = None) -> None:
    """Fused hyperconnection prenorm forward.

    Computes split-K partials of D = float(A) @ B.T and sqr_sum = sum(float(A) ** 2, dim=-1).
    The K split is not reduced here: callers fold the split dimension themselves in ascending
    order, which is what makes the result independent of the batch a token lands in.

    Args:
        a: [M, K] BF16 or FP32 input.
        b: [N, K] FP32 weight.
        d: [M, N] FP32 output when num_splits is None, or [num_splits, M, N] when num_splits is given.
        sqr_sum: [M] FP32 output when num_splits is None, or [num_splits, M] when num_splits is given.
        num_splits: K split count (defaults to 1 when None). One split owns every K block of its
            tokens, so it only fills `ceil(M / BLOCK_M)` AIC cores; pass the platform default
            (`tile_kernels.mhc.default_n_splits()`) for throughput.
    """
    m, k = a.shape
    n, k_ = b.shape
    num_aic_cores = _C.get_num_sms()
    assert k == k_, f'Shape mismatch: {a.shape=} vs {b.shape=}'
    assert 0 < n <= 64 and n % 8 == 0, f'HC prenorm fwd requires 0 < n <= 64 and n % 8 == 0, got {n}'
    assert k > 0 and k % 256 == 0, f'HC prenorm fwd requires k to be a positive multiple of 256, got {k}'
    if num_splits is None:
        assert d.shape == (m, n), f'Expected d shape {(m, n)}, got {tuple(d.shape)}'
        assert sqr_sum.shape == (m,), f'Expected sqr_sum shape {(m,),}, got {tuple(sqr_sum.shape)}'
        actual_splits = 1
        out_mul = d.unsqueeze(0)
        out_sqr_sum = sqr_sum.unsqueeze(0)
    else:
        # Each split needs at least one AIC core and one 256-wide K block; cores_per_split is
        # clamped by get_config so active_aic_cores is always a multiple of num_splits.
        max_splits = min(num_aic_cores, k // 256)
        assert 1 <= num_splits <= max_splits, (
            f'num_splits must be in 1..{max_splits}, got {num_splits}'
        )
        assert d.shape == (num_splits, m, n), f'Expected d shape {(num_splits, m, n)}, got {tuple(d.shape)}'
        assert sqr_sum.shape == (num_splits, m), f'Expected sqr_sum shape {(num_splits, m)}, got {tuple(sqr_sum.shape)}'
        actual_splits = num_splits
        out_mul = d
        out_sqr_sum = sqr_sum

    if m == 0:
        return
    assert a.dtype in (torch.bfloat16, torch.float32), f'Expected BF16 or FP32 a, got {a.dtype}'
    config = get_config(m, actual_splits, T.dtype(a.dtype))
    if _C.get_dry_run():
        # par_compile() sets the dry run flag: build the kernel but execute nothing.
        tf32_hc_prenorm_gemm_kernel.compile(a, b, out_mul, out_sqr_sum, **config)
        return
    tf32_hc_prenorm_gemm_kernel(a, b, out_mul, out_sqr_sum, **config)
