import torch
import gc
import deep_gemm
import random
from dataclasses import replace
from deep_gemm.testing.bench import bench_msprof
from utils import assert_equal, assert_stochastic_bf16_matches_fp32, count_bytes, print_perf
from utils import assert_direct_output_matches_fp32_accumulation
from generators import major_opt, enumerate_gemm_tests, GemmType, Major, GemmTestDesc
from deep_gemm.testing.disable_fallback import disable_cpu_fallback
from deep_gemm.testing.par_compile import par_compile
from functools import partial
from itertools import product


def get_tag_name(gemm_type: GemmType, unaligned_mn: bool, unaligned_k: bool, strided: bool):
    tag = f'{gemm_type.name}'
    if unaligned_mn:
        tag += ' [unaligned-mn]'
    if unaligned_k:
        tag += ' [unaligned-k]'
    if strided:
        tag += ' [strided]'
    return tag


def get_tag_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return 'bf16'
    elif dtype == torch.float8_e4m3fn:
        return 'fp8'
    elif dtype == torch.float4_e2m1fn_x2:
        return 'fp4'
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')


def has_sf(a_dtype: torch.dtype, b_dtype: torch.dtype):
    return a_dtype != torch.bfloat16 or b_dtype != torch.bfloat16


def get_transposed(x: torch.Tensor, dim0: int, dim1: int):
    if isinstance(x, tuple):
        data, sf = x
        return data.transpose(dim0, dim1), sf.transpose(dim0, dim1)
    return x.transpose(dim0, dim1)


def gemm_dg(test: GemmTestDesc, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor,
            g_layout: torch.Tensor | None = None, trans_ab: str = 'nt', alpha: float | None = None, epilogue=None):
    gemm_name = 'fp8_fp4' if test.has_sf else 'bf16'
    extra_args = {'epilogue': epilogue}
    if test.with_bias:
        extra_args['c'] = c
    elif test.accumulate:
        assert test.gemm_type == GemmType.Normal or test.gemm_type == GemmType.KGroupedContiguousWithPsumLayout
        d.copy_(c)
        extra_args['c'] = d
    if test.gemm_type == GemmType.Normal:
        extra_args['alpha'] = alpha
        if test.has_sf:
            extra_args['recipe_a'] = test.a_recipe
            extra_args['recipe_b'] = test.b_recipe
        a = a if trans_ab[0] == 'n' else get_transposed(a, 0, 1)
        b = b if trans_ab[1] == 't' else get_transposed(b, 0, 1)
        getattr(deep_gemm, f'{gemm_name}_gemm_{trans_ab}')(a, b, d, **extra_args)
    elif test.gemm_type == GemmType.MGroupedContiguousWithPsumLayout:
        extra_args['ensure_zero_padding'] = test.ensure_zero_padding
        if test.has_sf:
            extra_args['recipe_a'] = test.a_recipe
            extra_args['recipe_b'] = test.b_recipe
        assert trans_ab[0] == 'n'
        b = b if trans_ab[1] == 't' else get_transposed(b, 1, 2)
        getattr(deep_gemm, f'm_grouped_{gemm_name}_gemm_{trans_ab}_contiguous')(a, b, d, g_layout, **extra_args)
    elif test.gemm_type == GemmType.KGroupedContiguousWithPsumLayout:
        # The k-grouped API takes a single combined `recipe` (gran_m, gran_n, gran_k)
        # rather than per-operand recipe_a/recipe_b (gran_k is shared, so it is equivalent).
        if test.has_sf:
            extra_args['recipe'] = test.recipe
        assert trans_ab == 'tn'
        k_grouped_gemm_name = 'fp8' if test.has_sf else gemm_name
        getattr(deep_gemm, f'k_grouped_{k_grouped_gemm_name}_gemm_{trans_ab}_contiguous')(
            a, b, d, test.host_ks_cpu, g_layout, **extra_args)
    else:
        return False


def gemm_aclnn(test: GemmTestDesc, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, g_layout: torch.Tensor | None = None, trans_ab: str='nt'):
    # ACLNN only provides a reference for the dense (Normal) GEMM; m-/k-grouped ops have
    # no aclnn counterpart, so skip the cross-check for them.
    if test.gemm_type != GemmType.Normal:
        return False
    if test.accumulate:
        d.copy_(c)
    if test.gemm_type == GemmType.Normal:
        a = a if trans_ab[0] == 'n' else get_transposed(a, 0, 1)
        b = b if trans_ab[1] == 't' else get_transposed(b, 0, 1)
        if test.has_sf:
            if test.k % 64 != 0:  # ACLNN requires K to be a multiple of 64
                return False
            # aclnnQuantMatmulV5's MX (ue8m0) path only supports per-row 1x32 scaling
            # (group_sizes [1, 1, 32]); block scaling along M/N (gran_mn != 1) is rejected
            # by the op ("Unsupported groupSize"). Skip the aclnn cross-check for those
            # recipes -- the DeepGEMM kernel is still validated against d_ref above.
            if test.a_recipe[0] != 1 or test.b_recipe[0] != 1:
                return False
            # ACLNN's MX QuantMatmul mishandles a transposed (MN-major) scale that
            # has only a single K-pair (ceil(k/32) <= 2, i.e. k <= 64): the permuted
            # [k/64, n, 2] scale has a leading singleton it misreads as n=1. The
            # deep_gemm result is still validated against d_ref above, so skip the
            # supplementary ACLNN comparison for this degenerate shape.
            if (test.k + 31) // 32 <= 2 and (test.a_major == Major.MN or test.b_major == Major.MN):
                return False
            n_fp4 = int(test.a_dtype == torch.float4_e2m1fn_x2) + int(test.b_dtype == torch.float4_e2m1fn_x2)
            if n_fp4 == 1:
                # see the document of aclnn for details about supported dtypes and layouts
                aclnn_supported = (
                    test.a_dtype == torch.float8_e4m3fn and test.b_dtype == torch.float4_e2m1fn_x2
                    and test.a_major == Major.K and test.b_major == Major.K
                    and test.cd_dtype in (torch.bfloat16, torch.float16)
                )
                if not aclnn_supported:
                    return False
            getattr(deep_gemm, f'aclnn_fp8_fp4_gemm_{trans_ab}')(a, b, d, d if test.accumulate else None, recipe_a=test.a_recipe, recipe_b=test.b_recipe)
        else:
            # ACLNN follows the same in-place accumulation convention for `c`.
            getattr(deep_gemm, f'aclnn_bf16_gemm_{trans_ab}')(a, b, d, d if test.accumulate else None)
    else:
        return False


def run_gemm_test(
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
        gemm_type: GemmType = GemmType.Normal,
        unaligned_mn: bool = False,
        unaligned_k: bool = False,
        strided: bool = False,
        skip_prof: bool = False,
        test_recipes: bool = False,
        threshold: float = 1e-5,
):
    disable_cpu_fallback()
    alpha_rng = random.Random(0)
    is_k_grouped = gemm_type == GemmType.KGroupedContiguousWithPsumLayout

    title = f'{get_tag_dtype(a_dtype)} x {get_tag_dtype(b_dtype)} gemm: {get_tag_name(gemm_type, unaligned_mn, unaligned_k, strided)}'
    print('Testing', title)

    tests = list(enumerate_gemm_tests(gemm_type, a_dtype, b_dtype, unaligned_mn, unaligned_k, strided, test_recipes))
    print('Num tests: ', len(tests))

    kernels = []
    for test in tests:
        for with_bias in (False, True) if gemm_type == GemmType.Normal else (False,):
            compile_test = replace(test, with_bias=with_bias)
            # Compile from metadata without allocating NPU storage.
            _, a, b, c, d_ref, (g_layout, _) = compile_test.generate(device='meta')
            for trans_ab in test.get_available_trans_ab():
                kernels.append(partial(gemm_dg, compile_test, a, b, c, d_ref, g_layout=g_layout, trans_ab=trans_ab))
                if gemm_type == GemmType.Normal and not with_bias:
                    alpha = alpha_rng.uniform(-1.0, 1.0)
                    alpha_test = replace(compile_test, alpha=alpha)
                    kernels.append(partial(gemm_dg, alpha_test, a, b, c, d_ref, g_layout=g_layout,
                                           trans_ab=trans_ab, alpha=alpha))
                if is_k_grouped and not test.accumulate:
                    fp32_test = replace(compile_test, accumulate=True, cd_dtype=torch.float32)
                    fp32 = torch.empty_like(d_ref, dtype=torch.float32)
                    kernels.append(partial(gemm_dg, fp32_test, a, b, fp32, fp32, g_layout=g_layout, trans_ab=trans_ab))
                if a_dtype == b_dtype and not with_bias and not test.accumulate and test.cd_dtype == torch.bfloat16:
                    kernels.append(partial(gemm_dg, compile_test, a, b, None, d_ref, g_layout=g_layout, trans_ab=trans_ab,
                                           epilogue=deep_gemm.epilogue.BF16StochasticRounding()))
                    if not is_k_grouped:
                        kernels.append(partial(gemm_dg, compile_test, a, b, None, torch.empty_like(d_ref, dtype=torch.float32),
                                               g_layout=g_layout, trans_ab=trans_ab))
    par_compile(kernels)
    del kernels, a, b, c, d_ref
    gc.collect()
    torch.npu.empty_cache()

    alpha_rng.seed(0)
    for test in tests:
        layout = major_opt(test.a_major, test.b_major)
        out_opt = 'bf16' if test.cd_dtype == torch.bfloat16 else 'fp32'

        for trans_ab in test.get_available_trans_ab():
            for use_alpha, with_bias in product((False, True), repeat=2) if gemm_type == GemmType.Normal else ((False, False),):
                alpha = alpha_rng.uniform(-1.0, 1.0) if use_alpha else None
                alpha_test = replace(test, alpha=alpha, with_bias=with_bias)
                _, a, b, c, d_ref, (g_layout, cmp_spans) = alpha_test.generate()
                d = torch.full_like(d_ref, float('nan')) if is_k_grouped and not test.accumulate else torch.randn_like(d_ref)
                if use_alpha and with_bias:
                    try:
                        gemm_dg(alpha_test, a, b, c, d, g_layout=g_layout, trans_ab=trans_ab, alpha=alpha)
                    except RuntimeError as e:
                        assert 'alpha and bias cannot be used together' in str(e), str(e)
                    else:
                        raise AssertionError('alpha + bias must be rejected')
                    continue
                gemm_dg(alpha_test, a, b, c, d, g_layout=g_layout, trans_ab=trans_ab, alpha=alpha)
                torch.npu.synchronize()
                assert_equal(d, d_ref, f'dg: {alpha_test} {trans_ab}', cmp_spans, threshold=threshold)
                if is_k_grouped and not test.accumulate:
                    fp32_test = replace(test, accumulate=True, cd_dtype=torch.float32)
                    assert_direct_output_matches_fp32_accumulation(
                        d, lambda output, accumulator: gemm_dg(fp32_test, a, b, accumulator, output, g_layout, trans_ab),
                        f'direct output: {test} {trans_ab}')

            (m, n, k), a, b, c, d_ref, (g_layout, cmp_spans) = test.generate()
            d = torch.randn_like(d_ref)
            has_ref = gemm_aclnn(test, a, b, c, d, g_layout=g_layout, trans_ab=trans_ab)
            torch.npu.synchronize()
            if has_ref is not False:
                assert_equal(d, d_ref, f'aclnn: {test} {trans_ab}', cmp_spans, threshold=threshold)

            if a_dtype == b_dtype and not test.accumulate and test.cd_dtype == torch.bfloat16:
                gemm_dg(test, a, b, None, d, g_layout=g_layout, trans_ab=trans_ab,
                        epilogue=deep_gemm.epilogue.BF16StochasticRounding())
                fp32_test = replace(test, accumulate=True, cd_dtype=torch.float32) if is_k_grouped else test
                assert_stochastic_bf16_matches_fp32(
                    d, lambda output: gemm_dg(fp32_test, a, b, output.zero_() if fp32_test.accumulate else None, output, g_layout, trans_ab),
                    f'stochastic rounding: {test} {trans_ab}')

        if is_k_grouped:
            k = sum(test.ks_cpu)
        prof = None
        if not skip_prof and test.profile and k:
            trans_ab = test.get_available_trans_ab()[0]
            prof = bench_msprof(lambda: gemm_dg(test, a, b, c, d, g_layout=g_layout, trans_ab=trans_ab), kernel_names='gemm_impl')

        print_perf(
            m, n, k,
            f'layout={layout}, {out_opt}, acc={int(test.accumulate)}, {test.tag}',
            2.0 * m * n * k,
            count_bytes(a, b, c, d),
            prof
        )
    print()
