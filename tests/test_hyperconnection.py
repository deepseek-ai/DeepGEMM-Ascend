import gc
import random
from functools import partial

import torch

import deep_gemm
from deep_gemm.testing.bench import bench_msprof
from deep_gemm.testing.disable_fallback import disable_cpu_fallback
from deep_gemm.testing.par_compile import par_compile
from utils import calc_diff, count_bytes, print_perf


# (m, n, k): m tokens, n = hc_mult * (hc_mult + 2) mix outputs, k = hc_mult * hidden.
# k = 16384 / 20480 / 28672 are the production shapes (hidden 4096 / 5120 / 7168); 7680 and
# 7168 keep coverage for K block counts that are not a power of two.
CASE_SLICES = {
    'default': tuple(
        (m, n, k)
        for m in (13, 137, 512, 4096, 8192)
        for n, k in ((24, 28672), (24, 20480), (24, 16384), (24, 7680), (24, 7168))
    ),
    'peak': tuple(
        (m, 24, 28672)
        for m in (1, 13, 128, 4096, 8192, 16384, 32768)
    ),
}



def compile_hc_prenorm_gemm(test_slices: tuple[str, ...]) -> None:
    """Compile all selected shapes in parallel."""
    disable_cpu_fallback()
    cases = tuple(dict.fromkeys(
        case
        for test_slice in test_slices
        for case in CASE_SLICES[test_slice]
    ))

    kernels = []
    for a_dtype in (torch.bfloat16, torch.float32):
        for m, n, k in cases:
            a = torch.zeros((m, k), dtype=a_dtype, device='npu')
            b = torch.zeros((n, k), dtype=torch.float32, device='npu')
            for num_splits in (None, 16):
                d = torch.empty((m, n) if num_splits is None else (num_splits, m, n), dtype=torch.float32, device='npu')
                sqr_sum = torch.empty((m,) if num_splits is None else (num_splits, m), dtype=torch.float32, device='npu')
                kernels.append(partial(deep_gemm.tf32_hc_prenorm_gemm, a, b, d, sqr_sum, num_splits=num_splits))


    par_compile(kernels)

    del kernels, a, b, d, sqr_sum
    gc.collect()
    torch.npu.empty_cache()


def test_hc_prenorm_gemm(test_slice: str = 'default', skip_prof: bool = False) -> None:
    disable_cpu_fallback()

    print(f'Testing MHC prenorm gemm: {test_slice}')
    for a_dtype in (torch.bfloat16, torch.float32):
        for m, n, k in CASE_SLICES[test_slice]:
            for num_splits in (None, 16):
                a = torch.randn((m, k), dtype=a_dtype, device='npu')
                b = torch.randn((n, k), dtype=torch.float32, device='npu')
                d = torch.empty((m, n) if num_splits is None else (num_splits, m, n), dtype=torch.float32, device='npu')
                sqr_sum = torch.empty((m,) if num_splits is None else (num_splits, m), dtype=torch.float32, device='npu')
                deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=num_splits)

                final_d = d if num_splits is None else d.sum(0)
                final_sqr_sum = sqr_sum if num_splits is None else sqr_sum.sum(0)

                ref_d = a.float() @ b.T
                ref_sqr_sum = a.float().square().sum(-1)
                diff_d = calc_diff(final_d, ref_d)
                diff_sqr_sum = calc_diff(final_sqr_sum, ref_sqr_sum)
                assert diff_d < 1e-6, f'{a_dtype=}, {m=}, {n=}, {k=}, {num_splits=}, {diff_d=:.10f}'
                assert diff_sqr_sum < 1e-6, f'{a_dtype=}, {m=}, {n=}, {k=}, {num_splits=}, {diff_sqr_sum=:.10f}'

                prof = None
                if not skip_prof:
                    prof = bench_msprof(
                        lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=num_splits),
                        kernel_names='tf32_hc_prenorm_gemm')
                print_perf(
                    m, n, k, f'a={a_dtype}, tf32, prenorm (num_splits={num_splits or 1})',
                    2.0 * m * n * k,
                    count_bytes(a, b, d, sqr_sum),
                    prof,
                )
    print()


def test_hc_prenorm_gemm_deterministic(test_slice: str = 'default') -> None:
    disable_cpu_fallback()

    print(f'Testing MHC prenorm gemm deterministic & batch invariance: {test_slice}')
    for a_dtype in (torch.bfloat16, torch.float32):
        for m, n, k in CASE_SLICES[test_slice]:
            a = torch.randn((m, k), dtype=a_dtype, device='npu')
            b = torch.randn((n, k), dtype=torch.float32, device='npu')
            d_2d = torch.empty((m, n), dtype=torch.float32, device='npu')
            sqr_2d = torch.empty((m,), dtype=torch.float32, device='npu')
            deep_gemm.tf32_hc_prenorm_gemm(a, b, d_2d, sqr_2d)
            d_1s = torch.empty((1, m, n), dtype=torch.float32, device='npu')
            sqr_1s = torch.empty((1, m), dtype=torch.float32, device='npu')
            deep_gemm.tf32_hc_prenorm_gemm(a, b, d_1s, sqr_1s, num_splits=1)
            assert torch.equal(d_2d.view(torch.int32), d_1s[0].view(torch.int32)), (
                f'num_splits=None vs 1 D mismatch: {a_dtype=}, {m=}, {n=}, {k=}'
            )
            assert torch.equal(sqr_2d.view(torch.int32), sqr_1s[0].view(torch.int32)), (
                f'num_splits=None vs 1 sqr_sum mismatch: {a_dtype=}, {m=}, {n=}, {k=}'
            )

            num_splits = 16
            d = torch.empty((num_splits, m, n), dtype=torch.float32, device='npu')
            sqr_sum = torch.empty((num_splits, m), dtype=torch.float32, device='npu')
            deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=num_splits)

            ref_d = a.float() @ b.T
            ref_sqr_sum = a.float().square().sum(-1)
            diff_d = calc_diff(d.sum(0), ref_d)
            diff_sqr_sum = calc_diff(sqr_sum.sum(0), ref_sqr_sum)
            assert diff_d < 1e-6, f'deterministic: {a_dtype=}, {m=}, {n=}, {k=}, {diff_d=:.10f}'
            assert diff_sqr_sum < 1e-6, f'deterministic: {a_dtype=}, {m=}, {n=}, {k=}, {diff_sqr_sum=:.10f}'
            deterministic_d, deterministic_sqr_sum = d.clone(), sqr_sum.clone()
            for _ in range(10):
                deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=num_splits)
                assert torch.equal(d.view(torch.int32), deterministic_d.view(torch.int32)), f'D repeat: {a_dtype=}, {m=}, {n=}, {k=}'
                assert torch.equal(sqr_sum.view(torch.int32), deterministic_sqr_sum.view(torch.int32)), f'square sum repeat: {a_dtype=}, {m=}, {n=}, {k=}'

            # Check sampled rows individually against the full batch.
            single_d = torch.empty((num_splits, 1, n), dtype=torch.float32, device='npu')
            single_sqr_sum = torch.empty((num_splits, 1), dtype=torch.float32, device='npu')
            for row_idx in sorted({0, m // 2, m - 1}):
                deep_gemm.tf32_hc_prenorm_gemm(a[row_idx:row_idx + 1], b, single_d, single_sqr_sum, num_splits=num_splits)
                assert torch.equal(single_d.view(torch.int32), deterministic_d[:, row_idx:row_idx + 1].view(torch.int32)), f'D BI: {a_dtype=}, {m=}, {n=}, {k=}, {row_idx=}'
                assert torch.equal(single_sqr_sum.view(torch.int32), deterministic_sqr_sum[:, row_idx:row_idx + 1].view(torch.int32)), f'square sum BI: {a_dtype=}, {m=}, {n=}, {k=}, {row_idx=}'
            print(f' > OK ({a_dtype=}, {m=}, {n=}, {k=})', flush=True)
    print()




if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('select', nargs='?', default='all', choices=['all', *CASE_SLICES])
    parser.add_argument('--skip-prof', '--no-prof', dest='skip_prof', action='store_true',
                        help='Skip performance measurements and just test correctness')
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    selected_slices = tuple(CASE_SLICES) if args.select == 'all' else (args.select,)
    compile_hc_prenorm_gemm(selected_slices)

    for test_slice in selected_slices:
        test_hc_prenorm_gemm(test_slice, skip_prof=args.skip_prof)
        test_hc_prenorm_gemm_deterministic(test_slice)
