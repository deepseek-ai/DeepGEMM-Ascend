import torch
from generators import GemmType
from common import run_gemm_test, run_syrk_test, run_symm_test


def test_bf16_gemm(
        gemm_type: GemmType = GemmType.Normal,
        unaligned_mn: bool = False,
        unaligned_k: bool = False,
        strided: bool = False,
        skip_prof: bool = False
):
    return run_gemm_test(
        torch.bfloat16, torch.bfloat16,
        gemm_type, unaligned_mn, unaligned_k, strided,
        skip_prof=skip_prof,
        threshold=1e-5,
    )


def test_batched_syrk(
        batched: bool = True,
        strided: bool = False,
        skip_prof: bool = False
) -> None:
    return run_syrk_test(batched, strided, skip_prof=skip_prof, threshold=1e-5)


def test_batched_symm(
        batched: bool = True,
        strided: bool = False,
        skip_prof: bool = False
) -> None:
    return run_symm_test(batched, strided, skip_prof=skip_prof)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('select', nargs='?', default='all', choices=['all', 'unaligned-k', 'unaligned-mn', 'strided', 'normal', 'mgrouped', 'kgrouped', 'syrk', 'symm'])
    parser.add_argument('--skip-prof', '--no-prof', dest='skip_prof', action='store_true', help='Skip performance measurements and just test correctness')

    args = parser.parse_args()
    sel = args.select
    skip_prof = args.skip_prof
    if sel == 'unaligned-k':
        test_bf16_gemm(GemmType.Normal, unaligned_k=True, skip_prof=skip_prof)
    elif sel == 'unaligned-mn':
        test_bf16_gemm(GemmType.Normal, unaligned_mn=True, skip_prof=skip_prof)
    elif sel == 'strided':
        test_bf16_gemm(GemmType.Normal, strided=True, skip_prof=skip_prof)
    elif sel == 'normal':
        test_bf16_gemm(GemmType.Normal, skip_prof=skip_prof)
    elif sel == 'mgrouped':
        test_bf16_gemm(GemmType.MGroupedContiguousWithPsumLayout, skip_prof=skip_prof)
    elif sel == 'kgrouped':
        test_bf16_gemm(GemmType.KGroupedContiguousWithPsumLayout, skip_prof=skip_prof)
    elif sel == 'symm':
        test_batched_symm(batched=False, strided=True, skip_prof=skip_prof)
        test_batched_symm(strided=True, skip_prof=skip_prof)
        test_batched_symm(skip_prof=skip_prof)
    elif sel == 'syrk':
        test_batched_syrk(batched=False, strided=True, skip_prof=skip_prof)
        test_batched_syrk(strided=True, skip_prof=skip_prof)
        test_batched_syrk(skip_prof=skip_prof)
    else:
        test_bf16_gemm(GemmType.Normal, skip_prof=skip_prof)
        test_bf16_gemm(GemmType.Normal, unaligned_mn=True, skip_prof=skip_prof)
        test_bf16_gemm(GemmType.Normal, unaligned_k=True, skip_prof=skip_prof)
        test_bf16_gemm(GemmType.Normal, strided=True, skip_prof=skip_prof)

        test_bf16_gemm(GemmType.MGroupedContiguousWithPsumLayout, skip_prof=skip_prof)
        test_bf16_gemm(GemmType.KGroupedContiguousWithPsumLayout, skip_prof=skip_prof)

        test_batched_symm(batched=False, strided=True, skip_prof=skip_prof)
        test_batched_symm(strided=True, skip_prof=skip_prof)
        test_batched_symm(skip_prof=skip_prof)

        test_batched_syrk(batched=False, strided=True, skip_prof=skip_prof)
        test_batched_syrk(strided=True, skip_prof=skip_prof)
        test_batched_syrk(skip_prof=skip_prof)
