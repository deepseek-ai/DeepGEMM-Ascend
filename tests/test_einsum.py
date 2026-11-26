import gc
from functools import partial
import torch
import torch_npu  # noqa: F401

import deep_gemm
from deep_gemm.testing.bench import bench_msprof
from deep_gemm.testing.par_compile import par_compile
from generators import enumerate_einsum, generate_einsum, quantize_with_major
from utils import assert_stochastic_bf16_matches_fp32, assert_stochastic_fp8_matches_fp32, calc_diff, count_bytes

_EXPR_ROLE = {
    'bhr,hdr->bhd': 'forward',
    'bhd,hdr->bhr': 'dgrad',
    'bhd,bhr->hdr': 'wgrad',
}


def test_einsum(dtype: torch.dtype = torch.float8_e4m3fn, skip_prof: bool = False) -> None:
    is_fp8 = dtype == torch.float8_e4m3fn
    name = 'MXFP8' if is_fp8 else 'BF16'
    print(f'Testing {name} einsum (forward / dgrad / wgrad, permute -> BMM):')
    torch.manual_seed(0)

    out_dtype = torch.bfloat16
    recipe = (1, 1, 32)   # Unified recipe: gran_k=32, one SF controls 32 K elements

    def run_case(expr, b, h, r, d, collect=None):
        role = _EXPR_ROLE[expr]
        gen = generate_einsum(expr, b, h, r, d, out_dtype, dtype)

        if is_fp8:
            (A, sfa), (B, sfb), c, D, ref = gen

            def run():
                # Kernels accumulate in-place into D; seed D with C before every launch.
                if c is not None:
                    D.copy_(c)
                return deep_gemm.fp8_einsum(expr, (A, sfa), (B, sfb), D, c=D if c is not None else None,
                                            recipe=recipe)
            bytes_moved = count_bytes(A, B, sfa, sfb, D) + (count_bytes(c) if c is not None else 0)
        else:
            A, B, c, D, ref = gen

            def run():
                # Kernels accumulate in-place into D; seed D with C before every launch.
                if c is not None:
                    D.copy_(c)
                return deep_gemm.einsum(expr, A, B, D, c=D if c is not None else None)
            bytes_moved = count_bytes(A, B, D) + (count_bytes(c) if c is not None else 0)

        launch_direct = partial(deep_gemm.fp8_einsum if is_fp8 else deep_gemm.einsum, expr, gen[0], gen[1])
        if collect is not None:
            collect.append(run)
            if role != 'forward':
                collect.append(partial(launch_direct, torch.empty_like(D, device='meta', dtype=torch.bfloat16),
                                       epilogue=deep_gemm.epilogue.BF16StochasticRounding()))
                collect.append(partial(launch_direct, torch.empty_like(D, device='meta', dtype=torch.float32)))
                if is_fp8 and role == 'wgrad' and r % 128 == 0:
                    meta_out = torch.empty_like(D, device='meta', dtype=torch.float8_e4m3fn)
                    meta_sfd = torch.empty_strided((d, h * r // 64), (1, d), device='meta', dtype=torch.int16)
                    collect.append(partial(launch_direct, meta_out, epilogue=deep_gemm.epilogue.FP8StochasticRounding(meta_sfd)))
            return

        run()
        torch.npu.synchronize()

        # The FP8 reference is a manual dequantized matmul (a different compute path than the
        # cube), and BF16 differs from an fp32 matmul by a few ULP, so both use calc_diff.
        diff = calc_diff(D.float(), ref.float())
        threshold = 1.5e-3 if is_fp8 else 1.5e-4
        assert diff < threshold, f'einsum {expr} ({role}) {b=} {h=} {r=} {d=}: diff={diff:.2e}'

        if is_fp8 and expr == 'bhr,hdr->bhd' and d % 128 == 0:
            output = torch.empty_like(D, dtype=torch.float8_e4m3fn)
            sfd = torch.empty_strided((b, h * d // 64), (1, b), device='npu', dtype=torch.int16)
            deep_gemm.fp8_einsum(expr, (A, sfa), (B, sfb), (output, sfd))

            (ref_fp8, ref_sf), _ = quantize_with_major(
                D.float().reshape(b, h * d), [1, 32], torch.float8_e4m3fn, major_dim=1)
            sf_bytes = ((ref_sf.cpu().view(torch.int32) // (2 ** 23)) % 256).to(torch.int32).reshape(b, -1, 2)
            ref_sfd = (sf_bytes[..., 0] + sf_bytes[..., 1] * 256).to(torch.int16)

            torch.npu.synchronize()
            assert torch.equal(
                output.cpu().view(torch.uint8).reshape(b, h * d),
                ref_fp8.cpu().view(torch.uint8)), f'{b=}, {h=}, {r=}, {d=}'
            assert torch.equal(sfd.contiguous().cpu(), ref_sfd), f'{b=}, {h=}, {r=}, {d=}'
            sf = (sfd.contiguous().view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
            restored_output = output.float().reshape(b, h * d) * sf.repeat_interleave(32, dim=1)
            diff = calc_diff(restored_output, ref.reshape(b, h * d))
            assert diff < 2e-3, f'{b=}, {h=}, {r=}, {d=}, {diff=:.5f}'

        if role != 'forward':
            output = torch.empty_like(D, dtype=torch.bfloat16)
            launch_direct(output, epilogue=deep_gemm.epilogue.BF16StochasticRounding())
            assert_stochastic_bf16_matches_fp32(output, launch_direct, f'{expr}, {b=}, {h=}, {r=}, {d=}')
            if is_fp8 and role == 'wgrad' and r % 128 == 0:
                fp8_output = torch.empty_like(D, dtype=torch.float8_e4m3fn)
                sfd = torch.empty_strided((d, h * r // 64), (1, d), device='npu', dtype=torch.int16)
                launch_direct(fp8_output, epilogue=deep_gemm.epilogue.FP8StochasticRounding(sfd))
                assert_stochastic_fp8_matches_fp32(
                    fp8_output, sfd, launch_direct, f'FP8 stochastic rounding ({role}), {b=}, {h=}, {r=}, {d=}')

        flops = 2.0 * b * h * r * d
        prof = None
        if not skip_prof:
            prof = bench_msprof(run, kernel_names='gemm_impl')
        head = f' > Perf ({role:7} {expr:14} b={b:5}, h={h:3}, r={r:5}, d={d:5}): '
        if prof is not None:
            print(head +
                f'{prof.dur_us:7.1f} us | {prof.tflops(flops):4.0f} TFLOPS | '
                f'{prof.gbps(bytes_moved):5.0f} GB/s | '
                f'mad={prof.aic_mad * 100:4.1f}% | mte2={prof.aic_mte2 * 100:4.1f}%')
        else:
            print(head + 'OK')

    def run_all(collect=None):
        for args in enumerate_einsum():
            run_case(*args, collect=collect)
        if is_fp8:
            for b, h, r, d in ((4, 8, 128, 128), (13, 8, 512, 256), (4096, 2, 128, 128),
                               (1, 3, 128, 128), (13, 3, 128, 384), (257, 2, 128, 128)):
                run_case('bhr,hdr->bhd', b, h, r, d, collect=collect)
            for b, h, r, d in ((128, 16, 512, 128), (128, 16, 512, 64), (128, 3, 128, 65)):
                run_case('bhd,bhr->hdr', b, h, r, d, collect=collect)

    kernels = []
    run_all(collect=kernels)
    par_compile(kernels)
    del kernels
    gc.collect()
    torch.npu.empty_cache()

    run_all()
    print()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('select', nargs='?', default='all', choices=['all', 'fp8', 'bf16'])
    parser.add_argument('--skip-prof', action='store_true', help='Skip performance measurements and just test correctness')

    args = parser.parse_args()
    sel = args.select
    skip_prof = args.skip_prof
    if sel == 'fp8':
        test_einsum(torch.float8_e4m3fn, skip_prof=skip_prof)
    elif sel == 'bf16':
        test_einsum(torch.bfloat16, skip_prof=skip_prof)
    else:
        test_einsum(torch.float8_e4m3fn, skip_prof=skip_prof)
        test_einsum(torch.bfloat16, skip_prof=skip_prof)
