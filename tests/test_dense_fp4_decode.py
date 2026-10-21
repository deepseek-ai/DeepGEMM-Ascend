"""Native ND FP4 dense decode: exact output and strided-output guards."""
import argparse
import torch
import torch_npu  # noqa: F401
import deep_gemm


def test_dense_fp4_decode(m=16, n=16384, k=1024, device=0):
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(20261021)
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., 0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float64)

    def operand(rows):
        choices = torch.tensor([1, 2, 4, 9, 10, 12], dtype=torch.uint8)
        codes = choices[torch.randint(6, (rows, k), generator=generator)]
        exponents = torch.randint(126, 129, (rows, k // 32), generator=generator)
        decoded = values[codes.long()] * (2. ** (exponents - 127)).repeat_interleave(32, dim=1)
        packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).view(torch.int8).npu()
        scales = (exponents[:, ::2] | (exponents[:, 1::2] << 8)).to(torch.int16)
        sf = torch.empty_strided((rows, k // 64), (1, rows), dtype=torch.int16, device='npu')
        sf.copy_(scales)
        return packed, sf, decoded

    a, sa, av = operand(m)
    b, sb, bv = operand(n)
    sa = deep_gemm.transform_sf_into_required_layout(sa, m, k, (1, 1, 32), is_sfa=True)
    sb = deep_gemm.transform_sf_into_required_layout(sb, n, k, (1, 1, 32), is_sfa=False)
    expected = av @ bv.T
    passed = 0
    for dtype in (torch.bfloat16, torch.float32):
        for offset, extra in ((0, 0), (0, 1), (1, 2)):
            storage = torch.full((m + 2, n + extra), 77, dtype=dtype, device='npu')
            output = storage[1:-1, offset:offset + n]
            deep_gemm.fp8_fp4_gemm_nt((a, sa), (b, sb), output, recipe_a=(1, 32), recipe_b=(1, 32))
            torch.npu.synchronize()
            assert torch.equal(output.cpu(), expected.to(dtype)), (m, n, k, dtype, offset, extra)
            actual = storage.cpu()
            assert (actual[[0, -1]] == 77).all()
            if offset:
                assert (actual[1:-1, :offset] == 77).all()
            if offset + n < n + extra:
                assert (actual[1:-1, offset + n:] == 77).all()
            passed += 1
    print(f'PASS: {passed} exact cases for M={m}, N={n}, K={k}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key, default in [('m', 16), ('n', 16384), ('k', 1024), ('device', 0)]:
        parser.add_argument('--' + key, type=int, default=default)
    test_dense_fp4_decode(**vars(parser.parse_args()))
