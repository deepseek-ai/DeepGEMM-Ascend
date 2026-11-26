"""Compare packed UE8M0 and FP32 scales through the ACLNN reference API."""
import unittest
from itertools import product

import torch
import torch_npu

import deep_gemm


def make_operand(rows, k, dtype, mn_major):
    if dtype == torch.int8:
        # E2M1 encodes 1.0 as 0x2; two values share each storage byte.
        shape = (rows // 2, k) if mn_major else (rows, k // 2)
        value = torch.full(shape, 0x22, dtype=dtype, device='npu')
    else:
        value = torch.ones((rows, k), device='npu').to(dtype)
    return value.t().contiguous().t() if mn_major else value


def make_scales(rows, k, mn_major):
    # Distinct adjacent exponents detect swapped bytes and incorrect pair axes.
    exponents = (torch.arange(rows * (k // 32)) % 5 + 125).to(torch.uint8).reshape(rows, k // 32)
    fp32 = (exponents.to(torch.int32) * (1 << 23)).view(torch.float32).to('npu')
    packed = exponents.view(torch.int16).to('npu')
    if mn_major:
        fp32 = fp32.t().contiguous().t()
        packed = torch.empty_strided(packed.shape, (1, rows), dtype=packed.dtype,
                                     device=packed.device).copy_(packed)
    return fp32, packed


class ACLNNPackedScalesTest(unittest.TestCase):
    def test_packed_scales_match_float_scales(self):
        fp8, fp4 = torch.float8_e4m3fn, torch.int8
        for (dtype_a, dtype_b), k in product(((fp8, fp8), (fp4, fp4), (fp8, fp4)), (64, 128)):
            # ACLNN's mixed FP8/FP4 path accepts K-major operands only.
            for operand_mn in ((False, True) if dtype_a == dtype_b else (False,)):
                for scale_mn in (False, True):
                    for suffix in ('nt', 'nn', 'tn', 'tt'):
                        with self.subTest(dtypes=(dtype_a, dtype_b), operand_mn=operand_mn,
                                          scale_mn=scale_mn, suffix=suffix, k=k):
                            a = make_operand(32, k, dtype_a, operand_mn)
                            b = make_operand(64, k, dtype_b, operand_mn)
                            sfa_float, sfa_packed = make_scales(32, k, scale_mn)
                            sfb_float, sfb_packed = make_scales(64, k, scale_mn)
                            if suffix[0] == 't':
                                a, sfa_float, sfa_packed = a.t(), sfa_float.t(), sfa_packed.t()
                            if suffix[1] == 'n':
                                b, sfb_float, sfb_packed = b.t(), sfb_float.t(), sfb_packed.t()
                            expected = torch.empty((32, 64), dtype=torch.bfloat16, device='npu')
                            actual = torch.empty_like(expected)
                            run = getattr(deep_gemm, f'aclnn_fp8_fp4_gemm_{suffix}')

                            run((a, sfa_float), (b, sfb_float), expected, recipe=(1, 1, 32))
                            run((a, sfa_packed), (b, sfb_packed), actual, recipe=(1, 1, 32))

                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
