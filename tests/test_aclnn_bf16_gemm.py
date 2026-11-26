"""ACLNN output/accumulation contract; run on an Ascend NPU."""
import unittest

import torch
import torch_npu

import deep_gemm


class ACLNNBF16OutputTest(unittest.TestCase):
    def test_output_and_input_accumulation(self):
        # Integer-valued products are exactly representable in BF16, so every
        # layout and accumulation mode can be checked without a tolerance.
        for suffix in ('nt', 'nn', 'tn', 'tt'):
            for mode in ('none', 'separate', 'bias', 'alias'):
                for strided in (False, True):
                    with self.subTest(suffix=suffix, mode=mode, strided=strided):
                        a = torch.full((16, 64), 2, dtype=torch.bfloat16, device='npu')
                        b = torch.ones((32, 64), dtype=torch.bfloat16, device='npu')
                        d = torch.full((16, 48 if strided else 32), -7,
                                       dtype=torch.bfloat16, device='npu')[:, :32]
                        if mode == 'none':
                            c = None
                        elif mode == 'alias':
                            c = d.fill_(3)
                        else:
                            c = torch.full((1 if mode == 'bias' else 16, 32), 3,
                                           dtype=torch.bfloat16, device='npu')
                        c_before = c.clone() if c is not None else None
                        a_in = a.t() if suffix[0] == 't' else a
                        b_in = b if suffix[1] == 't' else b.t()

                        getattr(deep_gemm, f'aclnn_bf16_gemm_{suffix}')(a_in, b_in, d, c)

                        torch.testing.assert_close(d, torch.full_like(d, 128 if c is None else 131),
                                                   rtol=0, atol=0)
                        if mode in ('separate', 'bias'):
                            torch.testing.assert_close(c, c_before, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
