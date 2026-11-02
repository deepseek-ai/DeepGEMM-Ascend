"""Empty contractions must finish before native kernel dispatch."""
import unittest

import torch
import torch_npu

import deep_gemm


class EmptyGEMMTest(unittest.TestCase):
    def test_dense_zero_k(self):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for suffix in ('nt', 'nn', 'tn', 'tt'):
                for mode in ('none', 'alias', 'bias'):
                    with self.subTest(dtype=dtype, suffix=suffix, mode=mode):
                        a = torch.empty((8, 0), dtype=dtype, device='npu')
                        b = torch.empty((16, 0), dtype=dtype, device='npu')
                        d = torch.full((8, 16), -7, dtype=torch.bfloat16, device='npu')
                        c = None if mode == 'none' else (
                            d.fill_(3) if mode == 'alias' else torch.full_like(d[:1], 3))
                        c_before = c.clone() if c is not None else None
                        if dtype == torch.float8_e4m3fn:
                            a = (a, torch.empty((8, 0), device='npu'))
                            b = (b, torch.empty((16, 0), device='npu'))
                            if suffix[0] == 't':
                                a = tuple(t.t() for t in a)
                            if suffix[1] == 'n':
                                b = tuple(t.t() for t in b)
                            run = getattr(deep_gemm, f'fp8_fp4_gemm_{suffix}')
                        else:
                            a = a.t() if suffix[0] == 't' else a
                            b = b.t() if suffix[1] == 'n' else b
                            run = getattr(deep_gemm, f'bf16_gemm_{suffix}')

                        run(a, b, d, c=c)

                        torch.testing.assert_close(d, torch.full_like(d, 0 if c is None else 3),
                                                   rtol=0, atol=0)
                        if mode == 'bias':
                            torch.testing.assert_close(c, c_before, rtol=0, atol=0)

    def test_bf16_bmm_zero_k(self):
        for mode in ('none', 'alias', 'separate'):
            with self.subTest(mode=mode):
                a = torch.empty((2, 8, 0), dtype=torch.bfloat16, device='npu')
                b = torch.empty((2, 16, 0), dtype=torch.bfloat16, device='npu')
                d = torch.full((2, 8, 16), -7, dtype=torch.bfloat16, device='npu')
                c = None if mode == 'none' else (d.fill_(3) if mode == 'alias' else torch.full_like(d, 3))
                c_before = c.clone() if c is not None else None

                deep_gemm.bf16_bmm(a, b, d, c=c)

                torch.testing.assert_close(d, torch.full_like(d, 0 if c is None else 3), rtol=0, atol=0)
                if mode == 'separate':
                    torch.testing.assert_close(c, c_before, rtol=0, atol=0)

    def test_bf16_bmm_zero_batch(self):
        a = torch.empty((0, 8, 64), dtype=torch.bfloat16, device='npu')
        b = torch.empty((0, 16, 64), dtype=torch.bfloat16, device='npu')
        d = torch.empty((0, 8, 16), dtype=torch.bfloat16, device='npu')
        deep_gemm.bf16_bmm(a, b, d)
        self.assertEqual(d.numel(), 0)

    def test_bf16_bmm_zero_batch_validates_c(self):
        a = torch.empty((0, 8, 64), dtype=torch.bfloat16, device='npu')
        b = torch.empty((0, 16, 64), dtype=torch.bfloat16, device='npu')
        d = torch.empty((0, 8, 16), dtype=torch.bfloat16, device='npu')
        for c in (torch.empty_like(d, dtype=torch.float32),
                  torch.empty((0, 8, 17), dtype=d.dtype, device=d.device)):
            with self.subTest(shape=c.shape, dtype=c.dtype), self.assertRaises(RuntimeError):
                deep_gemm.bf16_bmm(a, b, d, c=c)

    def test_bf16_einsum_zero_reduction(self):
        a = torch.empty((8, 2, 0), dtype=torch.bfloat16, device='npu')
        b = torch.empty((2, 16, 0), dtype=torch.bfloat16, device='npu')
        d = torch.full((8, 2, 16), -7, dtype=torch.bfloat16, device='npu')
        deep_gemm.einsum('bhr,hdr->bhd', a, b, d)
        torch.testing.assert_close(d, torch.zeros_like(d), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
