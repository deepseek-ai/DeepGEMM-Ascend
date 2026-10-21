import argparse
import itertools

import numpy as np
import torch
import torch_npu  # noqa: F401

import deep_gemm


def test_small_m_grouped_gemm(expected_m=4, k=256):
    alignment = 256
    previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        assert deep_gemm.get_mk_alignment_for_contiguous_layout() == alignment
        counts = [0, 1, 17, 0, alignment + 3, 7]
        starts, ends, total = [], [], 0
        for count in counts:
            starts.append(total)
            ends.append(total + count)
            total = (total + count + alignment - 1) // alignment * alignment
        layout = torch.tensor(ends, dtype=torch.int32, device='npu')
        values = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
        for kind, n in itertools.product(('fp4', 'fp8', 'mixed'), (512, 528)):
            rng = np.random.default_rng(123)

            def operand(shape, packed, activation=False):
                codes = rng.integers(0, 16, shape, dtype=np.uint8)
                if activation:
                    for end in ends:
                        codes[end:(end + alignment - 1) // alignment * alignment] = 0
                exponents = rng.integers(125, 130, (*shape[:-1], shape[-1] // 32), dtype=np.uint16)
                decoded = values[codes] * np.exp2(exponents.astype(np.float32) - 127).repeat(32, axis=-1)
                data = (torch.from_numpy(codes[..., ::2] | codes[..., 1::2] << 4).view(torch.int8)
                        if packed else torch.from_numpy(values[codes]).to(torch.float8_e4m3fn))
                sf = (exponents[..., ::2] | exponents[..., 1::2] << 8).view(np.int16)
                sf = torch.from_numpy(sf.swapaxes(-1, -2).copy()).npu().transpose(-1, -2)
                return (data.npu(), sf), torch.from_numpy(decoded).double()

            a, a_ref = operand((total, k), kind == 'fp4', activation=True)
            b, b_ref = operand((len(counts), n, k), kind != 'fp8')
            expected = torch.zeros((total, n), dtype=torch.float64)
            for group, (start, end) in enumerate(zip(starts, ends)):
                expected[start:end] = a_ref[start:end] @ b_ref[group].T
            # Include padded rows, N tails, a strided D, and both output dtypes.
            for dtype in (torch.bfloat16, torch.float32):
                storage = torch.full((total, n + 16), float('nan'), dtype=dtype, device='npu')
                d = storage[:, :n]
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                    a, b, d, layout, expected_m_for_psum_layout=expected_m)
                torch.npu.synchronize()
                assert deep_gemm.get_mk_alignment_for_contiguous_layout() == 256
                torch.testing.assert_close(d.cpu(), expected.to(dtype), rtol=0, atol=0)
                assert torch.isnan(storage[:, n:]).all().item()
                if kind == 'mixed' and expected_m == 4:
                    # An unaligned D pointer/stride must retain the AIV output fallback.
                    unaligned = torch.full((total, n + 17), float('nan'), dtype=dtype, device='npu')
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                        a, b, unaligned[:, 1:n + 1], layout, expected_m_for_psum_layout=expected_m)
                    torch.testing.assert_close(unaligned[:, 1:n + 1].cpu(), expected.to(dtype), rtol=0, atol=0)
                    assert torch.isnan(unaligned[:, :1]).all().item()
                    assert torch.isnan(unaligned[:, n + 1:]).all().item()
            print(f'PASS expected_m={expected_m} {kind} N={n}', flush=True)
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(previous)


def test_invalid_alignment():
    previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
    for value in (0, -16, 15, 16, 48, 64, 128, 512):
        try:
            deep_gemm.set_mk_alignment_for_contiguous_layout(value)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f'accepted invalid alignment {value}')
        assert deep_gemm.get_mk_alignment_for_contiguous_layout() == previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-m', type=int, default=4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--k', type=int, default=256)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    assert deep_gemm.get_mk_alignment_for_contiguous_layout() == 256
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout() == 256
    test_invalid_alignment()
    test_small_m_grouped_gemm(args.expected_m, args.k)
