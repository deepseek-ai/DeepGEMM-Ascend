import argparse
import itertools

import numpy as np
import torch
import torch_npu  # noqa: F401

import deep_gemm


def test_small_m_grouped_gemm(alignment=16):
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

            a, a_ref = operand((total, 256), kind == 'fp4', activation=True)
            b, b_ref = operand((len(counts), n, 256), kind != 'fp8')
            expected = torch.zeros((total, n), dtype=torch.float64)
            for group, (start, end) in enumerate(zip(starts, ends)):
                expected[start:end] = a_ref[start:end] @ b_ref[group].T
            # Include padded rows, N tails, a strided D, and the FP32 epilogue fallback.
            for dtype in (torch.bfloat16, torch.float32):
                storage = torch.full((total, n + 16), float('nan'), dtype=dtype, device='npu')
                d = storage[:, :n]
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                    a, b, d, layout, expected_m_for_psum_layout=max(counts))
                torch.npu.synchronize()
                torch.testing.assert_close(d.cpu(), expected.to(dtype), rtol=0, atol=0)
                assert torch.isnan(storage[:, n:]).all().item()
            print(f'PASS alignment={alignment} {kind} N={n}', flush=True)
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(previous)


def test_invalid_alignment():
    previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
    for value in (0, -16, 15, 48, 512):
        try:
            deep_gemm.set_mk_alignment_for_contiguous_layout(value)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f'accepted invalid alignment {value}')
        assert deep_gemm.get_mk_alignment_for_contiguous_layout() == previous


def test_k_grouped_mx_alignment():
    previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(16)
        data = torch.empty((64, 16), dtype=torch.float8_e4m3fn, device='meta')
        sf = torch.empty((2, 16), dtype=torch.float32, device='meta')
        output = torch.empty((1, 16, 16), dtype=torch.bfloat16, device='meta')
        layout = torch.empty(1, dtype=torch.int32, device='meta')
        try:
            deep_gemm.k_grouped_fp8_gemm_tn_contiguous((data, sf), (data, sf), output, [64], layout)
        except RuntimeError as error:
            assert 'K-grouped MX' in str(error), str(error)
        else:
            raise AssertionError('K-grouped MX accepted alignment below one SF pair')
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alignment', type=int, default=16)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    assert deep_gemm.get_mk_alignment_for_contiguous_layout() == 256
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout() == 16
    test_invalid_alignment()
    test_k_grouped_mx_alignment()
    test_small_m_grouped_gemm(args.alignment)
