"""Byte-layout contract between weight loading and both MegaMoE MMA shapes."""

import unittest

import torch

from deep_gemm.mega import BLOCK_N, _pack_fp4_weights, transform_weights_for_mega_moe


def read_mma_tile(storage, packed_k, row, k, mad_m):
    """Model the GM-to-UB reads, independently of the Python pack transform."""
    block_k, mad_n = (512, 128) if mad_m == 128 else (256, 256)
    valid_bytes = min(block_k, 2 * packed_k - k) // 2
    tile_k = k // 512 * 512
    stride = min(2 * packed_k - tile_k, 512) // 2
    loaded = torch.empty((mad_n, valid_bytes), dtype=storage.dtype)
    for lane in range(mad_n):
        group, within = divmod(row + lane, 128)
        offset = group * 128 * packed_k + tile_k * 64 + within * stride + (k - tile_k) // 2
        loaded[lane] = storage[offset:offset + valid_bytes]
    return loaded


class TestMegaMoEWeights(unittest.TestCase):
    def test_mma_reads_preserve_every_weight_byte(self):
        generator = torch.Generator().manual_seed(348)
        for k in (256, 512, 768, 1280, 5120, 6144):
            for strided in (False, True):
                with self.subTest(k=k, strided=strided):
                    # Several experts and N blocks catch accidental cross-expert packing.
                    source = torch.randint(-128, 128, (3, 2, BLOCK_N, k // 2 * (2 if strided else 1)),
                                           dtype=torch.int8, generator=generator)
                    if strided:
                        source = source[..., ::2]
                    before = source.clone()
                    packed = _pack_fp4_weights(source)
                    self.assertEqual(packed.shape, source.shape)
                    self.assertTrue(packed.is_contiguous())
                    self.assertTrue(torch.equal(source, before))
                    for expert in range(3):
                        for n_block in range(2):
                            storage = packed[expert, n_block].flatten()
                            for mad_m, block_k, mad_n in ((128, 512, 128), (256, 256, 256)):
                                for row in range(0, BLOCK_N, mad_n):
                                    for col in range(0, k, block_k):
                                        actual = read_mma_tile(storage, k // 2, row, col, mad_m)
                                        expected = source[expert, n_block, row:row + mad_n,
                                                          col // 2:min(k, col + block_k) // 2]
                                        self.assertTrue(torch.equal(actual, expected),
                                                        (expert, n_block, mad_m, row, col))

    def test_public_transform_preserves_experts_gate_up_and_scales(self):
        experts, hidden, intermediate = 3, 768, 1280
        generator = torch.Generator().manual_seed(32)
        weights = []
        for n, k in ((2 * intermediate, hidden), (hidden, intermediate)):
            w = torch.randint(-128, 128, (experts, n, k // 2), dtype=torch.int8, generator=generator)
            sf = torch.randint(0, 249, (experts, n, k // 32), dtype=torch.uint8, generator=generator)
            sf[..., ::19] = 255  # Preserve the E8M0 special encoding.
            weights.append((w, sf.view(torch.int16)))
        transformed = transform_weights_for_mega_moe(*weights)
        for layer, ((weight, sf), (packed, packed_sf)) in enumerate(zip(weights, transformed)):
            n, packed_k = weight.shape[-2:]
            if layer == 0:
                half = n // 2
                order = torch.tensor([row + part * half for start in range(0, half, BLOCK_N // 2)
                                      for part in range(2) for row in range(start, start + BLOCK_N // 2)])
            else:
                order = torch.arange(n)
            for expert in range(experts):
                for block in range(n // BLOCK_N):
                    rows = order[block * BLOCK_N:(block + 1) * BLOCK_N]
                    storage = packed[expert, block].flatten()
                    restored = torch.cat([read_mma_tile(storage, packed_k, 0, col, 256)
                                          for col in range(0, packed_k * 2, 256)], dim=1)
                    self.assertTrue(torch.equal(restored, weight[expert, rows]))
                    original_sf = sf[expert, rows].view(torch.uint8)
                    expected_sf = torch.where(original_sf == 255, original_sf, original_sf + 6)
                    actual_sf = packed_sf[expert, block].transpose(-1, -2).contiguous().view(torch.uint8)
                    self.assertTrue(torch.equal(actual_sf, expected_sf))


if __name__ == '__main__':
    unittest.main()
