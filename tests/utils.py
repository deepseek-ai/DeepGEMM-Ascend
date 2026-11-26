"""Common utilities for GEMM tests: correctness metrics and perf reporting."""

from functools import cache
import math
import os
from pathlib import Path
import numpy as np
import torch


@cache
def _huge1g_extension():
    import torch_npu
    from torch.utils.cpp_extension import load

    npu_root = Path(torch_npu.__file__).parent
    ascend_root = Path(os.getenv('ASCEND_HOME_PATH', os.getenv(
        'ASCEND_TOOLKIT_HOME', '/usr/local/Ascend/ascend-toolkit/latest')))
    return load(
        name='deep_gemm_test_huge1g', sources=[str(Path(__file__).with_name('huge1g.cpp'))],
        extra_include_paths=[str(ascend_root / 'include'), str(npu_root / 'include')],
        extra_ldflags=[f'-L{ascend_root / "lib64"}', '-lascendcl', f'-L{npu_root / "lib"}', '-ltorch_npu'],
        with_cuda=False)


def empty_kv_cache(shape, *, dtype=torch.uint8):
    """Use the standard NPU allocator; DG_TEST_HUGE1G=1 opts into 1 GiB pages."""
    if os.getenv('DG_TEST_HUGE1G', '0') != '1':
        return torch.empty(shape, dtype=dtype, device='npu')
    num_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    return _huge1g_extension().empty_huge1g(num_bytes, torch.npu.current_device()).view(dtype).view(shape)




def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Cosine-similarity based diff, consistent with DeepGEMM."""
    x, y = x.float(), y.float()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def assert_equal(out: torch.Tensor, ref: torch.Tensor, msg: str,
                 slices: list[slice] = None, threshold: float = 0.0) -> None:
    """Check the kernel output against the aclnn reference, per group range if given (else
    the whole tensor). threshold == 0 demands bit-for-bit equality (FP8 MX: identical cube +
    cast path). threshold > 0 uses calc_diff (BF16: the cube kernel and aclnn differ by up to
    ~1 ULP from different K-accumulation, so an exact match is not expected)."""
    spans = slices if slices is not None else [slice(0, out.size(0))]
    for s in spans:
        if threshold == 0.0:
            assert torch.equal(out[s], ref[s]), f'{msg} (rows {s})'
        else:
            diff = calc_diff(out[s].float(), ref[s].float())
            assert diff < threshold, f'{msg} (rows {s}, diff={diff:.2e})'


def assert_direct_output_matches_fp32_accumulation(output: torch.Tensor, launch, label: str) -> None:
    assert output.dtype in (torch.float32, torch.bfloat16)
    fp32 = torch.zeros_like(output, dtype=torch.float32)
    launch(fp32, fp32)
    expected = fp32.to(output.dtype)
    bits_dtype = torch.int16 if output.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(output.view(bits_dtype), expected.view(bits_dtype)), label


def _quartet_hash(values: np.ndarray, with_second_pair: bool = False):
    h = (values[..., 0] * np.uint32(0x5671d42b) + values[..., 1] * np.uint32(0x9995e499) +
         values[..., 2] * np.uint32(0xace1b8a5) + values[..., 3] * np.uint32(0xe153538d) +
         np.arange(values.shape[-2], dtype=np.uint32))
    h = (h ^ (h >> 23)) * np.uint32(0x7feb352d)
    h ^= h >> 16
    h1 = h * np.uint32(0x846ca68b)
    h1 ^= h1 >> 11
    if not with_second_pair:
        return h1
    h2 = h * np.uint32(0xd35a2d97)
    h2 ^= h2 >> 11
    return h1, h2


def assert_stochastic_bf16_matches_fp32(output: torch.Tensor, launch, label: str) -> None:
    assert output.dtype == torch.bfloat16
    fp32 = torch.empty_strided(output.shape, output.stride(), device=output.device, dtype=torch.float32)
    launch(fp32)
    # CUDA's quartet hash, with deterministic zeros for out-of-bounds columns.
    values = fp32.cpu().contiguous().numpy().view(np.uint32)
    n = values.shape[-1]
    values = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, -n % 4)])
    values = values.reshape(*values.shape[:-1], -1, 4)
    h1, h2 = _quartet_hash(values, with_second_pair=True)
    random_bits = np.stack((h1 & 0xffff, h1 >> 16, h2 & 0xffff, h2 >> 16), axis=-1)
    # Compare the discarded tail with the complementary random threshold.
    rounded = (values >> 16) + ((values & 0xffff) > (0xffff - random_bits))
    rounded = np.where((values & 0x7fffffff) > 0x7f800000, 0x7fc0, rounded).astype(np.uint16)
    expected = torch.from_numpy(rounded.reshape(*fp32.shape[:-1], -1)[..., :n].copy()).view(torch.bfloat16)
    assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16)), label


def assert_stochastic_fp8_matches_fp32(output: torch.Tensor, sfd: torch.Tensor, launch, label: str) -> None:
    assert output.dtype == torch.float8_e4m3fn
    shape, n = output.shape, output.size(-1)
    assert n % 128 == 0
    num_batches = int(sfd.size(1)) * 64 // n
    num_rows = output.numel() // n
    num_group_rows = num_rows // num_batches

    fp32 = torch.empty_strided(output.shape, output.stride(), device=output.device, dtype=torch.float32)
    launch(fp32)

    # Per-row groups of 32 columns: |amax| -> UE8M0 exponent -> scaled FP32
    groups = fp32.contiguous().reshape(num_rows, n // 32, 32)
    amax_bits = groups.abs().amax(dim=-1).cpu().numpy().view(np.uint32)
    sf_exp = np.maximum((amax_bits + np.uint32(0x1fffff)) >> 23, np.uint32(113)) - np.uint32(8)
    sf_inv = torch.from_numpy(((np.uint32(254) - sf_exp) << 23).view(np.float32)).to(groups.device)
    bits = (groups * sf_inv.unsqueeze(-1)).cpu().numpy().view(np.uint32).reshape(num_rows, n // 4, 4)

    # Quartet hash and 16-bit thresholds (L, rev16(L), H, rev16(H))
    h = _quartet_hash(bits)
    pairs = np.stack((h & 0xffff, h >> 16), axis=-1).astype(np.uint16)
    rev = ((pairs & 0x5555) << 1) | ((pairs >> 1) & 0x5555)
    rev = ((rev & 0x3333) << 2) | ((rev >> 2) & 0x3333)
    rev = ((rev & 0x0f0f) << 4) | ((rev >> 4) & 0x0f0f)
    rev = ((rev & 0x00ff) << 8) | ((rev >> 8) & 0x00ff)
    random_bits = np.stack((pairs[..., 0], rev[..., 0], pairs[..., 1], rev[..., 1]), axis=-1).astype(np.uint32)

    # Align normal/subnormal mantissas, add the 16-bit threshold and saturate at 0x7e
    exp32 = (bits >> 23) & 0xff
    shift = np.minimum(121 - np.minimum(exp32, 121), 25)
    mant = np.where(exp32 > 120, ((exp32 - 120) << 23) | (bits & 0x7fffff),
                    ((bits & 0x7fffff) | 0x800000) >> shift)
    codes = np.minimum((mant >> 20) + ((((mant >> 4) & 0xffff) + random_bits) >> 16), 0x7e)
    expected = torch.from_numpy((((bits >> 31) << 7) | codes).astype(np.uint8)).reshape(shape)
    assert torch.equal(output.cpu().view(torch.uint8), expected), label

    expected_sf = sf_exp.reshape(num_batches, num_group_rows, n // 32).transpose(1, 0, 2)
    expected_sf = torch.from_numpy(expected_sf.reshape(num_group_rows, -1).astype(np.uint8))
    actual_sf = sfd.contiguous().cpu().view(torch.uint8)
    assert torch.equal(actual_sf, expected_sf), f'{label} (SFs)'


def count_bytes(*tensors) -> int:
    """Total bytes across all tensors."""
    flat_tensors: list[torch.Tensor] = []
    for t in tensors:
        if isinstance(t, tuple):
            flat_tensors.extend(t)
        else:
            flat_tensors.append(t)
    return sum(t.numel() * t.element_size() for t in flat_tensors if t is not None)


def print_perf(m, n, k, tags, flops, bytes_moved, prof):
    """Print a compact DeepGEMM-style perf line. `tags` (e.g. 'layout=NT, fp32, acc=0')
    is folded into the dimension parens."""
    line = (f' > Perf (m={m:6}, n={n:6}, k={k:6}, {tags}): ')
    if prof is not None:
        line += (f'{prof.dur_us:9.3f} us | {prof.tflops(flops):4.0f} TFLOPS | {prof.gbps(bytes_moved):5.0f} GB/s | '
            f'mad={prof.aic_mad*100:4.1f}% | mte2={prof.aic_mte2*100:4.1f}%')
    else:
        line += 'OK'
    print(line, flush=True)
