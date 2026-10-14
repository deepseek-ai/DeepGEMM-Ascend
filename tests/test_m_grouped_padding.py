import torch
import torch_npu  # noqa: F401

from common import gemm_dg
from generators import GemmTestDesc, GemmType, Major, generate_operand_with_major, zero_operand_slices_
from deep_gemm.testing.par_compile import dry_run
from utils import assert_equal


def test_m_grouped_padding():
    torch.manual_seed(0)
    spans = [slice(0, 1), slice(256, 556), slice(768, 768)]
    padding = [slice(1, 256), slice(556, 768)]
    layout = torch.tensor([1, 556, 768], dtype=torch.int32, device='npu')
    dtypes = [torch.bfloat16, torch.float8_e4m3fn, torch.float4_e2m1fn_x2]
    pairs = [(dtype, dtype) for dtype in dtypes] + [(dtypes[1], dtypes[2])]

    for a_dtype, b_dtype in pairs:
        a, a_ref = generate_operand_with_major((768, 128), (1, 32), a_dtype, 1, 'npu')
        zero_operand_slices_(a, a_ref, padding)
        b, b_ref = generate_operand_with_major((3, 64, 128), (1, 1, 32), b_dtype, 2, 'npu')
        for cd_dtype in (torch.bfloat16, torch.float32):
            ref = torch.zeros((768, 64), dtype=cd_dtype, device='npu')
            for group, span in enumerate(spans):
                ref[span] = a_ref[span].float() @ b_ref[group].float().T

            for ensure_zero_padding in (True, False):
                test = GemmTestDesc(
                    GemmType.MGroupedContiguousWithPsumLayout,
                    a_dtype, b_dtype, cd_dtype, (300, 64, 128), Major.K, Major.K,
                    num_groups=3, ensure_zero_padding=ensure_zero_padding,
                )
                d = torch.full_like(ref, 7)
                with dry_run():
                    gemm_dg(test, a, b, None, d, layout)
                assert torch.all(d == 7).item(), 'dry run changed D'

                gemm_dg(test, a, b, None, d, layout)
                torch.npu.synchronize()
                assert_equal(d, ref, str(test), spans, threshold=1e-5)
                for span in padding:
                    expected = 0 if ensure_zero_padding else 7
                    assert torch.all(d[span] == expected).item(), f'padding rows {span}'


if __name__ == '__main__':
    with torch.npu.stream(torch.npu.Stream()):
        test_m_grouped_padding()
    print('m-grouped padding: OK')
