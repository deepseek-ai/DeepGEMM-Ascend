"""Exercise repeated GEMMs across host-planning inputs on an Ascend device."""

import argparse

import torch
import torch_npu  # noqa: F401

import deep_gemm


def run(device: int) -> None:
    torch.npu.set_device(device)
    original_num_sms = deep_gemm.get_num_sms()
    alternate_num_sms = original_num_sms // 2 if original_num_sms > 1 else 2
    m, n, k = 128, 256, 128
    generator = torch.Generator().manual_seed(27)
    # Integer products and sums are exact in FP32, independent of K-reduction order.
    a_cpu = torch.randint(-2, 3, (m, k), generator=generator).to(torch.bfloat16)
    b_cpu = torch.randint(-2, 3, (n, k), generator=generator).to(torch.bfloat16)
    bias_cpu = torch.randint(-2, 3, (1, n), generator=generator).to(torch.bfloat16)
    product = a_cpu.float() @ b_cpu.float().T
    a, b, bias = (tensor.to('npu') for tensor in (a_cpu, b_cpu, bias_cpu))
    aligned_output = torch.empty((m, n), dtype=torch.bfloat16, device='npu')
    guarded_output = torch.empty((m, n + 17), dtype=torch.bfloat16, device='npu')
    offset_output = guarded_output[:, 1:n + 1]
    if aligned_output.data_ptr() % 32 or offset_output.data_ptr() % 32 == 0:
        raise RuntimeError('the scenario requires aligned and unaligned output addresses')

    cases = (
        ('identity', original_num_sms, False, aligned_output),
        ('bias', original_num_sms, True, aligned_output),
        ('identity after bias', original_num_sms, False, aligned_output),
        ('changed cores', alternate_num_sms, False, aligned_output),
        ('offset output', alternate_num_sms, False, offset_output),
        ('restored cores', original_num_sms, False, aligned_output),
    )
    try:
        for label, num_sms, with_bias, output in cases:
            deep_gemm.set_num_sms(num_sms)
            expected = (product + bias_cpu.float() if with_bias else product).to(torch.bfloat16)
            for repetition in range(5):
                guarded_output.fill_(float('nan'))
                output.fill_(float('nan'))
                deep_gemm.bf16_gemm_nt(a, b, output, c=bias if with_bias else None)
                torch.npu.synchronize()
                actual = output.cpu()
                if not torch.equal(actual, expected):
                    mismatches = torch.count_nonzero(actual != expected).item()
                    raise RuntimeError(f'{label}, repetition {repetition}: {mismatches} output mismatches')
                if output is offset_output:
                    guards = guarded_output.cpu()
                    if not torch.isnan(guards[:, :1]).all() or not torch.isnan(guards[:, n + 1:]).all():
                        raise RuntimeError(f'{label}, repetition {repetition}: output guard changed')
            print(f'{label}: five exact outputs, num_sms={num_sms}')
    finally:
        deep_gemm.set_num_sms(original_num_sms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    run(parser.parse_args().device)
