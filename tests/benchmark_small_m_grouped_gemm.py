"""Small-M grouped GEMM fixture; use the README runner for read-sum cold L2."""
import argparse
import dataclasses
import hashlib
import json
import os
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', required=True, type=Path)
p.add_argument('--device', type=int, default=0)
p.add_argument('--groups', type=int, default=32)
p.add_argument('--m', type=int, default=4)
p.add_argument('--n', type=int, default=3072)
p.add_argument('--k', type=int, default=5120)
p.add_argument('--dtype', choices=['fp4', 'fp8', 'mixed'], default='fp4')
p.add_argument('--rounds', type=int, default=3)
a = p.parse_args()
os.environ['DG_PRINT_CONFIGS'] = '1'
import numpy as np
import torch
import torch_npu
torch.set_num_threads(4)
torch.npu.set_device(a.device)
import deep_gemm as dg
from deep_gemm.testing import bench_msprof
alignment = dg.get_mk_alignment_for_contiguous_layout()
rng = np.random.default_rng(42)
mpad = (a.m + alignment - 1) // alignment * alignment

def operand(shape, fp4):
    # Every product is a multiple of 1/16. FP32 accumulation is exact here.
    codes = rng.integers(0, 16, shape, dtype=np.uint8)
    lut = np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6], np.float32)
    values = lut[codes]
    exps = rng.integers(126, 129, (*shape[:-1], shape[-1] // 32), dtype=np.uint16)
    scale = np.exp2(exps.astype(np.float32)-127)
    ref = torch.from_numpy(values * scale.repeat(32, axis=-1)).npu()
    raw = torch.from_numpy(codes[..., ::2] | (codes[..., 1::2] << 4)).view(torch.int8).npu() if fp4 else torch.from_numpy(values).to(torch.float8_e4m3fn).npu()
    packed = (exps[..., ::2] | (exps[..., 1::2] << 8)).view(np.int16)
    sf = torch.from_numpy(packed.swapaxes(-1,-2).copy()).npu().transpose(-1,-2)
    return (raw, sf), ref

A, ar = operand((a.groups, a.m, a.k), a.dtype == 'fp4')
B, br = operand((a.groups, a.n, a.k), a.dtype in ('fp4','mixed'))
ar_pad = torch.zeros((a.groups, mpad, a.k), dtype=torch.float32, device='npu')
ar_pad[:, :a.m] = ar
ar = ar_pad.reshape(a.groups*mpad, a.k)
raw, sf = A
raw_pad = torch.zeros((a.groups, mpad, raw.shape[-1]), dtype=torch.uint8)
raw_pad[:, :a.m] = raw.view(torch.uint8).cpu()
sf_pad = torch.full((a.groups, mpad, a.k//64), 32639, dtype=torch.int16)
sf_pad[:, :a.m] = sf.cpu()
A = (raw_pad.reshape(a.groups*mpad, -1).view(raw.dtype).npu(), sf_pad.reshape(a.groups*mpad, -1).T.contiguous().npu().T)
D = torch.full((a.groups * mpad, a.n), float('nan'), dtype=torch.bfloat16, device='npu')
layout = torch.tensor([g * mpad + a.m for g in range(a.groups)], dtype=torch.int32, device='npu')
ref = torch.bmm(ar.reshape(a.groups, mpad, a.k), br.transpose(-1,-2)).to(torch.bfloat16)
ref = ref[:, :a.m].cpu()
A = (A[0], dg.transform_sf_into_required_layout(A[1], a.groups*mpad, a.k, (1,1,32), is_sfa=True, psum_layout=layout))
B = (B[0], dg.transform_sf_into_required_layout(B[1], a.n, a.k, (1,1,32), is_sfa=False, num_groups=a.groups))
tensor_bytes = sum(t.numel() * t.element_size() for operand in (A, B, D) for t in (operand if isinstance(operand, tuple) else (operand,)))
def launch():
    dg.m_grouped_fp8_fp4_gemm_nt_contiguous(A, B, D, layout, expected_m_for_psum_layout=a.m)

def verify():
    actual = D.reshape(a.groups, mpad, a.n)[:, :a.m].cpu()
    assert torch.equal(actual, ref), (int((actual != ref).sum()), (actual.float()-ref.float()).abs().max().item())
    return hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest()

launch()
torch.npu.synchronize()
digest = verify()
a.output.mkdir(parents=True, exist_ok=False)
profiles = [bench_msprof(launch, kernel_names='gemm_impl', num_warmups=10, num_tests=30)
            for _ in range(a.rounds)]
assert verify() == digest
latency_us = statistics.median(x.dur_us for x in profiles)
result = dict(case=vars(a) | {'output': str(a.output)}, format='ND',
              output_sha256=digest, padded_m_total=a.groups*mpad, tensor_bytes=tensor_bytes,
              latency_us=latency_us,
              computation_tflops=2*a.groups*mpad*a.n*a.k/latency_us/1e6,
              effective_tflops=2*a.groups*a.m*a.n*a.k/latency_us/1e6,
              memory_bandwidth_gbps=tensor_bytes/latency_us/1000,
              windows=[dataclasses.asdict(x) for x in profiles])
(a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
