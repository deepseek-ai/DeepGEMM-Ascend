"""Run with torchrun to check external NPU storage and borrowed-view lifetime."""
import gc
import os

import torch
import torch.distributed as dist
import torch_npu

import deep_gemm


def main():
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl')
    group = dist.distributed_c10d._get_default_group()
    buffer = deep_gemm.SymmBuffer(group, 128, 16, 3, 768, 1280)
    for tensor in (buffer.buffer, buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights):
        assert torch_npu.get_npu_format(tensor) == 2, 'Symmetric storage must use ND'
    # A retained view must keep the HCCL lease alive after its Python owner dies.
    weights = buffer.topk_weights[:6]
    pointer = weights.data_ptr()
    buffer.destroy()
    del buffer
    gc.collect()
    values = torch.randn_like(weights)
    torch.mul(values, 1.5, out=weights)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        torch.mul(values, 1.5, out=weights)
    for seed in (0, 1):
        torch.manual_seed(seed)
        values.normal_()
        graph.replay()
        torch.npu.synchronize()
        assert weights.data_ptr() == pointer
        assert torch.equal(weights, values * 1.5)
    del graph, weights
    gc.collect()
    dist.barrier()
    print(f'rank {group.rank()}: ND storage and retained graph view PASS', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
