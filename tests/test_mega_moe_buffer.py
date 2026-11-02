"""Check the NPU descriptors of external MegaMoE storage and its HCCL use.

Run with: python tests/test_mega_moe_buffer.py --num-processes 8
This exercises buffer construction only; it does not compile MegaMoE kernels.
"""
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing
import torch_npu

import deep_gemm
from deep_gemm.testing.envs import init_dist


def check_gather(tensor, group):
    rank, world_size = group.rank(), group.size()
    tensor.fill_(rank + 1)
    output = torch.empty((tensor.shape[0] * world_size, *tensor.shape[1:]),
                         dtype=tensor.dtype, device=tensor.device)
    # Pass the external-storage tensor directly; cloning it would hide the bug.
    dist.all_gather_into_tensor(output, tensor, group=group)
    for peer, chunk in enumerate(output.chunk(world_size)):
        torch.testing.assert_close(chunk, torch.full_like(chunk, peer + 1), rtol=0, atol=0)


def check_buffer(local_rank, num_processes):
    _, world_size, group = init_dist(local_rank, num_processes)
    buffer = None
    try:
        buffer = deep_gemm.SymmBuffer(group, num_experts=world_size * 2,
                                     num_max_tokens_per_rank=8, num_topk=1,
                                     hidden=512, intermediate_hidden=512,
                                     num_shared_experts=1)
        views = (buffer.buffer, buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights,
                 buffer.shared_l1_acts, buffer.shared_l1_acts_sf,
                 buffer.shared_l2_acts, buffer.shared_l2_acts_sf,
                 buffer.l1_acts, buffer.l1_acts_sf, buffer.l2_acts, buffer.l2_acts_sf,
                 buffer.expert_recv_count)
        for tensor in views:
            reference = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            assert torch_npu.get_npu_format(tensor) == torch_npu.get_npu_format(reference), (
                tensor.shape, tensor.dtype)

        # Exercise both from_blob call sites: the owner and a typed layout slice.
        check_gather(buffer.buffer[:64], group)
        check_gather(buffer.topk_weights, group)
        torch.npu.synchronize()
        if local_rank == 0:
            print(f'MegaMoE external storage descriptors and HCCL gather: PASS ({world_size} ranks)')
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-processes', type=int, default=8)
    args = parser.parse_args()
    torch.multiprocessing.spawn(check_buffer, args=(args.num_processes,), nprocs=args.num_processes)
