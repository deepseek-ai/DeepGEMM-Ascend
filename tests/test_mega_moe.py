import argparse
from functools import partial
import importlib.util
from itertools import groupby, product
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing

import deep_gemm
from deep_gemm.testing import bench_msprof
from deep_gemm.testing.envs import dist_print, init_dist
from utils import calc_diff


# DeepEP rounds rank partials to BF16; MegaMoE keeps the full sum in FP32
REFERENCE_DIFF_THRESHOLD = 5e-6


def check_output(output, combined, rms_weight=None, rms_eps=None):
    actual, ref = output.float(), combined.float()
    if rms_weight is not None:
        ref = (ref * torch.rsqrt(ref.square().mean(dim=-1, keepdim=True) + rms_eps) * rms_weight.float()).to(output.dtype).float()
    if rms_weight is not None and combined.dtype == output.dtype == torch.float32:
        torch.testing.assert_close(actual, ref, rtol=1e-5, atol=1e-6)
    diff = calc_diff(actual, ref)
    assert diff < REFERENCE_DIFF_THRESHOLD, f'end-to-end output: diff={diff:.2e}'
    token_diff = ((actual - ref).square().sum(dim=1) /
                  (actual.square() + ref.square()).sum(dim=1).clamp_min(1e-30))
    max_token_diff = token_diff.max().item() if output.size(0) else 0.0
    assert max_token_diff < REFERENCE_DIFF_THRESHOLD, f'per-token output: max_diff={max_token_diff:.2e}'


def make_npu_fp8(shape: tuple[int, ...], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.npu.manual_seed(seed)
    data = (torch.randn(shape, dtype=torch.bfloat16, device='npu') * 32).to(torch.float8_e4m3fn)
    sf_exponent = torch.randint(123, 131, (shape[0], shape[1] // 32),
                                dtype=torch.uint8, device='npu')
    return data, sf_exponent.view(torch.int16)


def make_npu_fp4(shape: tuple[int, int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.npu.manual_seed(seed)
    data = torch.randint(-128, 128, (*shape[:-1], shape[-1] // 2),
                         dtype=torch.int8, device='npu')
    sf_exponent = torch.randint(123, 131, (*shape[:-1], shape[-1] // 32),
                                dtype=torch.uint8, device='npu')
    return data, sf_exponent.view(torch.int16)


def make_topk(num_tokens: int,
              num_topk: int,
              num_experts: int,
              seed: int,
              routing: str = 'uniform',
              routing_seed: int = 0,
              num_ranks: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device='npu').manual_seed(seed)
    if routing == 'uniform':
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32,
                             device='npu', generator=generator)
    elif routing == 'skew':
        # Skewed expert popularity with balanced rank placement and mild rank drift
        u = (torch.arange(num_experts, dtype=torch.float32, device='npu') + 0.5) / num_experts
        weights = (-u.log()).pow(0.80)
        shared_gen = torch.Generator(device='npu').manual_seed(26000 + routing_seed)
        if num_ranks > 1 and num_experts % num_ranks == 0:
            grid = weights.view(num_experts // num_ranks, num_ranks).clone()
            grid[1::2] = grid[1::2].flip(1)
            rank_perm = torch.randn((num_ranks,), dtype=torch.float32, device='npu', generator=shared_gen).argsort()
            rank_drift = torch.randn((num_ranks, 1), dtype=torch.float32, device='npu', generator=shared_gen).mul(0.12).exp()
            weights = (grid.t()[rank_perm] * rank_drift).reshape(-1)
        else:
            perm = torch.randn((num_experts,), dtype=torch.float32, device='npu', generator=shared_gen).argsort()
            weights = weights[perm]
        log_weights = weights.clamp_min(1e-12).log()
        u_rand = torch.empty((num_tokens, num_experts), dtype=torch.float32, device='npu').uniform_(
            1e-7, 1.0 - 1e-7, generator=generator)
        scores = log_weights.unsqueeze(0) - (-u_rand.log()).log()
    else:
        raise ValueError(f'Unknown routing mode: {routing}')
    topk_scores, topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False)
    return topk_idx, torch.softmax(topk_scores, dim=-1)


class MegaMoECase:
    """Own one model's inputs and the fused/unfused runners used by tests and profiling."""

    def __init__(self, group, hidden, intermediate_hidden, num_experts, num_topk,
                 num_max_tokens_per_rank, num_shared_experts=0, activation_clamp=7.0,
                 with_baseline=True, base=None, rms_norm=False,
                 out_dtype=torch.bfloat16, rms_eps=None, routing='uniform'):
        assert not rms_norm or rms_eps is not None, 'rms_eps is required when rms_norm is enabled'
        self.routing = routing
        rank_idx, num_ranks = group.rank(), group.size()
        num_experts_per_rank = num_experts // num_ranks
        self.buffer = deep_gemm.SymmBuffer(
            group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
            num_shared_experts=num_shared_experts, base=base)
        # Match the BF16 clamp consumed by the fused kernel
        self.activation_clamp = torch.tensor(activation_clamp, dtype=torch.bfloat16, device='cpu').item()
        self.rms_eps, self.out_dtype = rms_eps, out_dtype
        generator = torch.Generator(device='npu').manual_seed(26000)
        self.rms_weight = torch.randn((hidden,), dtype=out_dtype, device='npu', generator=generator) if rms_norm else None
        self.l1_weights = make_npu_fp4((num_experts_per_rank, 2 * intermediate_hidden, hidden), 20000 + rank_idx)
        self.l2_weights = make_npu_fp4((num_experts_per_rank, hidden, intermediate_hidden), 21000 + rank_idx)
        self.transformed_l1_weights, self.transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(
            self.l1_weights, self.l2_weights)
        self.shared_l1_weights = self.shared_l2_weights = None
        self.transformed_shared_l1_weights = self.transformed_shared_l2_weights = None
        if num_shared_experts:
            shared_intermediate_hidden = num_shared_experts * intermediate_hidden
            self.shared_l1_weights = make_npu_fp8((2 * shared_intermediate_hidden, hidden), 24000)
            self.shared_l2_weights = make_npu_fp8((hidden, shared_intermediate_hidden), 25000)
            self.transformed_shared_l1_weights, self.transformed_shared_l2_weights = deep_gemm.transform_weights_for_mega_moe(
                self.shared_l1_weights, self.shared_l2_weights)

        self.ep_buffer = None
        if with_baseline:
            import deep_ep
            import tilelang

            tilelang.set_log_level('ERROR')
            tilelang_ops = sys.modules.get('tilelang_ops')
            if tilelang_ops is None:
                ops_dir = Path(__file__).resolve().parents[1] / 'third-party/tilelang_ops'
                spec = importlib.util.spec_from_file_location(
                    'tilelang_ops', ops_dir / 'swiglu_apply_weight_to_fp8.py', submodule_search_locations=[str(ops_dir)])
                tilelang_ops = importlib.util.module_from_spec(spec)
                sys.modules['tilelang_ops'] = tilelang_ops
                spec.loader.exec_module(tilelang_ops)

            # Dispatch, grouped GEMMs and SwiGLU must agree on expert padding
            self.alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
            self.swiglu = partial(tilelang_ops.swiglu_apply_weight_to_fp8,
                                   expert_alignment=self.alignment, clamp_value=self.activation_clamp)
            # Give the DeepEP buffer its own communicator
            self.reference_group = dist.new_group(backend='hccl')
            self.ep_buffer = deep_ep.EPBuffer(
                self.reference_group, num_max_tokens_per_rank=num_max_tokens_per_rank,
                hidden=hidden, num_topk=num_topk, use_fp8_dispatch=True,
                allow_multiple_reduction=True, explicitly_destroy=True)
        else:
            # Fused-only profiling needs only the transformed weight storage
            self.l1_weights = self.l2_weights = None
            self.shared_l1_weights = self.shared_l2_weights = None

    def create_inputs(self, num_tokens, seed_offset=0):
        buffer = self.buffer
        rank_idx = buffer.group.rank()
        self.x = make_npu_fp8((num_tokens, buffer.hidden), 22000 + seed_offset + rank_idx)
        self.topk_idx, self.topk_weights = make_topk(
            num_tokens, buffer.num_topk, buffer.num_experts, 23000 + seed_offset + rank_idx,
            routing=self.routing, routing_seed=seed_offset, num_ranks=buffer.group.size())
        buffer.x[:num_tokens].copy_(self.x[0])
        buffer.x_sf[:num_tokens].copy_(self.x[1])
        buffer.topk_idx[:num_tokens].copy_(self.topk_idx)
        buffer.topk_weights[:num_tokens].copy_(self.topk_weights)
        self.y = torch.empty((num_tokens, buffer.hidden), dtype=self.out_dtype, device='npu')

    def run_fused(self, cumulative_local_expert_recv_stats=None, out=None, use_rms_norm=True, low_latency=None):
        out = self.y if out is None else out
        rms_weight = self.rms_weight if use_rms_norm else None
        deep_gemm.fp8_fp4_mega_moe(
            out, self.transformed_l1_weights, self.transformed_l2_weights, self.buffer,
            shared_l1_weights=self.transformed_shared_l1_weights,
            shared_l2_weights=self.transformed_shared_l2_weights,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats, activation_clamp=self.activation_clamp,
            rms_weight=rms_weight, rms_eps=self.rms_eps if rms_weight is not None else None,
            low_latency=low_latency)
        return out

    def run_baseline(self, synchronize_compile=False):
        buffer = self.buffer
        # Dispatch -> Linear1 -> SwiGLU -> Linear2 -> combine, without communication overlap
        # Allocate actual padded receive rows instead of the EP-scaled worst-case capacity
        recv_x, _, recv_topk_weights, handle, _ = self.ep_buffer.dispatch(
            self.x, topk_idx=self.topk_idx, topk_weights=self.topk_weights,
            num_experts=buffer.num_experts, expert_alignment=self.alignment,
            do_cpu_sync=True, do_expand=True, do_zero_padding=True, use_tma_aligned_col_major_sf=True)
        layout = handle.psum_num_recv_tokens_per_expert
        l1_y = torch.empty((recv_x[0].size(0), 2 * buffer.intermediate_hidden), dtype=torch.bfloat16, device='npu')
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            recv_x, self.l1_weights, l1_y, layout, recipe=(1, 1, 32), use_psum_layout=True)
        l2_x = self.swiglu(l1_y, psum_num_tokens_per_expert=layout, topk_weights=recv_topk_weights)
        l2_y = torch.empty((recv_x[0].size(0), buffer.hidden), dtype=torch.bfloat16, device='npu')
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            l2_x, self.l2_weights, l2_y, layout, recipe=(1, 1, 32), use_psum_layout=True)

        shared_y = None
        if buffer.num_shared_experts:
            shared_l1_y = torch.empty((self.y.size(0), 2 * buffer.num_shared_experts * buffer.intermediate_hidden),
                                      dtype=torch.bfloat16, device='npu')
            shared_y = torch.empty_like(self.y, dtype=torch.bfloat16)
            deep_gemm.fp8_gemm_nt(self.x, self.shared_l1_weights, shared_l1_y, recipe=(1, 1, 32))
            shared_l2_x = self.swiglu(shared_l1_y)
            deep_gemm.fp8_gemm_nt(shared_l2_x, self.shared_l2_weights, shared_y, recipe=(1, 1, 32))
        # Join ranks after cold compilation before DeepEP combine
        if synchronize_compile:
            torch.npu.synchronize()
            dist.barrier(self.reference_group)
        # DeepEP adds shared output during its FP32 combine accumulation
        return self.ep_buffer.combine(l2_y, handle=handle, bias=shared_y)[0]

    def destroy(self):
        if self.ep_buffer is not None:
            self.ep_buffer.destroy()
            dist.destroy_process_group(self.reference_group)
            self.ep_buffer = None
        self.buffer.destroy()


def check_rms_api(case):
    weight = case.rms_weight
    invalid_weights = (weight.to(torch.float32 if weight.dtype == torch.bfloat16 else torch.bfloat16),
                       weight.half(), weight.unsqueeze(0), weight[:-1],
                       weight.repeat_interleave(2)[::2], weight.cpu())
    invalid_cases = [(bad_weight, case.rms_eps, 'rms_weight') for bad_weight in invalid_weights]
    invalid_cases += [(weight, bad_eps, 'rms_eps') for bad_eps in (None, 0.0, -1.0, float('inf'), float('nan'))]
    for rms_weight, rms_eps, name in invalid_cases:
        try:
            deep_gemm.fp8_fp4_mega_moe(
                case.y, case.transformed_l1_weights, case.transformed_l2_weights, case.buffer,
                shared_l1_weights=case.transformed_shared_l1_weights,
                shared_l2_weights=case.transformed_shared_l2_weights,
                rms_weight=rms_weight, rms_eps=rms_eps)
        except RuntimeError as error:
            assert name in str(error), str(error)
        else:
            raise AssertionError(f'Expected RuntimeError for invalid {name}')


def check_correctness(case, num_runs, low_latency=None):
    buffer = case.buffer
    stats = torch.arange(buffer.num_experts // buffer.group.size(), device=case.y.device, dtype=torch.int32) + 17
    expected_stats = stats.clone()
    references = {}
    for run_idx in range(num_runs + 2):
        # Alternate inputs, ending on the original seed for benchmarking
        seed_offset = run_idx % 2 if run_idx < num_runs + 1 else 0
        case.create_inputs(case.y.size(0), seed_offset=seed_offset)
        case.y.fill_(float('nan'))
        output = case.run_fused(stats, low_latency=low_latency)
        if seed_offset not in references:
            if case.rms_weight is not None:
                unnormalized = case.run_fused(out=torch.full_like(output, float('nan')), use_rms_norm=False,
                                              low_latency=low_latency)
                check_output(output, unnormalized, case.rms_weight, case.rms_eps)
            if low_latency is None and seed_offset == 0:
                if case.y.size(0) <= 4096:
                    for mode in (True, False):
                        alt = case.run_fused(out=torch.full_like(output, float('nan')), low_latency=mode)
                        assert torch.equal(alt.view(torch.int16), output.view(torch.int16)), f'low_latency={mode} differs'
                else:
                    try:
                        case.run_fused(out=torch.full_like(output, float('nan')), low_latency=True)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError('Expected RuntimeError for low_latency=True with num_tokens > 4096')
            if case.ep_buffer is not None:
                check_output(output, case.run_baseline(synchronize_compile=True), case.rms_weight, case.rms_eps)

            # Count routed selections independently, including duplicates
            topk_idx = case.topk_idx.cpu().flatten()
            topk_idx = topk_idx[(topk_idx >= 0) & (topk_idx < buffer.num_experts)]
            recv_counts = torch.bincount(topk_idx, minlength=buffer.num_experts).to(device=output.device, dtype=torch.int32)
            dist.all_reduce(recv_counts, group=buffer.group)
            recv_counts = recv_counts.reshape(buffer.group.size(), -1)[buffer.group.rank()]
            references[seed_offset] = output.view(torch.int16).clone(), recv_counts

        expected_bits, recv_counts = references[seed_offset]
        expected_stats += recv_counts
        assert torch.equal(output.view(torch.int16), expected_bits), f'output differs in run {run_idx + 1}'
        assert torch.equal(stats, expected_stats), 'cumulative expert counts differ'


def benchmark(case, skip_baseline_timing=False, low_latency=None):
    profile = partial(bench_msprof, kernel_names='mega_moe_impl', barrier_comm_profiling=True, backend='fast')
    prof = profile(partial(case.run_fused, use_rms_norm=False, low_latency=low_latency))
    on_us = profile(partial(case.run_fused, low_latency=low_latency)).dur_us if case.rms_weight is not None else 0.0
    baseline_us = 0.0
    if case.ep_buffer is not None and not skip_baseline_timing:
        kernel_names = ['dispatch_impl', 'dispatch_copy_epilogue_impl',
                        'fp8_dequant_gemm_impl', 'fp8_gemm_impl', 'swiglu_forward_kernel',
                        'combine_impl', 'combine_reduce_epilogue_impl']
        # Keep every GEMM specialization; list queries in bench_msprof select only the first match
        profiles = bench_msprof(case.run_baseline, return_all_kernels=True, barrier_comm_profiling=True, backend='fast')
        num_swiglu_calls = 1 + (case.buffer.num_shared_experts > 0 and case.y.size(0) > 0)
        baseline_us = sum(profile.dur_us * (num_swiglu_calls if 'swiglu' in name else 1)
                          for name, profile in profiles.items() if any(kernel in name for kernel in kernel_names))
    buffer = case.buffer
    hidden, intermediate = buffer.hidden, buffer.intermediate_hidden
    num_tokens, shared = case.y.size(0), buffer.num_shared_experts
    recv_counts = buffer.expert_recv_count.cpu()
    num_recv_tokens = int(recv_counts.sum())
    num_touched_experts = int((recv_counts.sum(dim=0) > 0).sum())
    flops = 6 * hidden * intermediate * (num_recv_tokens + num_tokens * shared)
    # CUDA's logical HBM model: weights once per active expert + activation reads/writes + output
    # Excludes SFs, metadata, reduction traffic and cache/reload effects; divide by full kernel time
    num_hbm_bytes = 3 * hidden * intermediate * num_touched_experts / 2
    num_hbm_bytes += num_recv_tokens * (3 * hidden + 2 * intermediate)
    if shared and num_tokens:
        num_hbm_bytes += 3 * hidden * intermediate * shared
        num_hbm_bytes += num_tokens * (3 * hidden + 2 * intermediate * shared)
    num_hbm_bytes += num_tokens * hidden * (case.y.element_size() - 2)
    # Remote FP8 pulls + BF16 pushes; exclude local routes, SFs, metadata and cache effects
    num_remote_tokens = num_recv_tokens - int(recv_counts[buffer.group.rank()].sum())
    num_comm_bytes = num_remote_tokens * hidden * 3
    local_stats = torch.tensor((prof.dur_us, baseline_us, flops, num_hbm_bytes, num_comm_bytes, on_us),
                               dtype=torch.float32, device='npu')
    rank_stats = [torch.empty_like(local_stats) for _ in range(buffer.group.size())]
    dist.all_gather(rank_stats, local_stats, group=buffer.group)
    rank_stats = torch.stack(rank_stats).cpu()
    fused_us, baseline_us = rank_stats[:, :2].amax(dim=0).tolist()
    averages = rank_stats.mean(dim=0).tolist()
    # Latency is the slowest rank; throughput uses mean work/bytes per rank over that time
    speedup = f'{baseline_us / fused_us:.3f}x' if baseline_us else 'N/A'
    on_us = rank_stats[:, 5].amax().item()
    rms_prefix = 'OFF ' if case.rms_weight is not None else ''
    rms_perf = f' | ON {on_us:.2f} us | ON/OFF {100 * (on_us / fused_us - 1):+.2f}%' if case.rms_weight is not None else ''
    return (f'{rms_prefix}{fused_us:8.2f} us | speedup {speedup:>6} | {averages[2] / fused_us / 1e6:5.1f} TFLOPS/rank | '
            f'HBM {averages[3] / fused_us / 1e3:6.1f} GB/s/rank | '
            f'Comm {averages[4] / fused_us / 1e3:5.1f} GB/s/rank{rms_perf}')


def test(local_rank, args):
    _, num_ranks, group = init_dist(local_rank, args.num_processes)

    def buffer_key(config):
        (hidden, intermediate), experts, topk, tokens, shared, capacity, _ = config
        capacity = max(tokens, 1) if capacity is None else capacity
        num_bytes = deep_gemm._C.get_symm_buffer_size_for_mega_moe(
            num_ranks, experts, capacity, topk, hidden, intermediate, 'fp8xfp4', 'swiglu', shared)[0]
        return num_bytes, capacity, experts, topk, shared

    # In the Cartesian test matrix, break aligned-size ties by every progress-capacity dimension
    (hidden, intermediate), experts, topk, tokens, shared, capacity, _ = max(args.configs, key=buffer_key)
    base = deep_gemm.SymmBuffer(group, experts, max(tokens, 1) if capacity is None else capacity,
                                topk, hidden, intermediate, num_shared_experts=shared)
    case_width = len(str(len(args.configs)))
    case_idx = 0
    for model, configs in groupby(args.configs, key=lambda config: config[:3] + config[4:]):
        (hidden, intermediate_hidden), num_experts, num_topk, shared, capacity, clamp = model
        configs = list(configs)
        num_max_tokens = capacity if capacity is not None else max(1, max(config[3] for config in configs))
        # Reuse weights and the reference communicator; rebuild fused views for each token count
        case = MegaMoECase(group, hidden, intermediate_hidden, num_experts, num_topk, num_max_tokens, shared,
                           activation_clamp=clamp, with_baseline=not args.skip_baseline and num_ranks > 1,
                           base=base, routing=args.routing, rms_norm=args.rms_norm, rms_eps=args.rms_eps,
                           out_dtype=getattr(torch, args.out_dtype))
        def prepare(num_tokens):
            case.buffer.destroy()
            case.buffer = deep_gemm.SymmBuffer(
                group, num_experts, max(num_tokens, 1) if capacity is None else capacity, num_topk,
                hidden, intermediate_hidden, num_shared_experts=shared, base=base)
            case.create_inputs(num_tokens)

        try:
            # Distribute cold compilation across local workers
            deep_gemm._C.set_dry_run(True)
            try:
                for config in configs[local_rank::args.num_processes]:
                    prepare(config[3])
                    case.run_fused(low_latency=args.low_latency)
                    if args.rms_norm:
                        case.run_fused(use_rms_norm=False, low_latency=args.low_latency)
                    if not args.skip_correctness and args.low_latency is None and config[3] <= 4096:
                        case.run_fused(low_latency=True)
                        case.run_fused(low_latency=False)
            finally:
                deep_gemm._C.set_dry_run(False)
            dist.barrier(group)
            for _, _, _, num_tokens, _, _, _ in configs:
                case_idx += 1
                routing_tag = f' | routing={args.routing}' if args.routing != 'uniform' else ''
                config = (f'[{case_idx:>{case_width}}/{len(args.configs)}] H/I={hidden:4}/{intermediate_hidden:4} | E={num_experts:3} | '
                          f'topk={num_topk:2} | shared={shared} | tokens={num_tokens:5} | RMSNorm={args.rms_norm}{routing_tag}')
                if local_rank == 0:
                    print(f'{config} | ', end='', flush=True)
                prepare(num_tokens)
                if not args.skip_correctness:
                    if case_idx == 1 and case.rms_weight is not None:
                        check_rms_api(case)
                    check_correctness(case, args.num_correctness_tests, low_latency=args.low_latency)
                if not args.skip_prof:
                    dist_print(benchmark(case, args.skip_baseline_timing, low_latency=args.low_latency), once_in_node=True)
                else:
                    dist_print('SKIP' if args.skip_correctness else 'PASS', once_in_node=True)
        finally:
            case.destroy()
            del case
            # DeepEP allocates via ACL, which cannot reclaim PyTorch's cached memory
            torch.npu.empty_cache()
    base.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare MegaMoE against DeepGEMM + DeepEP + TileLang SwiGLU')

    # Resource settings
    parser.add_argument('--num-processes', type=int, default=8)

    # H/I pair by position; singleton lists broadcast, other model lists form a Cartesian product
    parser.add_argument('--hidden', type=int, nargs='+', default=[5120, 7168])
    parser.add_argument('--intermediate-hidden', type=int, nargs='+', default=[2304, 3072])
    parser.add_argument('--num-experts', type=int, nargs='+', default=[384])
    parser.add_argument('--num-topk', type=int, nargs='+', default=[6])
    parser.add_argument('--num-tokens', type=int, nargs='+', default=[64, 256, 4096, 16384], help='0 tests an empty input')
    parser.add_argument('--num-shared-experts', type=int, nargs='+', default=[1])
    parser.add_argument('--num-max-tokens-per-rank', type=int, nargs='+', default=[None],
                        help='default: actual token count, at least 1')
    parser.add_argument('--activation-clamp', type=float, nargs='+', default=[7.0])
    parser.add_argument('--rms-norm', type=int, choices=(0, 1), default=0)
    parser.add_argument('--rms-eps', type=float, default=None)
    parser.add_argument('--out-dtype', choices=('bfloat16', 'float32'), default='bfloat16')

    # Test settings
    parser.add_argument('--routing', choices=['uniform', 'skew'], default='uniform',
                        help='routing distribution: uniform (default) or skew')
    parser.add_argument('--low-latency', choices=('auto', '0', '1'), default='auto',
                        help='override low_latency mode (default: auto)')
    parser.add_argument('--num-correctness-tests', type=int, default=10, help='repeated runs across two inputs; 0 skips correctness')
    parser.add_argument('--skip-correctness', action='store_true')
    parser.add_argument('--skip-prof', action='store_true')
    parser.add_argument('--skip-baseline-timing', action='store_true', help='keep reference correctness but omit reference profiling')
    parser.add_argument('--skip-baseline', action='store_true', help='skip baseline profiling and correctness checks')
    args = parser.parse_args()
    args.low_latency = None if args.low_latency == 'auto' else args.low_latency == '1'
    if args.rms_norm and args.rms_eps is None:
        parser.error('--rms-eps is required when --rms-norm=1')

    args.skip_correctness |= args.num_correctness_tests == 0 or ((args.skip_baseline or args.num_processes == 1) and not args.rms_norm)
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_shapes = max(len(hidden), len(intermediate_hidden))
    assert len(hidden) in (1, num_shapes) and len(intermediate_hidden) in (1, num_shapes), \
        'H/I lists must have equal lengths, or one value for broadcasting'
    shapes = zip(hidden * num_shapes if len(hidden) == 1 else hidden,
                 intermediate_hidden * num_shapes if len(intermediate_hidden) == 1 else intermediate_hidden)
    args.configs = [(shape, experts, topk, tokens, shared, capacity, clamp)
                    for shared, shape, experts, topk, tokens, capacity, clamp in product(
                        args.num_shared_experts, shapes, args.num_experts, args.num_topk, args.num_tokens,
                        args.num_max_tokens_per_rank, args.activation_clamp)]

    print(f'Initializing {args.num_processes} ranks; {len(args.configs)} cases', flush=True)
    torch.multiprocessing.spawn(test, args=(args,), nprocs=args.num_processes)
