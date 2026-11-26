"""Collect MegaMoE msprof PIPE traces without extra kernel markers

    python scripts/profiling/msprof_mega_moe.py --case 7168,3072,64,6,128,1 --output profiling/pipes
    python scripts/profiling/msprof_mega_moe.py --merge --output profiling/pipes
    python scripts/profiling/msprof_mega_moe.py --capture profiling/pipes --export-ranks 0 8 16 24 --output profiling/reexport

Online capture writes node_<node>/rank_<global_rank>.json and merges automatically on one node
Use --merge after all nodes finish; offline --capture export writes rank_<global_rank>.json directly
Each trace selects the last MegaMoE call and sampled cores; missing PIPE records can look idle
BIU timings describe the profiled execution, not unprofiled end-to-end performance
"""
import argparse
import collections
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def worker(local_rank, args):
    # Offline export and --help do not need PyTorch or an NPU environment
    sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
    import torch
    import torch.distributed as dist
    from test_mega_moe import MegaMoECase
    from deep_gemm.testing.envs import init_dist
    from deep_gemm.testing.bench import flush_l2_cache

    rank, world_size, group = init_dist(local_rank, args.ranks)
    metadata = dict(rank=rank, local_rank=local_rank, world_size=world_size,
                    node_rank=int(os.getenv('RANK', 0)), pid=os.getpid(), case=args.case)
    (args.output / 'workers' / f'{os.getpid()}.json').write_text(json.dumps(metadata))
    case = None
    try:
        case = MegaMoECase(group, *args.case, with_baseline=False)
        case.create_inputs(args.case[4])
        for _ in range(args.warmups):
            case.run_fused()
        torch.npu.synchronize()
        for _ in range(args.iters):
            flush_l2_cache()
            dist.barrier(group)
            case.run_fused()
            torch.npu.synchronize()
        print(json.dumps(dict(rank=rank, case=args.case, captured_launches=args.iters,
                              instrumented=False, status='CAPTURE_OK')), flush=True)
    finally:
        if case is not None:
            case.destroy()
        dist.destroy_process_group()


PIPE = {'SU': 'S', 'CUBE': 'M', 'MTE1': 'MTE1', 'MTE2': 'MTE2',
        'FIXP': 'FIX', 'VEC': 'V', 'MTE3': 'MTE3'}
SUBCORE = {'aic': 'AIC', 'aiv0': 'AIV0', 'aiv1': 'AIV1'}


def export(capture, output, rank):
    traces = list(capture.glob('mindstudio_profiler_output/msprof_*.json'))
    if len(traces) != 1:
        raise ValueError(f'expected one exported msprof timeline in {capture}')
    events = json.loads(traces[0].read_text())
    if isinstance(events, dict):
        events = events['traceEvents']
    tasks = [e for e in events if e.get('ph') == 'X' and 'mega_moe_impl' in e.get('name', '')]
    if not tasks:
        raise ValueError('no MegaMoE kernel found in the profiler timeline')
    # Warmup calls are also captured; export the last cold-L2 launch
    target = max(reversed(tasks), key=lambda event: Decimal(str(event['ts'])))
    origin = Decimal(str(target['ts']))
    duration = Decimal(str(target['dur']))
    device = next(capture.glob('device_*'))
    device_id = int(device.name.split('_')[-1])
    info_path = next(f for f in device.glob('info.json.*') if f.suffix[1:].isdigit())
    info = json.loads(info_path.read_text())['DeviceInfo'][0]
    ticks_per_us = Decimal(info['hwts_frequency'])
    lanes = {}
    for event in events:
        if event.get('name') == 'thread_name':
            match = re.fullmatch(r'Group(\d+)-(aic|aiv0|aiv1)', event['args']['name'])
            if match:
                lanes[event['pid'], event['tid']] = (int(match[1]), match[2])
    # Infer each group's block from the most frequent AIC block in the middle 80% of the launch
    target_blocks = collections.defaultdict(collections.Counter)
    for event in events:
        lane = lanes.get((event.get('pid'), event.get('tid')))
        if lane and lane[1] == 'aic' and event.get('ph') == 'X' and 'Block Id' in event.get('args', {}):
            relative = Decimal(str(event['ts'])) - origin
            if duration / 10 < relative < duration * 9 / 10:
                target_blocks[lane[0]][event['args']['Block Id']] += 1
    # Groups without AIC evidence retain all overlapping block events
    expected_block = {group: counts.most_common(1)[0][0] for group, counts in target_blocks.items()}
    other_kernel_events = 0
    rows = collections.defaultdict(collections.deque)
    db = device / 'sqlite/biu_perf.db'
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as connection:
        for group, core, block, name, ts, dur, checkpoint in connection.execute(
                'select * from BiuInstrStatus order by rowid'):
            rows[group, core].append((block, name, Decimal(str(dur)), checkpoint))
    tracks = collections.defaultdict(list)
    boundary_clipped = collections.Counter()
    matched = 0
    for event in events:
        lane = lanes.get((event.get('pid'), event.get('tid')))
        if not lane or event.get('ph') != 'X' or 'Core Type' not in event.get('args', {}):
            continue
        block, name, db_duration, checkpoint = rows[lane].popleft()
        if (name, block) != (event['name'], event['args']['Block Id']):
            raise ValueError('msprof JSON and BIU DB event ordering differ')
        matched += 1
        if checkpoint is not None:
            continue
        start = Decimal(str(event['ts'])) - origin
        # BIU DB durations are hardware ticks; JSON timestamps are microseconds
        end = start + db_duration / ticks_per_us
        if end <= 0 or start >= duration or end <= start:
            continue
        if lane[0] in expected_block and block != expected_block[lane[0]]:
            other_kernel_events += 1
            continue
        key = (block, SUBCORE[lane[1]], PIPE[name])
        if start < 0 or end > duration:
            boundary_clipped[key] += 1
        start, end = max(start, Decimal(0)), min(end, duration)
        tracks[key].append([float(start), float(end-start)])
    if any(rows.values()) or not tracks:
        raise ValueError('unmatched BIU records or no PIPE events in the selected kernel')

    result_events = [dict(ph='M', name='process_name', pid=rank, tid=0,
                          args=dict(name=f'rank {rank}: msprof PIPE timeline')),
                     dict(ph='X', cat='msprof.kernel', name='mega_moe_impl', pid=rank,
                          tid=9999, ts=0, dur=float(duration))]
    statistics = []
    for tid, (key, spans) in enumerate(sorted(tracks.items())):
        core, subcore, pipe = key
        spans.sort()
        track_label = f'core {core:02d} / {subcore} / PIPE_{pipe}'
        result_events.append(dict(ph='M', name='thread_name', pid=rank, tid=tid,
                                  args=dict(name=track_label)))
        for start, span in spans:
            result_events.append(dict(ph='X', cat='msprof.pipe', name=f'PIPE_{pipe}',
                                      pid=rank, tid=tid, ts=start, dur=span,
                                      args=dict(core=core, subcore=subcore)))
        # Busy and idle describe retained events; missing records can appear as idle
        end, busy, longest_idle = 0.0, 0.0, 0.0
        for start, span in spans:
            longest_idle = max(longest_idle, start - end)
            busy += max(0.0, start + span - max(start, end))
            end = max(end, start + span)
        longest_idle = max(longest_idle, float(duration) - end)
        statistics.append(dict(core=core, subcore=subcore, pipe=pipe, events=len(spans),
                               busy_us=busy, busy_ratio=busy / float(duration), max_idle_us=longest_idle,
                               first_us=spans[0][0], last_us=end,
                               boundary_clipped=boundary_clipped[key]))
    trace_path = output / f'rank_{rank}.json'
    summary_path = output / f'rank_{rank}_summary.json'
    for path in (trace_path, summary_path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    output.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(dict(traceEvents=result_events, displayTimeUnit='us'),
                                     separators=(',', ':')))
    summary = dict(rank=rank, device_id=device_id, target_calls=len(tasks), selected_call=len(tasks)-1,
                   task_duration_us=float(duration), instrumented=False, source=str(traces[0]),
                   ignored_other_kernel_events=other_kernel_events, expected_blocks=expected_block,
                   matched_biu_events=matched, exported_pipe_events=sum(len(v) for v in tracks.values()),
                   sampled_cores=sorted({k[0] for k in tracks}), tracks=statistics)
    summary_path.write_text(json.dumps(summary, indent=2))
    return {k:v for k,v in summary.items() if k!='tracks'}


def capture_rank(capture):
    # Use worker PID metadata for global ranks; legacy captures fall back to device IDs
    host_info = json.loads((capture / 'host/info.json').read_text())
    metadata = capture.parent.parent / 'workers' / f"{host_info['pid']}.json"
    if metadata.exists():
        return json.loads(metadata.read_text())['rank']
    if capture.parent.parent.name.startswith('node_'):
        raise ValueError(f'missing global-rank metadata: {metadata}')
    return int(next(capture.glob('device_*')).name.split('_')[-1])


def merge(output):
    nodes = sorted(output.glob('node_*'))
    if not nodes:
        raise ValueError(f'no node outputs found in {output}')
    workers = [json.loads(path.read_text()) for node in nodes for path in (node / 'workers').glob('*.json')]
    world_sizes = {worker['world_size'] for worker in workers}
    if len(world_sizes) != 1:
        raise ValueError('missing or inconsistent worker metadata')
    expected = set(range(world_sizes.pop()))
    if {worker['rank'] for worker in workers} != expected or len(workers) != len(expected):
        raise ValueError('incomplete or duplicate workers')
    summaries = [entry for node in nodes for entry in json.loads((node / 'summary.json').read_text())]
    if {entry['rank'] for entry in summaries} != expected or len(summaries) != len(expected):
        raise ValueError('incomplete or duplicate rank summaries')
    summaries.sort(key=lambda entry: entry['rank'])
    path = output / 'summary.json'
    with path.open('x') as stream:
        json.dump(summaries, stream, indent=2)
    print(json.dumps(dict(ranks=len(summaries), slowest_profiled_rank=max(summaries, key=lambda e: e['task_duration_us'])['rank'],
                         summary=str(path))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='shared output directory, with a separate directory per node')
    parser.add_argument('--capture', type=Path, help='export an existing PROF directory or a complete multi-node capture')
    parser.add_argument('--merge', action='store_true', help='validate and merge completed node summaries without reparsing traces')
    parser.add_argument('--export-ranks', type=int, nargs='+', help='select global ranks for offline export')
    parser.add_argument('--ranks', type=int, default=8, help='number of local workers per node; use 8 on each of four nodes for EP32')
    parser.add_argument('--case', type=lambda s: tuple(map(int, s.split(','))),
                        default=(4096, 2048, 64, 6, 8192, 1), metavar='H,I,E,K,T,S')
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--iters', type=int, default=1, help='cold-L2 launches; export the last launch')
    parser.add_argument('--launch-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.ranks < 1 or args.warmups < 0 or args.iters < 1 or len(args.case) != 6:
        parser.error('expected positive ranks/iters, nonnegative warmups and H,I,E,K,T,S')
    if args.export_ranks is not None and args.capture is None:
        parser.error('--export-ranks requires --capture; all ranks must participate in live collection')
    args.output = args.output.resolve()
    if args.merge:
        if args.capture is not None:
            parser.error('--merge and --capture are mutually exclusive')
        merge(args.output)
        return
    if args.launch_only:
        import torch
        torch.multiprocessing.spawn(worker, args=(args,), nprocs=args.ranks)
        return
    if args.capture is None:
        num_nodes, node_rank = int(os.getenv('WORLD_SIZE', 1)), int(os.getenv('RANK', 0))
        if not 0 <= node_rank < num_nodes or args.case[2] % (num_nodes * args.ranks):
            parser.error('invalid node rank or expert count not divisible by total ranks')
        node_output = args.output / f'node_{node_rank}'
        node_output.mkdir(parents=True, exist_ok=False)
        (node_output / 'workers').mkdir()
        args.capture = node_output / 'raw'
        args.capture.mkdir()
        subprocess.run(['msprof', '--instr-profiling=on', f'--output={args.capture}',
                        sys.executable, str(Path(__file__).resolve()), '--launch-only', '--output', str(node_output),
                        '--ranks', str(args.ranks), '--case', ','.join(map(str, args.case)),
                        '--warmups', str(args.warmups), '--iters', str(args.iters)], cwd=ROOT, check=True)
        output = node_output
    else:
        output = args.output
        num_nodes = 0
    summary_path = output / 'summary.json'
    if summary_path.exists():
        raise FileExistsError(f'Refusing to overwrite {summary_path}')
    captures = [args.capture] if args.capture.name.startswith('PROF_') else sorted(args.capture.rglob('PROF_*'))
    captures = [capture for capture in captures if capture.is_dir()]
    if not captures:
        raise ValueError(f'no PROF captures found in {args.capture}')
    summaries = []
    seen = set()
    for capture in captures:
        rank = capture_rank(capture)
        if args.export_ranks is not None and rank not in args.export_ranks:
            continue
        if rank in seen:
            raise ValueError(f'duplicate global rank {rank}')
        seen.add(rank)
        summary = export(capture, output, rank)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    if args.export_ranks is not None and seen != set(args.export_ranks):
        raise ValueError(f'missing requested ranks: {set(args.export_ranks) - seen}')
    summary_path.write_text(json.dumps(sorted(summaries, key=lambda entry: entry['rank']), indent=2))
    if num_nodes == 1:
        merge(args.output)


if __name__ == '__main__':
    main()
