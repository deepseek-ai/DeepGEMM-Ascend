import atexit
import csv
import ctypes
import fnmatch
import inspect
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Literal

__all__ = [
    'KernelProfile',
    'bench_msprof',
    'close_persistent_profiler',
]

@dataclass
class KernelProfile:
    """Per-kernel profiling result with AIC and AIV pipe utilization.
    All pipe fields are ratios in [0, 1] relative to their own total_cycles."""
    dur_ns: float               # duration in nanoseconds
    # AIC pipes
    aic_total_cycles: int = 0
    aic_mad: float = 0.0        # Cube MAD
    aic_scalar: float = 0.0
    aic_mte1: float = 0.0       # L1 -> L0
    aic_mte2: float = 0.0       # GM -> L1
    aic_mte3: float = 0.0       # UB -> GM (on AIC, rarely used)
    aic_fixpipe: float = 0.0    # L0C -> out
    # AIV pipes
    aiv_total_cycles: int = 0
    aiv_vec: float = 0.0        # Vector compute
    aiv_scalar: float = 0.0
    aiv_mte2: float = 0.0       # GM -> UB
    aiv_mte3: float = 0.0       # UB -> GM

    @property
    def dur_us(self) -> float:
        return self.dur_ns / 1000

    def tflops(self, flops: float) -> float:
        return flops / self.dur_ns / 1000

    def gbps(self, bytes_moved: float) -> float:
        return bytes_moved / self.dur_ns

    def __repr__(self):
        def _fmt(label, val):
            return f'{label}={val*100:.1f}%' if val > 0.001 else None
        parts = [f'{self.dur_ns/1000:.1f}us']
        for s in [_fmt('mad', self.aic_mad), _fmt('aic_mte2', self.aic_mte2),
                  _fmt('aic_mte1', self.aic_mte1), _fmt('aic_fix', self.aic_fixpipe),
                  _fmt('aiv_vec', self.aiv_vec), _fmt('aiv_mte2', self.aiv_mte2),
                  _fmt('aiv_mte3', self.aiv_mte3)]:
            if s:
                parts.append(s)
        if self.aic_total_cycles:
            parts.append(f'aic_cycles={self.aic_total_cycles}')
        if self.aiv_total_cycles:
            parts.append(f'aiv_cycles={self.aiv_total_cycles}')
        return f'KernelProfile({", ".join(parts)})'


@dataclass
class _PipeSample:
    dur_ns: float = 0.0
    total_cycles: float = 0.0
    vec: float = 0.0
    mad: float = 0.0
    scalar: float = 0.0
    mte1: float = 0.0
    mte2: float = 0.0
    mte3: float = 0.0
    fixpipe: float = 0.0


@dataclass
class _KernelSamples:
    durations_ns: list[float] = field(default_factory=list)
    aic: list[_PipeSample] = field(default_factory=list)
    aiv: list[_PipeSample] = field(default_factory=list)


def _signed_u64(value):
    return value - 2**64 if value >= 2**63 else value


@dataclass(frozen=True)
class _TaskRecord:
    stream_id: int
    task_id: int
    task_type: int
    name_hash: int

    # One-shot files and persistent slices share this CANN v15 layout.
    _struct = struct.Struct('<HHIIIQHHHHQQ16B')

    @classmethod
    def from_bytes(cls, data: bytes) -> list['_TaskRecord']:
        if len(data) % cls._struct.size:
            raise ValueError(f'incomplete task record data: {len(data)} bytes')
        return [cls._from_values(values) for values in cls._struct.iter_unpack(data)]

    @classmethod
    def _from_values(cls, values):
        (
            _, _, _, _, _, _, _, stream_id, task_id_low, task_id_high,
            task_type, name_hash, *_,
        ) = values
        return cls(
            stream_id=stream_id,
            task_id=task_id_low | task_id_high << 16,
            task_type=task_type,
            name_hash=_signed_u64(name_hash),
        )


@dataclass(frozen=True)
class _PmuRecord:
    task_id: int
    total_cycles: int
    is_main: bool
    is_aiv: bool
    start_ns: int
    end_ns: int
    vec: int
    mad: int
    scalar: int
    mte1: int
    mte2: int
    mte3: int
    fixpipe: int

    _struct = struct.Struct('<4HQ2BH2B3HL12Q')
    counter_fields = ('vec', 'mad', 'scalar', 'mte1', 'mte2', 'mte3', 'fixpipe')

    @classmethod
    def from_bytes(cls, data: bytes) -> list['_PmuRecord']:
        if len(data) % cls._struct.size:
            raise ValueError(f'incomplete PMU record data: {len(data)} bytes')
        return [cls._from_values(values) for values in cls._struct.iter_unpack(data)]

    @classmethod
    def _from_values(cls, values):
        (
            _, _, task_id_low, task_id_high, total_cycles,
            _, flags, _, core_type, _, _, _, _, _,
            vec, mad, scalar, mte1, mte2, mte3, _, _, fixpipe, _,
            start_ns, end_ns,
        ) = values
        return cls(
            task_id=task_id_low | task_id_high << 16,
            total_cycles=total_cycles,
            is_main=bool(flags & 1),
            is_aiv=bool(core_type & 1),
            start_ns=start_ns,
            end_ns=end_ns,
            vec=vec,
            mad=mad,
            scalar=scalar,
            mte1=mte1,
            mte2=mte2,
            mte3=mte3,
            fixpipe=fixpipe,
        )

    @property
    def dur_ns(self) -> int:
        return self.end_ns - self.start_ns

    def to_pipe_sample(self) -> _PipeSample:
        if self.total_cycles <= 0:
            raise ValueError(f'PMU record for task {self.task_id} has no cycles')
        return _PipeSample(
            dur_ns=float(self.dur_ns),
            total_cycles=self.total_cycles,
            **{
                name: getattr(self, name) / self.total_cycles
                for name in self.counter_fields
            },
        )


def _decode_hash_line(line):
    try:
        value, name = line.decode(errors='replace').split(':', 1)
        return _signed_u64(int(value)), name
    except ValueError:
        return None


def _mean_field(samples, name):
    return sum(getattr(sample, name) for sample in samples) / len(samples) if samples else 0.0


def _make_profile(samples):
    duration_samples = samples.durations_ns
    if not duration_samples:
        duration_samples = [sample.dur_ns for sample in (samples.aic or samples.aiv)]
    if not duration_samples:
        raise RuntimeError('cannot build a profile without duration samples')

    return KernelProfile(
        dur_ns=sum(duration_samples) / len(duration_samples),
        aic_total_cycles=int(_mean_field(samples.aic, 'total_cycles')),
        aic_mad=_mean_field(samples.aic, 'mad'),
        aic_scalar=_mean_field(samples.aic, 'scalar'),
        aic_mte1=_mean_field(samples.aic, 'mte1'),
        aic_mte2=_mean_field(samples.aic, 'mte2'),
        aic_mte3=_mean_field(samples.aic, 'mte3'),
        aic_fixpipe=_mean_field(samples.aic, 'fixpipe'),
        aiv_total_cycles=int(_mean_field(samples.aiv, 'total_cycles')),
        aiv_vec=_mean_field(samples.aiv, 'vec'),
        aiv_scalar=_mean_field(samples.aiv, 'scalar'),
        aiv_mte2=_mean_field(samples.aiv, 'mte2'),
        aiv_mte3=_mean_field(samples.aiv, 'mte3'),
    )


def _make_profiles(samples_by_name):
    return {name: _make_profile(samples) for name, samples in samples_by_name.items()}


@contextmanager
def suppress_stdout_stderr(suppress: bool):
    if not suppress:
        yield
        return

    libc = ctypes.CDLL(None)
    libc.fflush.argtypes = [ctypes.c_void_p]
    libc.fflush.restype = ctypes.c_int
    sys.stdout.flush()
    sys.stderr.flush()
    libc.fflush(None)
    with open(os.devnull, 'w') as devnull:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with redirect_stderr(devnull), redirect_stdout(devnull):
                yield
        finally:
            # Flush C stdio before restoring the original file descriptors
            libc.fflush(None)
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _make_l2_flush_buffer():
    import torch

    properties = torch.npu.get_device_properties(torch.npu.current_device())
    num_bytes = 2 * properties.L2_cache_size
    return torch.empty(num_bytes // 4, dtype=torch.int, device='npu')


def _select_profiles(profiles, kernel_names, return_all_kernels):
    if return_all_kernels:
        return profiles
    if isinstance(kernel_names, str):
        matched = {name: profile for name, profile in profiles.items() if kernel_names in name}
        if len(matched) != 1:
            summary = '\n'.join(f'  - {name}' for name in profiles) or '  (none)'
            raise RuntimeError(f"No unique kernel matching '{kernel_names}':\n{summary}")
        return next(iter(matched.values()))
    return [next((profile for name, profile in profiles.items() if query in name), None)
            for query in kernel_names]


def bench_msprof(
        fn,
        kernel_names: str | list[str] = None,
        num_warmups: int = 10,
        num_tests: int = 30,
        flush_l2: bool = True,
        barrier_comm_profiling: bool = False,
        barrier: Callable[[], None] | None = None,
        return_all_kernels: bool = False,
        suppress_verbose_output: bool = True,
        backend: Literal['slow', 'fast', 'persistent'] = 'persistent',
    ) -> KernelProfile | list[KernelProfile | None] | dict[str, KernelProfile]:
    """Profile kernel(s) with pipe utilization from a selected PMU backend.

    Args:
        kernel_names: str  -> match one kernel by substring, return KernelProfile
                      list[str] -> return each kernel in list (or None if not found)
        flush_l2: If True, flush L2 cache between iterations.
        barrier_comm_profiling: If True, call barrier() before each iteration.
        barrier: Optional callable to synchronize across ranks (e.g. dist.barrier).
        return_all_kernels: If True, return dict[str, KernelProfile] of every kernel
                      captured in the profiled window. Note this includes the helper
                      kernels launched inside the window (L2 flush, npu_sleep, the
                      all_reduce used for barrier profiling), so use kernel_names to
                      pick the kernel(s) of interest for a clean measurement.
        suppress_verbose_output: If True, suppress torch_npu.profiler output.
        backend: ``slow`` parses exported CSV, ``fast`` parses one-shot raw data,
                 and ``persistent`` reuses one process-resident raw PMU session.

    Returns:
        - KernelProfile                if kernel_names is a str
        - list[KernelProfile | None]   if kernel_names is a list[str]
        - dict[str, KernelProfile]     if return_all_kernels=True
    """
    if kernel_names is None and not return_all_kernels:
        raise ValueError('kernel_names is required unless return_all_kernels=True')
    if backend not in ('slow', 'fast', 'persistent'):
        raise ValueError(f'Unsupported profiler backend: {backend}')
    if num_warmups < 0 or num_tests <= 0:
        raise ValueError('num_warmups must be non-negative and num_tests must be positive')
    barrier_comm_profiling &= int(os.environ.get('EP_DISABLE_BARRIER_PROFILING', 0)) == 0

    import torch
    from deep_gemm._C import npu_sleep

    npu_sleep(0) # warmup npu sleep
    for _ in range(num_warmups):
        fn()
    torch.npu.synchronize()

    flush_buffer = _make_l2_flush_buffer() if flush_l2 else None
    profile_barrier = barrier if barrier_comm_profiling else None
    if barrier_comm_profiling and profile_barrier is None:
        import torch.distributed as dist

        dummy = torch.ones(1, dtype=torch.float, device='npu')
        profile_barrier = partial(dist.all_reduce, dummy)

    def profile_fn():
        for _ in range(num_tests):
            if flush_buffer is not None:
                flush_buffer.zero_()
            if profile_barrier is not None:
                # 20ms, keep the device busy so the barrier dominates instead
                # of kernel launch gaps.
                npu_sleep(int(2e7))
                profile_barrier()
            fn()

    # Both resident and one-shot backends execute the same profiling workload.
    # The resident profiler only provides markers and PMU collection around it.
    if backend == 'persistent':
        profiles = _profile_persistent(profile_fn, suppress_verbose_output)
        return _select_profiles(profiles, kernel_names, return_all_kernels)

    close_persistent_profiler()
    import torch_npu.profiler

    # Use ASCEND_WORK_PATH to control where the profiler dumps temp files.
    old_work_path = os.environ.get('ASCEND_WORK_PATH')
    try:
        with tempfile.TemporaryDirectory(prefix='dg_bench_npu_') as prof_dir:
            os.environ['ASCEND_WORK_PATH'] = prof_dir

            if backend == 'fast':
                def on_trace_ready(_prof):
                    pass
            else:
                on_trace_ready = torch_npu.profiler.tensorboard_trace_handler(str(prof_dir))

            with suppress_stdout_stderr(suppress_verbose_output):
                with torch_npu.profiler.profile(
                    activities=[ torch_npu.profiler.ProfilerActivity.NPU ],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                    on_trace_ready=on_trace_ready,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False,
                        data_simplification=False,
                    ),
                ) as prof:
                    profile_fn()
                    torch.npu.synchronize()
                    prof.step()

                prof_path = Path(prof.prof_if.prof_path)
                if backend == 'fast':
                    profiles = _parse_ffts_profile(prof_path)
                else:
                    profiles = _parse_kernel_details_csv(prof_path)
    finally:
        if old_work_path is None:
            os.environ.pop('ASCEND_WORK_PATH', None)
        else:
            os.environ['ASCEND_WORK_PATH'] = old_work_path

    return _select_profiles(profiles, kernel_names, return_all_kernels)


def _parse_ffts_profile(prof_path: Path) -> dict[str, KernelProfile]:
    """Parse ffts_profile.data and correlate with kernel names.

    Returns {kernel_name: KernelProfile} with averaged metrics across iterations.
    """
    # Build name lookup from hash_dic + task_track
    hash_to_name: dict[int, str] = {}
    for f in prof_path.rglob('*hash_dic.slice_*'):
        if f.name.endswith('.done'):
            continue
        for line in f.read_bytes().splitlines():
            decoded = _decode_hash_line(line)
            if decoded is not None:
                value, name = decoded
                hash_to_name[value] = name

    task_to_name: dict[int, str] = {}
    for f in prof_path.rglob('*task_track.slice_*'):
        if f.name.endswith('.done'):
            continue
        tasks, _tail = _decode_available(f.read_bytes(), _TaskRecord)
        for task in tasks:
            if task.name_hash != 0:
                task_to_name[task.task_id] = hash_to_name.get(
                    task.name_hash, f'<unknown {task.name_hash}>')

    # Parse ffts_profile records (128 bytes each)
    # Mix kernels produce AIC and AIV records for the same task ID.
    kernels: dict[str, _KernelSamples] = defaultdict(_KernelSamples)
    for f in prof_path.rglob('ffts_profile*'):
        if f.name.endswith('.done') or f.stat().st_size == 0:
            continue
        pmus, _tail = _decode_available(f.read_bytes(), _PmuRecord)
        for pmu in pmus:
            if pmu.total_cycles == 0:
                continue
            name = task_to_name.get(pmu.task_id, '<unknown>')
            sample = pmu.to_pipe_sample()
            if pmu.is_aiv:
                kernels[name].aiv.append(sample)
            else:
                kernels[name].aic.append(sample)

    return _make_profiles(kernels)


def _parse_kernel_details_csv(prof_path: Path) -> dict[str, KernelProfile]:
    """Parse kernel_details.csv produced by msprof analyse.

    Returns {kernel_name: KernelProfile} with averaged metrics across iterations.
    The CSV has per-task rows with columns like:
      Name, Duration(us), aic_total_cycles, aic_mac_ratio, aic_scalar_ratio,
      aic_mte1_ratio, aic_mte2_ratio, aic_mte3_ratio, aic_fixpipe_ratio,
      aiv_total_cycles, aiv_vec_ratio, aiv_scalar_ratio, aiv_mte2_ratio, aiv_mte3_ratio
    """
    # Find kernel_details.csv (in ASCEND_PROFILER_OUTPUT/)
    csv_files = list(prof_path.rglob('kernel_details.csv'))
    if not csv_files:
        raise FileNotFoundError(f'kernel_details.csv not found under {prof_path}')
    csv_file = csv_files[0]

    # Parse CSV rows grouped by kernel Name
    records: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Name', '').strip()
            if name:
                records[name].append(row)

    def _float(row, key, default=0.0):
        v = row.get(key, '').strip()
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def _int(row, key, default=0):
        v = row.get(key, '').strip()
        try:
            return int(v)
        except (ValueError, TypeError):
            return default

    kernels: dict[str, _KernelSamples] = {}
    for name, rows in records.items():
        kernel = kernels[name] = _KernelSamples()
        for row in rows:
            kernel.durations_ns.append(_float(row, 'Duration(us)') * 1000)
            kernel.aic.append(_PipeSample(
                total_cycles=_int(row, 'aic_total_cycles'),
                mad=_float(row, 'aic_mac_ratio'),
                scalar=_float(row, 'aic_scalar_ratio'),
                mte1=_float(row, 'aic_mte1_ratio'),
                mte2=_float(row, 'aic_mte2_ratio'),
                mte3=_float(row, 'aic_mte3_ratio'),
                fixpipe=_float(row, 'aic_fixpipe_ratio'),
            ))
            kernel.aiv.append(_PipeSample(
                total_cycles=_int(row, 'aiv_total_cycles'),
                vec=_float(row, 'aiv_vec_ratio'),
                scalar=_float(row, 'aiv_scalar_ratio'),
                mte2=_float(row, 'aiv_mte2_ratio'),
                mte3=_float(row, 'aiv_mte3_ratio'),
            ))
    return _make_profiles(kernels)


_file_patterns = ('*hash_dic.slice_*', '*task_track.slice_*', 'ffts_profile.data.*.slice_*')
_begin_marker = 'sleep_implILj1E'
_end_marker = 'sleep_implILj2E'
_pipe_events = ('0x501', '0x301', '0x1', '0x701', '0x202', '0x203', '0x34', '0x35', '0x714')
_data_types = 0x0004 | 0x0800  # AICORE_METRICS | TASK_TIME_L0
_core_task_types = {0, 66, 108, 109, 600}  # AIC, AIV, MIX_AIC, MIX_AIV, SIMT
_shared_ptr_size = 2 * ctypes.sizeof(ctypes.c_void_p)


class _ReaderVector(ctypes.Structure):
    _fields_ = [
        ('begin', ctypes.c_void_p),
        ('end', ctypes.c_void_p),
        ('capacity_end', ctypes.c_void_p),
    ]


class _ChannelManagerPrefix(ctypes.Structure):
    _fields_ = [
        ('vtable', ctypes.c_void_p),
        ('poller', ctypes.c_void_p),
    ]


class api:
    """Bind a ctypes symbol using the annotations on its declaration."""

    def __init__(self, symbol, check_ret=None):
        self.symbol = symbol
        self.check_ret = check_ret

    def __call__(self, declaration):
        signature = inspect.signature(declaration)
        parameters = list(signature.parameters.values())
        if not parameters or parameters[0].name != 'self':
            raise TypeError('API declarations must be instance methods')
        if any(parameter.annotation is inspect.Parameter.empty for parameter in parameters[1:]):
            raise TypeError(f'{declaration.__name__} has an unannotated argument')
        if signature.return_annotation is inspect.Signature.empty:
            raise TypeError(f'{declaration.__name__} has no return annotation')

        self.name = declaration.__name__
        self.argtypes = [parameter.annotation for parameter in parameters[1:]]
        self.restype = signature.return_annotation
        return self

    def __get__(self, instance, owner):
        if instance is None:
            return self

        function = getattr(instance.library, self.symbol)
        function.argtypes = self.argtypes
        function.restype = self.restype
        if self.check_ret is not None:
            raw_function = function

            def checked_function(*args):
                result = raw_function(*args)
                if result != self.check_ret:
                    raise RuntimeError(f'{self.symbol} failed with {result}')
                return result

            function = checked_function

        setattr(instance, self.name, function)
        return function


class MsProfiler:
    def __init__(self, lib='libmsprofiler.so'):
        self.library = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    @api('aclprofInit', check_ret=0)
    def init(self, output: ctypes.c_char_p, output_len: ctypes.c_size_t) -> ctypes.c_int: ...
    @api('aclprofCreateConfig')
    def create_config(
        self,
        devices: ctypes.POINTER(ctypes.c_uint32), # type: ignore
        device_count: ctypes.c_uint32,
        metric: ctypes.c_int,
        events: ctypes.c_void_p,
        data_types: ctypes.c_uint64,
    ) -> ctypes.c_void_p: ...
    @api('aclprofWarmup', check_ret=0)
    def warmup(self, config: ctypes.c_void_p) -> ctypes.c_int: ...
    @api('aclprofStart', check_ret=0)
    def start(self, config: ctypes.c_void_p) -> ctypes.c_int: ...
    @api('aclprofStop', check_ret=0)
    def stop(self, config: ctypes.c_void_p) -> ctypes.c_int: ...
    @api('aclprofDestroyConfig', check_ret=0)
    def destroy_config(self, config: ctypes.c_void_p) -> ctypes.c_int: ...
    @api('aclprofFinalize', check_ret=0)
    def finalize(self) -> ctypes.c_int: ...


class ProfImpl:
    def __init__(self, lib='libprofimpl.so'):
        self.library = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    @api('_ZN8analysis4dvvp6common9singleton9SingletonIN8Analysis4Dvvp10JobWrapper18ProfChannelManagerEE8instanceEv')
    def channel_manager(self) -> ctypes.c_void_p: ...
    @api('_ZN8analysis4dvvp6common9singleton9SingletonINS0_9transport8HashDataEE8instanceEv')
    def hash_data(self) -> ctypes.c_void_p: ...
    @api('_ZN8analysis4dvvp9transport11ChannelPoll13GetAllReadersEv')
    def get_readers(self, poller: ctypes.c_void_p) -> _ReaderVector: ...
    @api('_ZNSt6vectorISt10shared_ptrIN8analysis4dvvp9transport13ChannelReaderEESaIS5_EED1Ev')
    def destroy_readers(self, readers: ctypes.POINTER(_ReaderVector)) -> None: ... # type: ignore
    @api('_ZN8analysis4dvvp9transport13ChannelReader17FlushBuffToUploadEv')
    def flush_reader(self, reader: ctypes.c_void_p) -> None: ...
    @api('_ZN8analysis4dvvp9transport8HashData15SaveNewHashDataEb')
    def save_hash(self, hash_data: ctypes.c_void_p, force: ctypes.c_bool) -> None: ...


def _decode_available(data, record_type):
    size = len(data) // record_type._struct.size * record_type._struct.size
    return record_type.from_bytes(data[:size]), data[size:]


class _Session:
    """One ACL profiler session plus an incremental raw-record reader."""

    def __init__(self, device):
        self.device = device
        self._lock = threading.Lock()
        self._output = Path(tempfile.mkdtemp(prefix='dg_pmu_', dir='/dev/shm'))
        self._capture = None
        self._config = None
        self._initialized = False
        self._started = False
        self._poller = None
        self._hash_data = None
        self._offsets = {}
        self._hash_names = {}
        self._hash_tail = b''
        self._task_tail = b''
        self._pmu_tail = b''
        self._unresolved_tasks = []
        self._tasks = []
        self._pmu_by_task = defaultdict(list)
        self._ignored_task_ids = set()

        self._open()

    def _open(self):
        existing = set(self._output.glob('PROF_*'))
        try:
            self._start_profiler()
            captures = set(self._output.glob('PROF_*')) - existing
            if len(captures) != 1:
                raise RuntimeError(f'expected one profiler capture, got {len(captures)}')
            self._capture = captures.pop()
            self._validate_metadata()
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _load_profiler(self):
        self._msprof = MsProfiler()
        self._prof_impl = ProfImpl()

    def _start_profiler(self):
        self._load_profiler()
        output = os.fsencode(self._output)
        self._msprof.init(output, len(output))
        self._initialized = True

        device = ctypes.c_uint32(self.device)
        self._config = self._msprof.create_config(
            ctypes.byref(device), 1, 1, None, _data_types)
        if self._config is None:
            raise RuntimeError('aclprofCreateConfig returned null')
        self._msprof.warmup(self._config)
        self._msprof.start(self._config)
        self._started = True

        manager = self._prof_impl.channel_manager()
        self._hash_data = self._prof_impl.hash_data()
        if manager is None or self._hash_data is None:
            raise RuntimeError('could not initialize profiler singletons')
        manager_prefix = ctypes.cast(
            manager, ctypes.POINTER(_ChannelManagerPrefix)).contents
        self._poller = manager_prefix.poller
        if self._poller is None:
            raise RuntimeError('profiler channel poller is not active')

    def _stop_profiler(self):
        errors = []
        if self._config is not None:
            if self._started:
                try:
                    self._msprof.stop(self._config)
                except RuntimeError as error:
                    errors.append(str(error))
                self._started = False
            try:
                self._msprof.destroy_config(self._config)
            except RuntimeError as error:
                errors.append(str(error))
            self._config = None
        if self._initialized:
            try:
                self._msprof.finalize()
            except RuntimeError as error:
                errors.append(str(error))
            self._initialized = False
        self._poller = None
        self._hash_data = None
        if errors:
            raise RuntimeError('; '.join(errors))

    def _validate_metadata(self):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            info_paths = [p for p in self._capture.glob('device_*/info.json.*') if not p.name.endswith('.done')]
            sample_paths = list(self._capture.glob('device_*/sample.json'))
            if info_paths and sample_paths:
                try:
                    info = json.loads(info_paths[0].read_text())
                    sample = json.loads(sample_paths[0].read_text())
                except (OSError, json.JSONDecodeError):
                    pass
                else:
                    version = int(info.get('platform_version', -1))
                    events = tuple(filter(None, sample.get('ai_core_profiling_events', '').split(',')))
                    if version != 15:
                        raise RuntimeError(f'unsupported profiler platform_version {version}')
                    if events == _pipe_events:
                        return
            time.sleep(0.001)
        raise RuntimeError('pipe PMU metadata was not published within 1 second')

    @staticmethod
    def _slice_key(path):
        return str(path.parent), int(path.name.rsplit('_', 1)[1])

    def _read_new(self, paths, pattern):
        chunks = []
        selected = (path for path in paths if fnmatch.fnmatch(path.name, pattern))
        for path in sorted(selected, key=self._slice_key):
            offset = self._offsets.get(path, 0)
            size = path.stat().st_size
            if size < offset:
                raise RuntimeError(f'profiling slice shrank: {path}')
            with path.open('rb') as file:
                file.seek(offset)
                chunk = file.read(size - offset)
            chunks.append(chunk)
            self._offsets[path] = offset + len(chunk)
        return b''.join(chunks)

    def _snapshot(self):
        readers = self._prof_impl.get_readers(self._poller)
        try:
            begin = readers.begin or 0
            end = readers.end or 0
            if end < begin or (end - begin) % _shared_ptr_size != 0:
                raise RuntimeError('invalid profiler reader vector')
            for address in range(begin, end, _shared_ptr_size):
                reader = ctypes.c_void_p.from_address(address).value
                if reader is not None:
                    self._prof_impl.flush_reader(reader)
        finally:
            self._prof_impl.destroy_readers(ctypes.byref(readers))
        self._prof_impl.save_hash(self._hash_data, True)
        self._ingest()

    def _ingest(self):
        paths = [path for path in self._capture.rglob('*slice_*') if not path.name.endswith('.done')]
        hash_data = self._hash_tail + self._read_new(paths, _file_patterns[0])
        lines = hash_data.split(b'\n')
        self._hash_tail = lines.pop() if lines else b''
        for line in lines:
            decoded = _decode_hash_line(line)
            if decoded is not None:
                value, name = decoded
                self._hash_names[value] = name

        tasks, self._task_tail = _decode_available(
            self._task_tail + self._read_new(paths, _file_patterns[1]), _TaskRecord)
        for task in tasks:
            if task.name_hash == 0 or task.task_type not in _core_task_types:
                self._ignored_task_ids.add(task.task_id)
            else:
                self._unresolved_tasks.append(
                    (task.stream_id, task.task_id, task.name_hash))
        resolved = 0
        for stream_id, task_id, name_hash in self._unresolved_tasks:
            if name_hash not in self._hash_names:
                break
            self._tasks.append((stream_id, task_id, self._hash_names[name_hash]))
            resolved += 1
        del self._unresolved_tasks[:resolved]

        pmus, self._pmu_tail = _decode_available(
            self._pmu_tail + self._read_new(paths, _file_patterns[2]), _PmuRecord)
        for pmu in pmus:
            if pmu.total_cycles <= 0 or pmu.start_ns >= pmu.end_ns:
                raise RuntimeError(f'invalid PMU record for task {pmu.task_id}')
            if pmu.task_id not in self._ignored_task_ids:
                self._pmu_by_task[pmu.task_id].append(pmu)

    def _consume_until(self, kernel_name, discard=False):
        marker = next((i for i, task in enumerate(self._tasks) if kernel_name in task[2]), None)
        if marker is None:
            return None
        consumed = self._tasks[:marker + 1]
        task_ids = [task[1] for task in consumed]
        if len(set(task_ids)) != len(task_ids):
            raise RuntimeError('task ID collision in profiling window')
        if discard:
            if task_ids[-1] not in self._pmu_by_task:
                return None
            self._ignored_task_ids.update(task_ids)
            for task_id in task_ids:
                self._pmu_by_task.pop(task_id, None)
            del self._tasks[:marker + 1]
            return []
        if any(task_id not in self._pmu_by_task for task_id in task_ids):
            return None

        records = []
        for stream_id, task_id, name in consumed[:-1]:
            records.extend((name, stream_id, task_id, row) for row in self._pmu_by_task.pop(task_id))
        self._pmu_by_task.pop(consumed[-1][1])
        del self._tasks[:marker + 1]
        return records

    def _read_until(self, marker, dummy, discard=False):
        import torch

        deadline = time.monotonic() + 1.0
        while True:
            records = self._consume_until(marker, discard)
            if records is not None:
                return records
            if time.monotonic() >= deadline:
                marker_tasks = [task for task in self._tasks if marker in task[2]]
                marker_has_pmu = bool(marker_tasks and marker_tasks[0][1] in self._pmu_by_task)
                marker_index = next((i for i, task in enumerate(self._tasks) if marker in task[2]), -1)
                missing = [task[2] for task in self._tasks[:marker_index + 1]
                           if task[1] not in self._pmu_by_task]
                raise RuntimeError(
                    f'profiling data did not reach marker {marker!r} within 1 second '
                    f'(tasks={len(self._tasks)}, unresolved={len(self._unresolved_tasks)}, '
                    f'marker_seen={bool(marker_tasks)}, marker_has_pmu={marker_has_pmu}, '
                    f'missing={missing[:5]!r}/{len(missing)})')
            for _ in range(10):
                dummy.zero_()
            torch.npu.synchronize()
            self._snapshot()

    def profile(self, fn, suppress_output):
        import torch
        from deep_gemm._C import npu_sleep

        with self._lock:
            dummy = torch.empty(1, dtype=torch.int32, device='npu')
            with suppress_stdout_stderr(suppress_output):
                npu_sleep(0, marker=1)
                fn()
                npu_sleep(0, marker=2)

                self._read_until(_begin_marker, dummy, discard=True)
                records = self._read_until(_end_marker, dummy)
            return _parse_persistent_records(records)

    def close(self):
        try:
            self._stop_profiler()
        finally:
            shutil.rmtree(self._output, ignore_errors=True)


def _aggregate_pmus(pmus):
    total_cycles = sum(pmu.total_cycles for pmu in pmus)
    return _PipeSample(
        total_cycles=total_cycles / len(pmus),
        **{
            name: sum(getattr(pmu, name) for pmu in pmus) / total_cycles if total_cycles else 0.0
            for name in _PmuRecord.counter_fields
        },
    )


def _parse_persistent_records(records):
    tasks = defaultdict(list)
    for name, stream_id, task_id, pmu in records:
        tasks[(name, stream_id, task_id)].append(pmu)
    kernels: dict[str, _KernelSamples] = defaultdict(_KernelSamples)
    pmus_by_name = defaultdict(list)
    for (name, _stream_id, task_id), pmus in tasks.items():
        main = [pmu for pmu in pmus if pmu.is_main]
        if len(main) != 1:
            raise RuntimeError(f'task {task_id} has {len(main)} main PMU records')
        kernels[name].durations_ns.append(main[0].dur_ns)
        pmus_by_name[name].extend(pmus)

    for name, pmus in pmus_by_name.items():
        aic = [pmu for pmu in pmus if not pmu.is_aiv]
        aiv = [pmu for pmu in pmus if pmu.is_aiv]
        if aic:
            kernels[name].aic.append(_aggregate_pmus(aic))
        if aiv:
            kernels[name].aiv.append(_aggregate_pmus(aiv))
    return _make_profiles(kernels)


_session = None
_session_lock = threading.Lock()


def close_persistent_profiler():
    """Close the process-resident PMU profiler session."""
    global _session
    with _session_lock:
        session, _session = _session, None
    if session is not None:
        session.close()


def _close_at_exit():
    try:
        close_persistent_profiler()
    except Exception:
        pass


def _profile_persistent(fn, suppress_output):
    """Run a callable inside one window of the resident pipe-PMU session."""
    global _session

    import torch
    from deep_gemm._C import npu_sleep

    device = torch.npu.current_device()
    with _session_lock:
        if _session is not None and _session.device != device:
            old, _session = _session, None
            old.close()
        if _session is None:
            # Compile/load the shared begin/end marker before profiling starts.
            npu_sleep(0, marker=1)
            npu_sleep(0, marker=2)
            torch.npu.synchronize()
            _session = _Session(device)
        session = _session
    try:
        return session.profile(fn, suppress_output)
    except BaseException:
        close_persistent_profiler()
        raise


atexit.register(_close_at_exit)
