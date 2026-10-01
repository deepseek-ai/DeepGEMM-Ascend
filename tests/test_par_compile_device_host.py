"""Host control-flow tests with a thread-local NPU double, not an NPU runtime test."""

import importlib.util
import sys
import threading
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class ThreadLocalNPU:
    def __init__(self):
        self.local = threading.local()
        self.stream_devices = []

    def current_device(self):
        return getattr(self.local, "device", 0)

    @contextmanager
    def device(self, index):
        previous = self.current_device()
        self.local.device = index
        try:
            yield
        finally:
            self.local.device = previous

    def current_stream(self):
        self.stream_devices.append((threading.get_ident(), self.current_device()))
        return self.current_device()


class ParallelCompileTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "_par_compile_host", root / "deep_gemm/testing/par_compile.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.npu = ThreadLocalNPU()
        self.module.torch = SimpleNamespace(npu=self.npu)
        self.dry_run_states = []
        parent = ModuleType("deep_gemm")
        extension = ModuleType("deep_gemm._C")
        extension.set_dry_run = self.dry_run_states.append
        self.stack.enter_context(
            patch.dict(sys.modules, {"deep_gemm": parent, "deep_gemm._C": extension})
        )

    def test_workers_inherit_callers_nonzero_device(self):
        seen = []
        with self.npu.device(3):
            self.module.par_compile(
                [lambda: seen.append(self.npu.current_device()) for _ in range(8)],
                max_workers=2,
                progress=False,
            )
            self.assertEqual(self.npu.current_device(), 3)
        self.assertEqual(seen, [3] * 8)

    def test_worker_stream_is_initialized_on_selected_device(self):
        caller = threading.get_ident()
        with self.npu.device(2):
            self.module.par_compile([lambda: None], max_workers=1, progress=False)
        workers = [device for tid, device in self.npu.stream_devices if tid != caller]
        self.assertEqual(workers, [2])

    def test_each_invocation_captures_its_own_device(self):
        seen = []
        for index in (1, 4):
            with self.npu.device(index):
                self.module.par_compile(
                    [lambda: seen.append(self.npu.current_device())],
                    max_workers=1,
                    progress=False,
                )
        self.assertEqual(seen, [1, 4])

    def test_failure_propagates_and_dry_run_is_reset(self):
        error = RuntimeError("compile failed")

        def fail():
            raise error

        with self.npu.device(3):
            with self.assertRaises(RuntimeError) as caught:
                self.module.par_compile([fail], max_workers=1, progress=False)
            self.assertIs(caught.exception, error)
            self.assertEqual(self.npu.current_device(), 3)
        self.assertEqual(self.dry_run_states, [True, False])

    def test_empty_input_restores_dry_run(self):
        self.module.par_compile([], max_workers=1, progress=False)
        self.assertEqual(self.dry_run_states, [True, False])


if __name__ == "__main__":
    unittest.main()
