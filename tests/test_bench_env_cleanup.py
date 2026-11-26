"""Host-only tests for profiler environment cleanup."""

import importlib.util
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


class _Profile:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class ProfilerEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(__file__).resolve().parents[1]

        package = ModuleType('deep_gemm')
        package.__path__ = []
        extension = ModuleType('deep_gemm._C')
        extension.npu_sleep = Mock()

        torch = ModuleType('torch')
        torch.npu = SimpleNamespace(synchronize=Mock())

        profiler = ModuleType('torch_npu.profiler')
        profiler.ProfilerActivity = SimpleNamespace(NPU='npu')
        profiler.ProfilerLevel = SimpleNamespace(Level1='level1')
        profiler.AiCMetrics = SimpleNamespace(PipeUtilization='pipe')
        profiler.schedule = Mock(return_value=None)
        profiler._ExperimentalConfig = Mock(return_value=None)
        profiler.profile = Mock(return_value=_Profile())
        torch_npu = ModuleType('torch_npu')
        torch_npu.__path__ = []
        torch_npu.profiler = profiler

        module_name = 'deep_gemm.testing.bench'
        modules = {
            'deep_gemm': package,
            'deep_gemm._C': extension,
            'torch': torch,
            'torch_npu': torch_npu,
            'torch_npu.profiler': profiler,
        }
        self.stack.enter_context(patch.dict(sys.modules, modules))
        spec = importlib.util.spec_from_file_location(
            module_name, root / 'deep_gemm/testing/bench.py')
        self.bench = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = self.bench
        self.stack.callback(sys.modules.pop, module_name, None)
        spec.loader.exec_module(self.bench)

    def run_failing_profile(self):
        def fail_inside_profile():
            work_path = Path(os.environ['ASCEND_WORK_PATH'])
            self.assertTrue(work_path.is_dir())
            raise RuntimeError('profile failure')

        with self.assertRaisesRegex(RuntimeError, 'profile failure'):
            self.bench.bench_msprof(
                fail_inside_profile,
                kernel_names='kernel',
                num_warmups=0,
                num_tests=1,
                flush_l2=False,
                backend='fast',
                suppress_verbose_output=False,
            )

    def test_failure_restores_existing_work_path(self):
        with patch.dict(os.environ, {'ASCEND_WORK_PATH': '/caller/work'}, clear=False):
            self.run_failing_profile()
            self.assertEqual(os.environ['ASCEND_WORK_PATH'], '/caller/work')

    def test_failure_removes_injected_work_path(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ASCEND_WORK_PATH', None)
            self.run_failing_profile()
            self.assertIsNone(os.environ.get('ASCEND_WORK_PATH'))


if __name__ == '__main__':
    unittest.main()
