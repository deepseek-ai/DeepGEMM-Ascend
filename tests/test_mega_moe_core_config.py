"""Host-only Python API checks; native layout/kernel tests run separately."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class CoreConfigTest(unittest.TestCase):
    def setUp(self):
        torch = ModuleType('torch')
        torch.Tensor = object
        dist = ModuleType('torch.distributed')
        dist.ProcessGroup = object
        torch.distributed = dist
        package = ModuleType('_mega_moe_core_test')
        package.__path__ = []
        self.native = SimpleNamespace(
            _mega_moe_block_n=256, get_num_sms=Mock(return_value=32),
            get_symm_buffer_size_for_mega_moe=Mock(return_value=(16, lambda _: tuple(range(13)))),
            fp8_fp4_mega_moe=Mock())
        package._C = self.native
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'torch': torch, 'torch.distributed': dist,
                                 '_mega_moe_core_test': package}).start()
        source = Path(__file__).resolve().parents[1] / 'deep_gemm/mega/__init__.py'
        spec = importlib.util.spec_from_file_location('_mega_moe_core_test.mega', source)
        self.mega = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mega)
        self.group = SimpleNamespace(size=lambda: 8, rank=lambda: 0)
        self.base = SimpleNamespace(buffer=SimpleNamespace(nbytes=1 << 30),
                                    handle=SimpleNamespace(buffer_ptrs=list(range(8))), group=self.group)

    def create_buffer(self, **kwargs):
        return self.mega.SymmBuffer(self.group, 384, 4096, 6, 5120, 2304, base=self.base, **kwargs)

    def launch(self, buffer):
        self.mega.fp8_fp4_mega_moe(object(), (object(), object()), (object(), object()), buffer)
        return self.native.fp8_fp4_mega_moe.call_args.args[-1]

    def test_default_is_snapshotted_for_allocation_and_launch(self):
        buffer = self.create_buffer()
        self.assertEqual(self.native.get_symm_buffer_size_for_mega_moe.call_args.args[-1], 32)
        self.native.get_num_sms.return_value = 28
        self.assertEqual(self.launch(buffer), 32)
        other = self.create_buffer()
        self.assertEqual(self.launch(other), 28)
        self.assertEqual(self.launch(buffer), 32)

    def test_explicit_core_counts_are_independent_of_global_setting(self):
        for count in [28, 32, 36]:
            with self.subTest(num_sms=count):
                buffer = self.create_buffer(num_sms=count)
                self.assertEqual(self.native.get_symm_buffer_size_for_mega_moe.call_args.args[-1], count)
                self.assertEqual(self.launch(buffer), count)
        self.native.get_num_sms.assert_not_called()

    def test_invalid_core_counts_fail_before_allocation(self):
        for count in [0, -1, 24, 40]:
            with self.subTest(num_sms=count), self.assertRaises(ValueError):
                self.create_buffer(num_sms=count)
        self.native.get_symm_buffer_size_for_mega_moe.assert_not_called()

    def test_core_count_is_read_only(self):
        buffer = self.create_buffer(num_sms=28)
        with self.assertRaises(AttributeError):
            buffer.num_sms = 32


if __name__ == '__main__':
    unittest.main()
