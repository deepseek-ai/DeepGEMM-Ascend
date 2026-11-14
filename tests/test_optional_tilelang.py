"""MegaMoE installation must not require the unrelated TileLang HC backend."""

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class TestOptionalTileLang(unittest.TestCase):
    def test_package_import_without_tilelang(self):
        subprocess.run([sys.executable, '-c', '''
import sys
import torch_npu
sys.modules['deep_gemm.tilelang_ops'] = None
import deep_gemm
assert callable(deep_gemm.fp8_fp4_mega_moe)
assert callable(deep_gemm.tf32_hc_prenorm_gemm)
try:
    deep_gemm.tf32_hc_prenorm_gemm(None, None, None, None)
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('Explicit HC use must report the missing dependency')
'''], check=True)

    def test_hc_arguments_are_forwarded(self):
        import deep_gemm
        implementation = Mock()
        optional = SimpleNamespace(tf32_hc_prenorm_gemm=implementation)
        values = tuple(object() for _ in range(4))
        with patch.dict(sys.modules, {'deep_gemm.tilelang_ops': optional}):
            deep_gemm.tf32_hc_prenorm_gemm(*values, num_splits=1)
        implementation.assert_called_once_with(*values, 1)


if __name__ == '__main__':
    unittest.main()
