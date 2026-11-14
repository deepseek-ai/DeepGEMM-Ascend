__version__ = '0.1.1'

import os
import torch

# Set default environment variables captured when the package was built.
try:
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Ascend toolkit path discovery
ASCEND_HOME = os.environ.get('ASCEND_HOME_PATH',
    os.environ.get('ASCEND_TOOLKIT_HOME', '/usr/local/Ascend/ascend-toolkit/latest'))


def get_library_root_path():
    """Get the root path of the deep_gemm package (where include/ lives)."""
    return os.path.dirname(os.path.realpath(__file__))


from . import _C
from ._C import (
    set_num_sms,
    get_num_sms,
    set_npu_arch,
    get_npu_arch,
    set_mk_alignment_for_contiguous_layout,
    get_mk_alignment_for_contiguous_layout,
    get_theoretical_mk_alignment_for_contiguous_layout,
    use_deterministic_algorithms,
    get_deterministic_algorithms,
)

# Upstream cuBLASLt API aliases backed by ACLNN
from ._C import (
    cublaslt_gemm_nt, cublaslt_gemm_nn,
    cublaslt_gemm_tn, cublaslt_gemm_tt,
)

# DeepGEMM Kernels
from ._C import (
    # FP8 FP4 GEMMs
    fp8_fp4_gemm_nt, fp8_fp4_gemm_nn,
    fp8_fp4_gemm_tn, fp8_fp4_gemm_tt,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_fp4_gemm_nn_contiguous,
    # FP8 GEMMs
    fp8_gemm_nt, fp8_gemm_nn,
    fp8_gemm_tn, fp8_gemm_tt,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nn_contiguous,
    k_grouped_fp8_gemm_nt_contiguous,
    k_grouped_fp8_gemm_tn_contiguous,
    # ACLNN reference GEMMs
    aclnn_fp8_fp4_gemm_nt, aclnn_fp8_fp4_gemm_nn,
    aclnn_fp8_fp4_gemm_tn, aclnn_fp8_fp4_gemm_tt,
    aclnn_bf16_gemm_nt, aclnn_bf16_gemm_nn,
    aclnn_bf16_gemm_tn, aclnn_bf16_gemm_tt,
    # BF16 GEMMs
    bf16_gemm_nt, bf16_gemm_nn,
    bf16_gemm_tn, bf16_gemm_tt,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nn_contiguous,
    k_grouped_bf16_gemm_tn_contiguous,
    # Einsum kernels
    einsum,
    fp8_einsum,
    # Attention kernels
    fp8_fp4_mqa_logits,
    get_paged_mqa_logits_metadata,
    fp8_fp4_paged_mqa_logits,
    get_sparse_mqa_logits_metadata,
    fp8_fp4_sparse_mqa_logits,
    # Layout kernels
    transform_sf_into_required_layout,
    transform_k_grouped_sf_into_required_layout,
)

# Public sparse APIs live in deep_gemm: the non-paged functions are re-exported above.
# Paged wrappers below own configuration and FIRST/REUSE state; their underscored _C
# bindings are internal and bypass this state management.
# The tensor owns mapping, KV split plan, physical blocks and compact KV; prepare scratch is temporary.
def get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv,
                                         sparse_kv_block_indices, qk_dtype, sparse_block_kv):
    """Prepare FIRST/REUSE state for MQA pages stored as [page DATA | page SF].

    Use the public deep_gemm entry points for both prepare and logits so that
    configuration and FIRST/REUSE state are managed together.

    For each query, let B = sparse_block_kv, C = context_lens[q] >= 0 and
    capacity = sparse_kv_block_indices.size(1). The first min(capacity, ceil(C / B))
    logical block IDs must be unique, strictly increasing and satisfy 0 <= i * B < C.
    The remaining entries are unused padding and may repeat.

    Queries with the same request ID in indices must be consecutive. Their block-table
    rows must agree on referenced logical pages. Every referenced logical page must
    fit within its block-table row, and its physical page ID must index the KV cache
    passed to the logits call.

    The caller must guarantee these data constraints; no device-value validation is
    performed. Violations can produce incorrect results or out-of-bounds accesses.
    """
    metadata = _C._get_paged_sparse_mqa_logits_metadata(
        context_lens, block_table, indices, page_kv, sparse_kv_block_indices, qk_dtype, sparse_block_kv)
    config = (sparse_kv_block_indices.shape[0], sparse_kv_block_indices.shape[1],
              sparse_block_kv, page_kv, get_num_sms(), qk_dtype)
    metadata._paged_sparse_mqa_state = (config, True)
    return metadata


def fp8_fp4_paged_sparse_mqa_logits(q, kv_cache, weights, metadata,
                                    num_max_sparse_blocks, sparse_block_kv):
    """Compute logits using the selected KV snapshot stored in metadata.

    The first call caches KV; later calls reuse that snapshot with the current Q and
    weights. Regenerate metadata after changing KV contents, even if indices are unchanged.
    """
    state = getattr(metadata, '_paged_sparse_mqa_state', None)
    if state is None:
        raise TypeError('metadata must come from get_paged_sparse_mqa_logits_metadata')
    config, is_first = state
    expected = (q[0].shape[0], num_max_sparse_blocks, sparse_block_kv,
                kv_cache.shape[1], get_num_sms(), q[0].dtype)
    if config != expected:
        raise RuntimeError('metadata does not match the current paged sparse MQA configuration')
    dry_run = _C.get_dry_run()
    logits = _C._fp8_fp4_paged_sparse_mqa_logits(
        q, kv_cache, weights, metadata, is_first, num_max_sparse_blocks, sparse_block_kv)
    if is_first and not dry_run:
        metadata._paged_sparse_mqa_state = (config, False)
    return logits

from .mega import (
    SymmBuffer,
    transform_weights_for_mega_moe,
    fp8_fp4_mega_moe,
)

# Epilogue classes
from ._C import epilogue

# Locality domain helpers
from . import locality_domain
from .locality_domain import (
    get_num_locality_domains,
    is_localization_available,
    localize,
    is_localized,
    destroy_localizer,
)

def tf32_hc_prenorm_gemm(a: torch.Tensor,
                       b: torch.Tensor,
                       d: torch.Tensor,
                       sqr_sum: torch.Tensor,
                       num_splits: int | None = None) -> None:
    """Load the optional TileLang HC kernel only when it is used."""
    from .tilelang_ops import tf32_hc_prenorm_gemm as implementation
    return implementation(a, b, d, sqr_sum, num_splits)

_C.init(__version__, get_library_root_path())
