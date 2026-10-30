import torch
import torch.distributed as dist

from .. import _C

BLOCK_N = _C._mega_moe_block_n


class SymmBuffer:
    """HCCL-registered symmetric buffer for the fused MegaMoE pipeline.

    Shape and capacity attributes describe the C++ layout and must not be modified.
    Registered allocations stay in the pool until process exit; destroy() does not free HBM.
    Allocation order and buffer/handle owner lifetimes must match across ranks.
    Use base to share one allocation across sequential configurations.
    num_sms selects 28, 32 or 36 AI cores, defaulting to get_num_sms().
    All ranks must use the same core count. It is fixed for this buffer's lifetime;
    later set_num_sms() calls affect only newly created buffers.
    Data views borrow memory: keep a SymmBuffer owner alive until all accesses complete.
    Views become invalid when the last owner releases the allocation, even if retained.
    """

    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu',
                 base: 'SymmBuffer | None' = None,
                 *, num_sms: int | None = None):
        assert mma_type == 'fp8xfp4', 'Ascend MegaMoE currently supports only fp8xfp4'
        assert activation == 'swiglu', 'Ascend MegaMoE currently supports only swiglu'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation
        self._num_sms = _C.get_num_sms() if num_sms is None else num_sms
        if self._num_sms not in (28, 32, 36):
            raise ValueError('Ascend MegaMoE supports 28, 32 or 36 AI cores')

        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts, num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, mma_type, activation, num_shared_experts, self.num_sms)
        if base is None:
            rank_idx = group.rank()
            self.handle = _C.SymmBuffer(rank_idx, group.size(),
                                        group._get_backend(torch.device('npu')).get_hccl_comm_name(rank_idx), num_bytes)
            self.buffer = self.handle.buffer
            torch.npu.synchronize()
            dist.barrier(group)
        else:
            assert base.buffer is not None and base.handle is not None and base.group is group, 'Cannot reuse an invalid symmetric buffer'
            assert num_bytes <= base.buffer.nbytes, f'Requires {num_bytes} bytes, but the base buffer has {base.buffer.nbytes}'
            # Shared scratch must be used in order, with inputs refilled for each layout
            self.buffer = base.buffer
            self.handle = base.handle

        (self.x, self.x_sf, self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf, self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf, self.l2_acts, self.l2_acts_sf,
         self._expert_recv_count) = slice_input_buffers(self.buffer)

    def destroy(self) -> None:
        """Drop this owner's references; views require another live owner of the allocation."""
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = self.x_sf = None
        self.topk_idx = self.topk_weights = None
        self.shared_l1_acts = self.shared_l1_acts_sf = None
        self.shared_l2_acts = self.shared_l2_acts_sf = None
        self.l1_acts = self.l1_acts_sf = None
        self.l2_acts = self.l2_acts_sf = None
        self._expert_recv_count = None

    @property
    def num_sms(self) -> int:
        return self._num_sms

    @property
    def expert_recv_count(self) -> torch.Tensor:
        return self._expert_recv_count

    @property
    def num_bytes(self) -> int:
        return self.buffer.nbytes


def _transform_fp8_nz(t: torch.Tensor) -> torch.Tensor:
    # Shared FP8 weights use the NZ order consumed directly by AIC.
    shape = t.shape
    blocks = t.view(torch.uint8).reshape(*shape[:-2], shape[-2], shape[-1] // 32, 32)
    return blocks.transpose(-3, -2).contiguous().view(shape).view(t.dtype)


def _interleave_weights(t: torch.Tensor) -> torch.Tensor:
    # Gate/up pairs occupy one BLOCK_N block in both weight data and scale factors.
    assert t.dim() in (2, 3)
    dtype = t.dtype
    if dtype == torch.float8_e4m3fn:
        t = t.view(torch.uint8)
    *batch, n, k = t.shape
    half = n // 2
    gran = BLOCK_N // 2
    assert n % 2 == 0 and half % gran == 0
    num_n_blocks = half // gran
    gate = t[..., :half, :].reshape(*batch, num_n_blocks, gran, k)
    up = t[..., half:, :].reshape(*batch, num_n_blocks, gran, k)
    result = torch.stack((gate, up), dim=-3).reshape(*batch, num_n_blocks, BLOCK_N, k)
    return _transform_fp8_nz(result.view(dtype)) if dtype == torch.float8_e4m3fn else result


def _reshape_weights(t: torch.Tensor) -> torch.Tensor:
    assert t.dim() in (2, 3)
    *batch, n, k = t.shape
    assert n % BLOCK_N == 0
    result = t.reshape(*batch, n // BLOCK_N, BLOCK_N, k)
    return _transform_fp8_nz(result) if t.dtype == torch.float8_e4m3fn else result


def _scale_fp4_weight_sf(sf: torch.Tensor) -> torch.Tensor:
    # Internal dequant emits FP4 values divided by 64, so compensate once in the weight layout.
    exponent_bias = 6
    max_finite_exponent = 254
    exponent = sf.view(torch.uint8)
    valid = (exponent <= max_finite_exponent - exponent_bias) | (exponent == 255)
    if not valid.all().item():
        raise ValueError('FP4 weight scale exponent must be 0..248 or 255')
    # Exponent 0 represents 2^-127; only 255 is a special value
    exponent = torch.where(exponent == 255, exponent, exponent + exponent_bias)
    return exponent.contiguous().view(torch.int16)


def transform_weights_for_mega_moe(
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    activation: str = 'swiglu'
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Transform routed FP4 or shared FP8 (weight, scale) pairs."""
    assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
    assert isinstance(l1_weights, tuple) and isinstance(l2_weights, tuple), 'Ascend MegaMoE requires FP4/FP8 weight-scale pairs'
    assert len(l1_weights) == len(l2_weights) == 2
    l1_w, l1_sf = l1_weights
    l2_w, l2_sf = l2_weights
    assert l1_w.shape[:-2] == l1_sf.shape[:-2] and l1_w.size(-2) == l1_sf.size(-2)
    assert l2_w.shape[:-2] == l2_sf.shape[:-2] and l2_w.size(-2) == l2_sf.size(-2)
    assert l1_w.dtype == l2_w.dtype
    assert l1_w.dtype in (torch.int8, torch.float8_e4m3fn)
    assert l1_sf.dtype == l2_sf.dtype == torch.int16
    assert l1_w.size(-1) * (2 if l1_w.dtype == torch.int8 else 1) == l1_sf.size(-1) * 64
    assert l2_w.size(-1) * (2 if l2_w.dtype == torch.int8 else 1) == l2_sf.size(-1) * 64
    l1_w = _interleave_weights(l1_w)
    l1_sf = _interleave_weights(l1_sf).transpose(-1, -2).contiguous()
    l2_w = _reshape_weights(l2_w)
    l2_sf = _reshape_weights(l2_sf).transpose(-1, -2).contiguous()
    if l1_w.dtype == torch.int8:
        l1_sf = _scale_fp4_weight_sf(l1_sf)
        l2_sf = _scale_fp4_weight_sf(l2_sf)
    l1_transformed = (l1_w, l1_sf)
    l2_transformed = (l2_w, l2_sf)
    return l1_transformed, l2_transformed


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: tuple[torch.Tensor, torch.Tensor],
                     l2_weights: tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
                     shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
                     cumulative_local_expert_recv_stats: torch.Tensor | None = None,
                     recipe: tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: float | None = None,
                     fast_math: bool = True) -> None:
    assert sym_buffer.buffer is not None and sym_buffer.handle is not None
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        sym_buffer.hidden, sym_buffer.intermediate_hidden, sym_buffer.num_shared_experts,
        recipe,
        activation, activation_clamp,
        fast_math, sym_buffer.num_sms
    )
