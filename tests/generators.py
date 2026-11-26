import torch
import enum
import random
from dataclasses import dataclass
from typing import Generator
from itertools import product


def reset_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)


class Major(enum.Enum):
    K = 0
    MN = 1


class GemmType(enum.Enum):
    Normal = 0
    MGroupedContiguousWithPsumLayout = 1
    KGroupedContiguousWithPsumLayout = 2
    Batched = 3


def ceil_to_ue8m0(x: torch.Tensor):
    bits = x.abs().float().view(torch.int)
    # use floor div to emulate right shift, cause torch-npu doesn't support right shift
    exp = ((bits // 2**23) % 256) + (bits % 2**23).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _quantize_to_fp4_with_major(x: torch.Tensor, major_dim: int) -> torch.Tensor:
    ax = x.abs()
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    # midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += (ax > boundary).to(torch.uint8)
    sign = (x < 0) & (code != 0)
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=x.device)
    f32_repr = values[code.long()]
    f32_repr = torch.where(sign, -f32_repr, f32_repr)
    fp4_repr = code = code | (sign.to(torch.uint8) << 3)
    fp4_repr = fp4_repr.transpose(major_dim, -1).contiguous()
    fp4_repr = (fp4_repr[..., ::2] & 0x0f) | ((fp4_repr[..., 1::2] & 0x0f) << 4)
    fp4_repr = fp4_repr.view(torch.int8)
    fp4_repr = fp4_repr.transpose(major_dim, -1)
    return fp4_repr, f32_repr


def _quantize_to_fp8_with_major(x: torch.Tensor, major_dim: int) -> torch.Tensor:
    fp8_repr = x.to(torch.float8_e4m3fn)
    fp8_repr = fp8_repr.transpose(major_dim, -1).contiguous()
    fp8_repr = fp8_repr.transpose(major_dim, -1)
    f32_repr = fp8_repr.float()
    return fp8_repr, f32_repr


def quantize_with_major(tensor_in: torch.Tensor, gran: list[int], quant_dtype: torch.dtype, major_dim: int = None) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    if quant_dtype == torch.bfloat16:
        tensor_bf16 = tensor_in.to(torch.bfloat16)
        tensor_bf16 = tensor_bf16.transpose(major_dim, -1).contiguous().transpose(major_dim, -1)
        return tensor_bf16, tensor_bf16

    shape = list(tensor_in.size())
    assert len(gran) == len(shape), f'gran ndim mismatch: shape={shape}, gran={gran}'

    padded_shape = [aligned(s, g) for s, g in zip(shape, gran)]
    tensor = torch.zeros(padded_shape, dtype=torch.float32, device='npu')
    tensor[tuple(slice(0, s) for s in shape)].copy_(tensor_in)

    # convert tensor => [shape_0 / gran_0, shape_1 / gran_1, ..., shape_n / gran_n, gran_0, gran_1, ..., gran_n]
    view_shape = sum([[s // g, g] for s, g in zip(padded_shape, gran)], [])
    perm_order = list(range(0, len(shape) * 2, 2)) + list(range(1, len(shape) * 2, 2))
    sf_shape = [s // g for s, g in zip(padded_shape, gran)]
    sf_view = tensor.reshape(*view_shape).permute(perm_order).reshape(*sf_shape, -1)

    sf_div = 448.0 if quant_dtype == torch.float8_e4m3fn else 6.0
    sf_float = ceil_to_ue8m0(sf_view.abs().amax(dim=-1).clamp(1e-4) / sf_div)
    sf = sf_float.contiguous()

    sf_full = sf_float
    for dim, g in enumerate(gran):
        sf_full = sf_full.repeat_interleave(g, dim=dim)
    sf_full = sf_full[tuple(slice(0, s) for s in shape)]

    if quant_dtype == torch.float8_e4m3fn:
        x_quant, f32_repr = _quantize_to_fp8_with_major(tensor[tuple(slice(0, s) for s in shape)] / sf_full, major_dim=major_dim)
    elif quant_dtype == torch.float4_e2m1fn_x2:
        x_quant, f32_repr = _quantize_to_fp4_with_major(tensor[tuple(slice(0, s) for s in shape)] / sf_full, major_dim=major_dim)
    else:
        raise ValueError(f'Unsupported quant dtype: {quant_dtype}')
    return (x_quant, sf), (f32_repr * sf_full).contiguous()


_FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
_FP4_BYTE_LUT = tuple((_FP4_VALUES[byte & 0x0f], _FP4_VALUES[byte >> 4]) for byte in range(256))


def _scale_quantized_reference(
        values: torch.Tensor,
        sf: torch.Tensor,
        shape: tuple[int, ...],
        gran: tuple[int, ...],
) -> torch.Tensor:
    padded_shape = tuple(aligned(size, block) for size, block in zip(shape, gran))
    slices = tuple(slice(0, size) for size in shape)
    if padded_shape != shape:
        padded = torch.zeros(padded_shape, dtype=torch.float32, device='npu')
        padded[slices].copy_(values)
        values = padded
    values = values.contiguous()
    value_view = sum(([size // block, block] for size, block in zip(padded_shape, gran)), [])
    sf_view = sum(([size // block, 1] for size, block in zip(padded_shape, gran)), [])
    values.reshape(value_view).mul_(sf.reshape(sf_view))
    return values[slices].contiguous()


def generate_random_quantized_with_major(
        shape: tuple[int, ...],
        gran: tuple[int, ...],
        quant_dtype: torch.dtype,
        major_dim: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Generate full-tensor random quantized codes, scales, and the exact decoded reference."""
    shape = tuple(shape)
    gran = tuple(gran)
    major_dim %= len(shape)
    assert len(shape) == len(gran)
    assert quant_dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2)

    sf_shape = tuple(aligned(size, block) // block for size, block in zip(shape, gran))
    sf_values = (
        (2.0**-9, 2.0**-8, 2.0**-7)
        if quant_dtype == torch.float8_e4m3fn
        else (0.25, 0.5, 1.0)
    )
    sf_table = torch.tensor(sf_values, dtype=torch.float32, device='npu')
    sf = sf_table[torch.randint(0, len(sf_values), sf_shape, device='npu')].contiguous()

    storage_shape = list(shape)
    storage_shape[major_dim], storage_shape[-1] = storage_shape[-1], storage_shape[major_dim]
    if quant_dtype == torch.float8_e4m3fn:
        # Cover all non-NaN E4M3 codes independently at every tensor element.
        raw = torch.randint(0, 254, storage_shape, dtype=torch.uint8, device='npu')
        raw += (raw >= 0x7f).to(torch.uint8)
        data = raw.view(quant_dtype).transpose(major_dim, -1)
        f32_repr = data.float()
    else:
        assert shape[major_dim] % 2 == 0, \
            f'FP4 packed dimension must be even, got shape={shape}, major_dim={major_dim}'
        storage_shape[-1] //= 2
        # A uniform random byte gives two independent uniform FP4 nibbles.
        packed = torch.randint(-128, 128, storage_shape, dtype=torch.int8, device='npu')
        data = packed.transpose(major_dim, -1)
        byte_lut = torch.tensor(_FP4_BYTE_LUT, dtype=torch.float32, device='npu')
        decoded = byte_lut[packed.view(torch.uint8).long()]
        decoded = decoded.reshape(*storage_shape[:-1], storage_shape[-1] * 2)
        f32_repr = decoded.transpose(major_dim, -1)

    return (data, sf), _scale_quantized_reference(f32_repr, sf, shape, gran)


def generate_operand_with_major(
        shape: tuple[int, ...],
        gran: tuple[int, ...],
        dtype: torch.dtype,
        major_dim: int,
        device: str = 'npu',
) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    if device == 'meta':
        storage_shape = list(shape)
        storage_shape[major_dim], storage_shape[-1] = storage_shape[-1], storage_shape[major_dim]
        if dtype == torch.float4_e2m1fn_x2:
            storage_shape[-1] //= 2
        data_dtype = torch.int8 if dtype == torch.float4_e2m1fn_x2 else dtype
        data = torch.empty(storage_shape, dtype=data_dtype, device=device).transpose(major_dim, -1)
        if dtype == torch.bfloat16:
            return data, None
        sf_shape = tuple(aligned(size, block) // block for size, block in zip(shape, gran))
        return (data, torch.empty(sf_shape, dtype=torch.float32, device=device)), None
    assert device == 'npu'
    if dtype == torch.bfloat16:
        tensor = torch.randn(shape, dtype=torch.float32, device='npu')
        return quantize_with_major(tensor, gran, dtype, major_dim)
    return generate_random_quantized_with_major(shape, gran, dtype, major_dim)


def zero_operand_slices_(
        operand: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        reference: torch.Tensor,
        slices,
) -> None:
    """Zero logical slices in both an operand's data storage and its reference."""
    data = operand[0] if isinstance(operand, tuple) else operand
    # torch_npu has no fill kernel for FP8, but zero has the all-zero bit pattern.
    writable_data = data.view(torch.uint8) if data.dtype == torch.float8_e4m3fn else data
    for span in slices:
        writable_data[span, :] = 0
        reference[span, :] = 0


def aligned(size: int, alignment: int):
    return (size + alignment - 1) // alignment * alignment


def generate_psum_layout(expected_size: int, num_groups: int, alignment: int, device: str = 'npu', sizes: list[int] | None = None):
    assert sizes is None or len(sizes) == num_groups
    psum_layout = []
    psum = 0
    for group_idx in range(num_groups):
        size = sizes[group_idx] if sizes is not None else max(int(random.uniform(0.7, 1.3) * expected_size), 0)
        psum_layout.append(psum + size)
        psum = aligned(psum + size, alignment)
    return psum, psum_layout, torch.tensor(psum_layout, dtype=torch.int32, device=device)


def iter_psum_slice(layout: list[int], alignment: int):
    last = 0
    for ptr in layout:
        yield slice(last, ptr)
        last = aligned(ptr, alignment)


def iter_psum_sep(layout: list[int], alignment: int, gran: int = 1):
    for ptr in layout:
        s0 = (ptr + gran - 1) // gran
        s1 = aligned(ptr, alignment) // gran
        if s0 < s1:
            yield slice(s0, s1)


def make_strided(tensor: torch.Tensor, stride_dims: list[int]):
    shape = list(tensor.size())
    stride = list(tensor.stride())
    for dim in stride_dims:
        shape[dim]  += random.randint(1, 256)
        stride[dim] += random.randint(1, 256)
    strided = torch.empty(shape, dtype=tensor.dtype, device=tensor.device).as_strided(shape, stride)
    strided.copy_(tensor)
    return strided


@dataclass
class GemmTestDesc:
    gemm_type: GemmType
    a_dtype: torch.dtype
    b_dtype: torch.dtype
    cd_dtype: torch.dtype
    shape_mnk: tuple[int, int, int]
    a_major: Major
    b_major: Major
    accumulate: bool = False
    num_groups: int = None
    alignment: int = 256
    strided: list[int] | None = None
    ensure_zero_padding: bool = True
    recipe: tuple[int, int, int] = (1, 1, 32)
    alpha: float | None = None
    tag: str | None = None
    with_bias: bool = False
    ks_cpu: list[int] | None = None
    host_ks_cpu: list[int] | None = None
    profile: bool = True
    same_ab: bool = False

    @property
    def m(self): return self.shape_mnk[0]
    @property
    def n(self): return self.shape_mnk[1]
    @property
    def k(self): return self.shape_mnk[2]

    @property
    def a_recipe(self):
        return self.recipe[0], self.recipe[2]

    @property
    def b_recipe(self):
        return self.recipe[1], self.recipe[2]

    @property
    def has_sf(self):
        return self.a_dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2) or self.b_dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2)

    def get_available_trans_ab(self) -> list[str]:
        if self.gemm_type == GemmType.Normal:
            return ['nt', 'tn', 'nn', 'tt']
        elif self.gemm_type == GemmType.MGroupedContiguousWithPsumLayout:
            return ['nt']
        elif self.gemm_type == GemmType.KGroupedContiguousWithPsumLayout:
            return ['tn']
        elif self.gemm_type == GemmType.Batched:
            return ['nt', 'tn', 'nn', 'tt']

    def _generate_normal(self, device: str):
        m, n, k = self.shape_mnk

        c = None
        if self.accumulate or self.with_bias:
            if device == 'meta':
                c = torch.empty(1 if self.with_bias else m, n, dtype=self.cd_dtype, device=device)
            else:
                c = torch.randn(1 if self.with_bias else m, n, dtype=self.cd_dtype, device='npu')

        a, a_repr = generate_operand_with_major(
            (m, k), self.a_recipe, self.a_dtype,
            major_dim=0 if self.a_major == Major.MN else 1, device=device)
        b, b_repr = (a, a_repr) if self.same_ab else generate_operand_with_major(
            (n, k), self.b_recipe, self.b_dtype,
            major_dim=0 if self.b_major == Major.MN else 1, device=device)

        if device == 'meta':
            d = torch.empty((m, n), dtype=self.cd_dtype, device=device)
        else:
            alpha_value = self.alpha if self.alpha is not None else 1.0
            d = (alpha_value * (a_repr.float() @ b_repr.float().T)
                 + (c if c is not None else 0)).to(self.cd_dtype)

        return (m, n, k), a, b, c, d, (None, None)

    def _generate_m_grouped_contiguous(self, device: str):
        expected_m, n, k = self.shape_mnk
        total_m, g_layout_cpu, g_layout = generate_psum_layout(expected_m, self.num_groups, self.alignment, device)

        assert not self.accumulate, "grouped matmul doesn't support accumulation yet"

        assert self.a_major == Major.K, "a must be K-major for m-grouped layout"
        a, a_repr = generate_operand_with_major(
            (total_m, k), self.a_recipe, self.a_dtype, major_dim=1, device=device)
        b, b_repr = generate_operand_with_major(
            (self.num_groups, n, k), (1, *self.b_recipe), self.b_dtype,
            major_dim=1 if self.b_major == Major.MN else 2, device=device)
        if self.ensure_zero_padding and device != 'meta':
            zero_operand_slices_(
                a, a_repr, iter_psum_sep(g_layout_cpu, self.alignment))

        # fill random garbage into the psum space of the scaling factors
        if self.has_sf and device != 'meta':
            sfa = a[1]
            for ms in iter_psum_sep(g_layout_cpu, self.alignment):
                sfa[ms, :] = torch.randn_like(sfa[ms, :])

        if device == 'meta':
            d = torch.empty(total_m, n, dtype=self.cd_dtype, device=device)
        else:
            d = torch.zeros(total_m, n, dtype=self.cd_dtype, device='npu')
            for i, ms in enumerate(iter_psum_slice(g_layout_cpu, self.alignment)):
                d[ms, :] = a_repr[ms, :] @ b_repr[i].T

        # when ensure zero_padding, we should compute all region including the padded region
        # otherwise we only compare the valid region
        cmp_spans = None if self.ensure_zero_padding else list(iter_psum_slice(g_layout_cpu, self.alignment))

        return (total_m, n, k), a, b, None, d, (g_layout, cmp_spans)

    def _generate_k_grouped_contiguous(self, device: str):
        m, n, expected_k = self.shape_mnk
        total_k, g_layout_cpu, g_layout = generate_psum_layout(expected_k, self.num_groups, self.alignment, device, self.ks_cpu)

        c = None
        if self.accumulate:
            if device == 'meta':
                c = torch.empty(self.num_groups, m, n, dtype=self.cd_dtype, device=device)
            else:
                c = torch.randn(self.num_groups, m, n, dtype=self.cd_dtype, device='npu')

        assert self.a_major == Major.MN, "a must be MN-major for k-grouped layout"
        assert self.b_major == Major.MN, "b must be MN-major for k-grouped layout"
        a, a_repr = generate_operand_with_major(
            (total_k, m), self.a_recipe[::-1], self.a_dtype, major_dim=1, device=device)
        b, b_repr = generate_operand_with_major(
            (total_k, n), self.b_recipe[::-1], self.b_dtype, major_dim=1, device=device)

        # k-group is always zero-padded between groups
        if device != 'meta':
            zero_operand_slices_(a, a_repr, iter_psum_sep(g_layout_cpu, self.alignment))
            zero_operand_slices_(b, b_repr, iter_psum_sep(g_layout_cpu, self.alignment))

        # fill random garbage into the psum space of the scaling factors
        if self.has_sf and device != 'meta':
            sfa, sfb = a[1], b[1]
            for ks in iter_psum_sep(g_layout_cpu, self.alignment, self.recipe[2]):
                sfa[ks, :] = torch.randn_like(sfa[ks, :])
                sfb[ks, :] = torch.randn_like(sfb[ks, :])

        if device == 'meta':
            d = torch.empty(self.num_groups, m, n, dtype=self.cd_dtype, device=device)
        else:
            d = torch.zeros(self.num_groups, m, n, dtype=self.cd_dtype, device='npu')
            for i, ks in enumerate(iter_psum_slice(g_layout_cpu, self.alignment)):
                product = (a_repr[ks, :].float().T @ b_repr[ks, :].float()).to(self.cd_dtype)
                d[i, :, :] = (c[i, :, :] if self.accumulate else 0) + product

        return (m, n, total_k), a, b, c, d, (g_layout, None)

    def _generate_batched(self, device: str):

        m, n, k = self.shape_mnk

        c = None
        if self.accumulate:
            if device == 'meta':
                c = torch.empty(self.num_groups, m, n, dtype=self.cd_dtype, device=device)
            else:
                c = torch.randn(self.num_groups, m, n, dtype=self.cd_dtype, device='npu')

        a, a_repr = generate_operand_with_major(
            (self.num_groups, m, k), (1, *self.a_recipe), self.a_dtype,
            major_dim=1 if self.a_major == Major.MN else 2, device=device)
        b, b_repr = (a, a_repr) if self.same_ab else generate_operand_with_major(
            (self.num_groups, n, k), (1, *self.b_recipe), self.b_dtype,
            major_dim=1 if self.b_major == Major.MN else 2, device=device)

        d = torch.empty(self.num_groups, m, n, dtype=self.cd_dtype, device=device)
        if device != 'meta':
            if self.same_ab:
                a_repr = b_repr = a_repr.float()
            for i in range(self.num_groups):
                if self.accumulate:
                    d[i] = a_repr[i] @ b_repr[i].T + c[i]
                else:
                    d[i] = a_repr[i] @ b_repr[i].T

        return (m, n, k), a, b, c, d, (None, None)

    def generate(self, device: str = 'npu'):
        reset_seed()
        if self.gemm_type == GemmType.Normal:
            return self._generate_normal(device)
        elif self.gemm_type == GemmType.MGroupedContiguousWithPsumLayout:
            return self._generate_m_grouped_contiguous(device)
        elif self.gemm_type == GemmType.KGroupedContiguousWithPsumLayout:
            return self._generate_k_grouped_contiguous(device)
        elif self.gemm_type == GemmType.Batched:
            return self._generate_batched(device)


def _enumerate_recipe(a_dtype: torch.dtype, b_dtype: torch.dtype):
    assert a_dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2)
    assert b_dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2)
    shapes = [
        (4096, 4096, 4096),
        (18, 18, 18),
    ]
    recipes = [
        (1, 32, 32),
        (32, 1, 32),
        (32, 32, 32),
    ]
    for shape, recipe in product(shapes, recipes):
        yield GemmTestDesc(
            GemmType.Normal,
            a_dtype, b_dtype, torch.float,
            shape,
            Major.K, Major.K,
            accumulate=True,
            recipe=recipe,
            tag=f'recipe=({recipe[0]:2}, {recipe[1]:2}, {recipe[2]:2})'
        )



def _enumerate_normal_strided(a_dtype: torch.dtype, b_dtype: torch.dtype):
    shapes = [
        (    4,     4,     4),
        (    4,     4, 16384),
        (    4, 16384,     4),
        (16384,     4,     4),
        (16384,     4, 16384),
        (    4, 16384, 16384),
        (  128,   128,   128),
        (  256,   512,  1024),
        ( 1024,   256,   512),
        ( 4096,  4096,  4096),
    ]
    for shape, a_major, b_major in product(shapes, [Major.K, Major.MN], [Major.K, Major.MN]):
        yield GemmTestDesc(
            GemmType.Normal,
            a_dtype, b_dtype, torch.float,
            shape,
            a_major, b_major,
            accumulate=True,
            strided=True,
            tag='strided'
        )


def _enumerate_normal_unaligned_mn(a_dtype: torch.dtype, b_dtype: torch.dtype):
    shapes = [
        (   4,    4,   64),
        (  17, 4096, 4096),
        (4096,   17, 4096),
        (  15, 4096, 4096),
        (4096,   15, 4096),
        ( 130,  257, 2048),
        ( 513,  257, 4096),
    ]
    for shape, a_major, b_major in product(shapes, [Major.K, Major.MN], [Major.K, Major.MN]):
        m, n, k = shape
        # fp4 packs two int4 values into one byte along the MN(major) dim, so that
        # dim must be even. Double the odd MN-major dim to keep it packable.
        if a_dtype == torch.float4_e2m1fn_x2 and a_major == Major.MN and m % 2 != 0:
            m *= 2
        if b_dtype == torch.float4_e2m1fn_x2 and b_major == Major.MN and n % 2 != 0:
            n *= 2
        yield GemmTestDesc(
            GemmType.Normal,
            a_dtype, b_dtype, torch.float,
            (m, n, k),
            a_major, b_major,
            accumulate=True,
            tag='unaligned-mn'
        )


def _enumerate_normal_unaligned_k(a_dtype: torch.dtype, b_dtype: torch.dtype):
    shapes = [
        (  64,   64,    4),
        (1024, 2048,   17),
        (8192, 4096,   15),
        ( 128,  128,  234),
        (2048, 8192, 4095),
    ]
    for shape, a_major, b_major in product(shapes, [Major.K, Major.MN], [Major.K, Major.MN]):
        m, n, k = shape
        # FP4 packs two values per byte; double odd K to make it packable while keeping a K tail.
        if (a_dtype == torch.float4_e2m1fn_x2 or b_dtype == torch.float4_e2m1fn_x2) and k % 2 != 0:
            k *= 2
        yield GemmTestDesc(
            GemmType.Normal,
            a_dtype, b_dtype, torch.float,
            (m, n, k),
            a_major, b_major,
            accumulate=True,
            tag='unaligned-k'
        )


def _enumerate_normal(
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
):
    def create_testcase(m, n, k, a_major, b_major, acc, cd_dtype, tag=None):
        return GemmTestDesc(
            GemmType.Normal,
            a_dtype, b_dtype, cd_dtype,
            (m, n, k),
            a_major, b_major,
            accumulate=acc,
            recipe=(1, 1, 32),
            tag=tag
        )

    fp32_output_nk = [(256, 7168), (129280, 7168)]
    bf16_output_nk = [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]
    m_fwd_list, m_bwd_list = [1, 128, 4096], [4096, ]
    nk_list = list(bf16_output_nk)
    if a_dtype == torch.bfloat16:
        nk_list += fp32_output_nk

    for m in m_fwd_list:
        for i in range(len(nk_list)):
            n, k = nk_list[i]
            out_dtype = torch.bfloat16 if i < len(bf16_output_nk) else torch.float
            yield create_testcase(m, n, k, Major.K, Major.K, False, out_dtype, tag='  fwd')

    for m in m_bwd_list:
        for n, k in nk_list:
            yield create_testcase(m, k, n, Major.K, Major.MN, False, torch.bfloat16, tag='dgrad')
            yield create_testcase(n, m, k, Major.MN, Major.MN, True, torch.float, tag='wgrad')
            yield create_testcase(n, m, k, Major.MN, Major.MN, False, torch.bfloat16, tag='wgrad')


def _enumerate_m_grouped_contiguous(
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
):
    def create_test(num_groups, expected_m, n, k, major_b):
        return GemmTestDesc(
            GemmType.MGroupedContiguousWithPsumLayout,
            a_dtype, b_dtype, torch.bfloat16,
            (expected_m, n, k),
            Major.K, major_b,
            accumulate=False,
            num_groups=num_groups,
            recipe=(1, 1, 32),
            tag=f'm-grouped, expected-m={expected_m:>4}'
        )

    yield create_test(128, 1, 4, 4, Major.K)  # minimal case

    m_group_list = [(4, 1), (4, 8192), (8, 4096), (16, 2048)]
    n_k_list = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]

    for num_groups, expected_m in m_group_list:
        for n, k in n_k_list:
            for major_b in [Major.K, Major.MN]:
                yield create_test(num_groups, expected_m, n, k, major_b)


def enumerate_k_grouped_contiguous_test_variants(real_ks_cpu: list[int]):
    yield list(real_ks_cpu)

    empty_ks_cpu = [0 if random.random() < 0.25 else k for k in real_ks_cpu]
    empty_ks_cpu[0], empty_ks_cpu[-1] = 0, real_ks_cpu[-1]
    yield empty_ks_cpu

    yield [k if group_idx == len(real_ks_cpu) // 2 else 0 for group_idx, k in enumerate(real_ks_cpu)]


def _enumerate_k_grouped_contiguous(
        a_dtype: torch.dtype,
        b_dtype: torch.dtype
):
    k_grouped_list = [
        ( 32, 4, 4, 1),
        (128, 4, 4, 1), ( 4, 4096, 7168, 1),
        ( 4, 4096, 7168, 8192), ( 4, 7168, 2048, 8192),   # EP64
        ( 8, 4096, 7168, 4096), ( 8, 7168, 2048, 4096),   # EP32
        (16, 4096, 7168, 2048), (16, 7168, 2048, 2048),   # EP16
    ]

    # CUDA's output modes plus the existing Ascend BF16 accumulation coverage.
    cd_options = [(True, torch.float32), (False, torch.float32), (False, torch.bfloat16), (True, torch.bfloat16)]
    alignment = 256
    for num_groups, m, n, expected_k in k_grouped_list:
        real_ks_cpu = [max(1, int(expected_k * random.uniform(0.7, 1.3))) for _ in range(num_groups)]
        for variant, ks_cpu in zip(('normal', 'empty', 'single'), enumerate_k_grouped_contiguous_test_variants(real_ks_cpu)):
            host_ks_options = (('list', [aligned(k, alignment) for k in ks_cpu]), ('none', None), ('empty', []))
            for (accumulate, cd_dtype), (hint, host_ks_cpu) in product(cd_options, host_ks_options):
                yield GemmTestDesc(
                    GemmType.KGroupedContiguousWithPsumLayout,
                    a_dtype, b_dtype, cd_dtype,
                    (m, n, expected_k),
                    Major.MN, Major.MN,
                    accumulate=accumulate,
                    num_groups=num_groups,
                    alignment=alignment,
                    recipe=(1, 1, 32),
                    tag=f'k-grouped, expected-k={expected_k:>4}, variant={variant}, ks-cpu={hint}',
                    ks_cpu=ks_cpu,
                    host_ks_cpu=host_ks_cpu,
                    profile=variant == 'normal' and bool(host_ks_cpu),
                )


def enumerate_gemm_tests(
        gemm_type: GemmType,
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
        unaligned_mn: bool = False,
        unaligned_k: bool = False,
        strided: bool = False,
        test_recipes: bool = False
    ):
    reset_seed()
    if gemm_type == GemmType.Normal:
        if unaligned_mn:
            gen = _enumerate_normal_unaligned_mn(a_dtype, b_dtype)
        elif unaligned_k:
            gen = _enumerate_normal_unaligned_k(a_dtype, b_dtype)
        elif strided:
            gen = _enumerate_normal_strided(a_dtype, b_dtype)
        elif test_recipes:
            gen = _enumerate_recipe(a_dtype, b_dtype)
        else:
            gen = _enumerate_normal(a_dtype, b_dtype)
    elif gemm_type == GemmType.MGroupedContiguousWithPsumLayout:
        gen = _enumerate_m_grouped_contiguous(a_dtype, b_dtype)
    elif gemm_type == GemmType.KGroupedContiguousWithPsumLayout:
        gen = _enumerate_k_grouped_contiguous(a_dtype, b_dtype)
    else:
        raise ValueError(f'Unsupported gemm type: {gemm_type}')
    return gen


def enumerate_syrk_tests(batched: bool = True, strided: bool = False):
    shapes = [
        (16, 4608, 5120), (16, 1536, 5120), (16, 576, 5120),
        (8, 1536, 24576), (8, 512, 32768), (4, 5120, 16384),
        (8, 4096, 7168), (8, 2048, 7168),
    ]
    boundaries = [(1, 1, 1), (2, 63, 31), (1, 64, 64), (2, 65, 33),
                  (0, 32, 64), (2, 0, 64), (2, 32, 0)]
    for major, cases in ((Major.K, shapes + boundaries), (Major.MN, [(2, 4, 8), (2, 8, 8)])):
        for num_batches, m, k in cases:
            if num_batches == 0 and (not batched or strided):
                continue
            yield GemmTestDesc(
                GemmType.Batched if batched else GemmType.Normal,
                torch.bfloat16, torch.bfloat16, torch.bfloat16,
                (m, m, k), major, major,
                num_groups=num_batches if batched else None,
                strided=strided, same_ab=True,
                profile=batched and not strided and (num_batches, m, k) in shapes,
            )


def enumerate_sf_layout():
    grans = (1, 32, 128)
    alignment = 256
    for dtype, major, mn, k, num_batches, gran_mn in product(
        (torch.float32, torch.int16), (Major.K, Major.MN),
        (4096, 4097, 8192), (128, 7168, 7296), (1, 2, 4), grans,
    ):
        gemm_type = GemmType.Normal if num_batches == 1 else GemmType.Batched
        yield gemm_type, dtype, major, mn, k, gran_mn, num_batches, alignment, ()


def enumerate_k_grouped_sf_layout():
    grans = (1, 32, 128)
    alignment = 256
    group_configs = ((16, 2048), (8, 4096), (72, 384), (128, 256))
    reset_seed()
    for dtype, major, mn, group_config, gran_mn in product(
        (torch.float32, torch.int16), (Major.K, Major.MN),
        (4096, 7168), group_configs, grans,
    ):
        num_groups, avg_k = group_config
        ks_cpu = tuple(aligned(int(random.uniform(0.7, 1.3) * avg_k), alignment)
                       for _ in range(num_groups))
        yield mn, ks_cpu, num_groups, gran_mn, alignment, dtype, major


def generate_psum_layout_with_sizes(group_sizes, alignment: int):
    psum_layout, psum = [], 0
    for size in group_sizes:
        psum_layout.append(psum + size)
        psum = aligned(psum + size, alignment)
    return psum, tuple(psum_layout)


def enumerate_k_grouped_psum_sf_layout():
    gran_k = 32
    for mn, ks_cpu, num_groups, gran_mn, alignment, dtype, major in enumerate_k_grouped_sf_layout():
        real_ks_cpu = tuple(k - (gran_k // 2 if i % 2 else 0) for i, k in enumerate(ks_cpu))
        aligned_ks_cpu = tuple(aligned(k, alignment) for k in real_ks_cpu)
        _, psum_layout = generate_psum_layout_with_sizes(real_ks_cpu, alignment)
        yield mn, real_ks_cpu, aligned_ks_cpu, psum_layout, num_groups, gran_mn, alignment, dtype, major


def enumerate_sf_layout_performance_tests():
    grans = (1, 32, 128)
    alignment = 256
    batches = 8
    batch_mn = 32768
    total_mn = batches * batch_mn

    for k, dtype, gran_mn in product((7168, 8192, 16384), (torch.float32, torch.int16), grans):
        yield GemmType.Batched, dtype, Major.K, batch_mn, k, gran_mn, batches, alignment, ()
        for major in (Major.K, Major.MN):
            for num_groups in (1, 128):
                group_sizes = (total_mn,) if num_groups == 1 else (total_mn // num_groups - 1,) * num_groups
                grouped_mn, psum_layout = generate_psum_layout_with_sizes(group_sizes, alignment)
                assert grouped_mn == total_mn
                yield (GemmType.MGroupedContiguousWithPsumLayout, dtype, major,
                       total_mn, k, gran_mn, num_groups, alignment, psum_layout)

            for num_groups in (1, k // alignment):
                group_sizes = (k,) if num_groups == 1 else (alignment - 1,) * num_groups
                grouped_k, psum_layout = generate_psum_layout_with_sizes(group_sizes, alignment)
                assert grouped_k == k
                yield (GemmType.KGroupedContiguousWithPsumLayout, dtype, major,
                       total_mn, k, gran_mn, num_groups, alignment, psum_layout)


def major_opt(major_a: Major, major_b: Major) -> str:
    """DeepGEMM-style NT/NN/TN/TT label for a (major_a, major_b) pair."""
    return ('N' if major_a == Major.K else 'T') + ('T' if major_b == Major.K else 'N')


def count_bytes(*tensors) -> int:
    """Total bytes across all tensors."""
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)

# (A_shape, B_shape, D_shape) and the K (contraction) axis position in each operand's
# natural 3D layout.
_EINSUM_SHAPES = {
    'bhr,hdr->bhd': lambda b, h, r, d: ((b, h, r), (h, d, r), (b, h, d)),
    'bhd,hdr->bhr': lambda b, h, r, d: ((b, h, d), (h, d, r), (b, h, r)),
    'bhd,bhr->hdr': lambda b, h, r, d: ((b, h, d), (b, h, r), (h, d, r)),
}
_EINSUM_K_AXIS = {
    'bhr,hdr->bhd': (2, 2),   # K=r : A[b,h,r] axis2, B[h,d,r] axis2
    'bhd,hdr->bhr': (2, 1),   # K=d : A[b,h,d] axis2, B[h,d,r] axis1
    'bhd,bhr->hdr': (0, 0),   # K=b : A[b,h,d] axis0, B[b,h,r] axis0
}


def _einsum_shapes(expr: str, b: int, h: int, r: int, d: int):
    """Return (A_shape, B_shape, D_shape) concrete sizes for an expr given symbol sizes."""
    return _EINSUM_SHAPES[expr](b, h, r, d)


def enumerate_einsum() -> Generator:
    hrd = [(2, 2, 2), (128, 512, 128), (8, 4096, 1024)]
    for expr in ('bhr,hdr->bhd', 'bhd,hdr->bhr'):     # forward, dgrad (K=r / K=d)
        for h, r, d in hrd:
            for b in (4, 32, 128, 4096, 8192):
                yield expr, b, h, r, d
    for h, r, d in [(8, 4096, 1024)]:                  # wgrad (K=b -> keep b 32-aligned)
        for b in (4096, 8192):
            yield 'bhd,bhr->hdr', b, h, r, d


def generate_einsum(expr: str, b: int, h: int, r: int, d: int,
                    out_dtype: torch.dtype, dtype: torch.dtype):
    assert dtype in (torch.float8_e4m3fn, torch.bfloat16)
    reset_seed()
    sa_shape, sb_shape, sd_shape = _einsum_shapes(expr, b, h, r, d)
    ka, kb = _EINSUM_K_AXIS[expr]
    # Wgrad accumulates (D = C + dW); the accumulation path is fp32 (C must match D's dtype),
    # so wgrad always outputs fp32 regardless of the requested out_dtype.
    is_wgrad = (expr == 'bhd,bhr->hdr')
    od = torch.float if is_wgrad else out_dtype

    a_in = torch.randn(sa_shape, dtype=torch.float32, device='npu') * 0.1
    b_in = torch.randn(sb_shape, dtype=torch.float32, device='npu') * 0.1
    c = (torch.randn(sd_shape, device='npu', dtype=torch.float) * 0.1) if is_wgrad else None
    d = torch.empty(sd_shape, device='npu', dtype=od)

    if dtype == torch.bfloat16:
        a = a_in.to(torch.bfloat16)
        b = b_in.to(torch.bfloat16)
        ref = torch.einsum(expr, a.float(), b.float())
        if c is not None:
            ref = ref + c.float()
        return a, b, c, d, ref.to(od)

    # FP8 MX: gran_k = 32 along the contraction axis, gran = 1 on the other two.
    gran_k = 32
    gran_a = [1, 1, 1]
    gran_a[ka] = gran_k
    gran_b = [1, 1, 1]
    gran_b[kb] = gran_k
    a, a_repr = quantize_with_major(a_in, gran_a, dtype, major_dim=ka)
    b, b_repr = quantize_with_major(b_in, gran_b, dtype, major_dim=kb)

    ref = torch.einsum(expr, a_repr, b_repr)
    if c is not None:
        ref = ref + c.float()
    return a, b, c, d, ref.to(od)
