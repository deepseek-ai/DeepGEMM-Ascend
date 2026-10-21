#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>

#include "../jit/jit.hpp"
#include "../utils/math.hpp"
#include "gemm_config.hpp"

namespace deep_gemm {

struct HWDesc {
    int gm_bw_gbps, l2_bw_gbps;
    int aic_clock_mhz;
    int bf16_flops_per_cycle;
    int gm_to_l1_bytes_per_cycle;
    int l1_to_l0_bytes_per_cycle;
    int l1_to_l0_sf_bytes_per_cycle;
    int l0c_to_ub_bytes_per_cycle;
    int ub_to_l1_bytes_per_cycle;
    int vector_reg_bytes, fp4_decode_bytes_per_cycle;
    int ub_bank_bytes, l1_bank_bytes;
    int num_pipe_events, num_intra_block_events;
    int num_ai_cores;
    int64_t l2_size_bytes;
    int l2_sector_bytes, sf_elements_per_pair, fractal_row_bytes;

    constexpr int get_mad_alignment_mn() const {
        return FRAC_MN;
    }

    constexpr int get_mad_alignment_k(int elem_bits) const {
        return fractal_row_bytes * 8 / elem_bits;
    }

    constexpr int get_mad_flops_per_cycle(int elem_bits) const {
        return bf16_flops_per_cycle * (sizeof(uint16_t) * 8) / elem_bits;
    }

};

static constexpr HWDesc get_hw_desc() {
    // Ascend 950: effective chip rates and CANN's per-core UB -> L1 rate.
    return {
        .gm_bw_gbps=3500, .l2_bw_gbps=4500,
        .aic_clock_mhz=1650,
        .bf16_flops_per_cycle=8192,  // 4096 multiply-adds; each counts as two FLOPs.
        .gm_to_l1_bytes_per_cycle=256,
        .l1_to_l0_bytes_per_cycle=256,
        .l1_to_l0_sf_bytes_per_cycle=32,  // Measured SF transfer rate.
        .l0c_to_ub_bytes_per_cycle=128,
        .ub_to_l1_bytes_per_cycle=128,
        .vector_reg_bytes=256, .fp4_decode_bytes_per_cycle=64,  // 512 decoded bytes per eight vector cycles.
        .ub_bank_bytes=32,
        .l1_bank_bytes=32,
        .num_pipe_events=kNumPipeEvents, .num_intra_block_events=16,
        .num_ai_cores=32,
        .l2_size_bytes=128 * 1024 * 1024,
        .l2_sector_bytes=kL2SectorBytes, .sf_elements_per_pair=64, .fractal_row_bytes=32,
    };
}

struct GemmDtype {
    int elem_bits;  // Operand width in L1; dequantized B is FP8 here.
    int epilogue_elem_bits;
    bool has_sf = false;
    bool dequant_b = false;
};

static GemmDtype get_gemm_dtype(const GemmDesc& d) {
    const bool with_transform = dynamic_cast<const IdentityEpilogue*>(d.epilogue_class.get()) == nullptr;
    int epilogue_elem_bits = (d.cd_dtype == at::kBFloat16 && !with_transform) || d.cd_dtype == at::kFloat8_e4m3fn ? 16 : 32;
    if (d.a_dtype == at::kBFloat16 && d.b_dtype == at::kBFloat16)
        return {.elem_bits=16, .epilogue_elem_bits=epilogue_elem_bits};
    if (d.a_dtype == kPackedFP4 && d.b_dtype == kPackedFP4)
        return {.elem_bits=4, .epilogue_elem_bits=epilogue_elem_bits, .has_sf=true};
    DJ_HOST_ASSERT(d.a_dtype == at::kFloat8_e4m3fn);
    DJ_HOST_ASSERT(d.b_dtype == at::kFloat8_e4m3fn || d.b_dtype == kPackedFP4);
    return {.elem_bits=8, .epilogue_elem_bits=epilogue_elem_bits, .has_sf=true, .dequant_b=d.b_dtype == kPackedFP4};
}

static std::optional<GemmConfig> make_gemm_candidate(const GemmDesc& d, const GemmDtype& dtype, int block_m, int block_n, int block_k) {
    const auto hw = get_hw_desc();
    GemmConfig c{};
    c.block_m = block_m;
    c.block_n = block_n;
    c.block_k = block_k;
    c.launch_options.num_blocks = d.num_cores;

    // Check BLOCK_M, BLOCK_N: must fit L0C
    const uint64_t block_bytes = static_cast<uint64_t>(block_m) * block_n * sizeof(float);
    if (block_bytes > L0CSizeBytes) return std::nullopt;

    // Group starts must remain aligned when using smaller compute tiles.
    if (is_m_grouped(d.gemm_type) && Runtime::get_mk_alignment_for_contiguous_layout() % block_m != 0) return std::nullopt;

    const int k_alignment = dtype.has_sf ? hw.sf_elements_per_pair : hw.get_mad_alignment_k(dtype.elem_bits);
    if (block_k % k_alignment) return std::nullopt;

    // Select L0 stages: always 2 stage
    c.num_l0_stages = 2;

    const bool with_bias = dynamic_cast<const BiasEpilogue*>(d.epilogue_class.get()) != nullptr;

    // Search MAD_M, MAD_N, MAD_K
    // 1. at least 2 stage in L0C, to overlap mad and store
    // 2. fit l0a and l0b, to overlap load and mad
    const bool with_fp8_cast = d.cd_dtype == at::kFloat8_e4m3fn;
    const int align_m = is_mn_major(d.major_a) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();
    const int align_n = with_bias ? hw.vector_reg_bytes / sizeof(float) : with_fp8_cast ? hw.vector_reg_bytes / sizeof(uint16_t) : is_mn_major(d.major_b) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();
    c.mad_m = c.mad_n = c.mad_k = 0;
    // search mad_m = block_m / split_m, mad_n = block_n / split_n
    for (auto [split_m, split_n]: {std::pair{1, 1}, std::pair{1, 2}, std::pair{2, 1}}) {
        if (block_m % (split_m * align_m) || block_n % (split_n * align_n)) continue;
        const int mad_m = block_m / split_m, mad_n = block_n / split_n;
        if (mad_m * mad_n * sizeof(float) > L0CSizeBytes / 2) continue;

        // Determine MAD_K:
        // 1. ensure enough l0a and l0b stages
        // 2. ensure mad_k is aligned to k_alignment
        // 3. ensure mad_k is a divisor of block_k
        const int l0a_k = L0ASizeBytes * 8 / (c.num_l0_stages * mad_m * dtype.elem_bits);
        const int l0b_k = L0BSizeBytes * 8 / (c.num_l0_stages * mad_n * dtype.elem_bits);
        int mad_k = std::min({l0a_k, l0b_k, std::max(k_alignment, block_k / static_cast<int>(c.num_l0_stages))});
        mad_k = mad_k / k_alignment * k_alignment;
        while (mad_k && block_k % mad_k) mad_k -= k_alignment;
        // Split MN before shortening the base K block's double-buffered L0 stages.
        const int mte1_copy_k = hw.l1_to_l0_bytes_per_cycle * 8 / dtype.elem_bits;
        // Prevent MAD_K from being too small, which wastes mte1 bandwidth and causes a heavy copy amplification
        // TODO: this is a bad heuristic, we should consider a better one, such as enumerate some possible MAD_K and select by the cost
        if (mad_k < std::min(block_k, mte1_copy_k) / static_cast<int>(c.num_l0_stages)) continue;
        c.mad_m = mad_m;
        c.mad_n = mad_n;
        c.mad_k = mad_k;
        break;
    }
    if (!c.mad_m || !c.mad_n || !c.mad_k) return std::nullopt;

    // Dequant kernel supports BLOCK_K <= 512 and 256 / 512 row elements for B.
    if (dtype.dequant_b) {
        const int row_elements = is_k_major(d.major_b) ? block_k : block_n;
        if (block_k > 512 || (row_elements != 256 && row_elements != 512)) return std::nullopt;
    }

    // Select SF_K_BLOCKS
    if (dtype.has_sf) {
        const int k = is_k_grouped(d.gemm_type) ? d.num_groups * d.expected_k : d.k;
        c.sf_k_blocks = std::min<int>(4, ceil_div(k, block_k));
        c.num_l1_sf_stages = 2;
    } else {
        c.sf_k_blocks = 0;
        c.num_l1_sf_stages = 0;
    }

    // Share L1 capacity and pipe events between operands and scale factors.
    constexpr int kNumMaxL1StagesWithSF = 4;
    const int num_max_l1_stages = dtype.has_sf ? std::min(kNumMaxL1StagesWithSF, hw.num_pipe_events - static_cast<int>(c.num_l1_sf_stages)) : hw.num_pipe_events;
    if (num_max_l1_stages < 2) return std::nullopt;
    const int64_t sf_bytes = static_cast<int64_t>(block_m + block_n) * (block_k / hw.sf_elements_per_pair) * sizeof(int16_t) * c.sf_k_blocks * c.num_l1_sf_stages;
    const int64_t ab_bytes = static_cast<int64_t>(block_m + block_n) * block_k * dtype.elem_bits / 8;
    const int num_l1_stages = std::min<int64_t>(num_max_l1_stages, (static_cast<int64_t>(L1SizeBytes) - sf_bytes) / ab_bytes);
    if (num_l1_stages < 2) return std::nullopt;
    c.num_l1_stages = num_l1_stages;

    // Native SF loads overlap operand transfers; dequant loads SF before A.
    if (dtype.has_sf) {
        const int k = is_k_grouped(d.gemm_type) ? d.num_groups * d.expected_k : d.k;
        const int num_k_blocks = ceil_div(k, block_k);
        const int sf_k_blocks = dtype.dequant_b ? ceil_div(num_k_blocks, c.num_l1_sf_stages) : num_k_blocks;
        const int num_sf_k_blocks_fit = (L1SizeBytes - ab_bytes * num_l1_stages) / (sf_bytes / c.sf_k_blocks);
        c.sf_k_blocks = std::clamp(sf_k_blocks, static_cast<int>(c.sf_k_blocks), num_sf_k_blocks_fit);
    }

    // Match epilogue::Params and operator UB layouts when selecting stages per AIV.
    const uint32_t num_aivs = dtype.dequant_b ? 1 : 2;
    const uint32_t num_cd_stages = (block_m / c.mad_m) * (block_n / c.mad_n);
    const uint32_t epilogue_bytes = c.mad_m * c.mad_n * dtype.epilogue_elem_bits / 8 +
                                    (with_bias ? block_n * c10::elementSize(d.cd_dtype) : 0) + (with_fp8_cast ? c.mad_m * c.mad_n / 32 : 0);
    c.num_epilogue_stages = std::min<uint32_t>(hw.num_intra_block_events, UBSizeBytes / epilogue_bytes);
    const uint32_t num_batches = is_batched(d.gemm_type) || is_k_grouped(d.gemm_type) ? d.num_groups : 1;
    const uint64_t num_blocks = static_cast<uint64_t>(ceil_div(d.m, block_m)) * ceil_div(d.n, block_n) * num_batches;
    c.num_epilogue_stages = std::min<uint64_t>(c.num_epilogue_stages, ceil_div(num_blocks, d.num_cores) * num_cd_stages);
    if (c.num_epilogue_stages * num_aivs < num_cd_stages) return std::nullopt;

    // Select Cache Hints
    // TODO: smarter heuristics for L2 hints
    const int num_m_blocks = ceil_div(d.m, block_m);
    const int num_n_blocks = ceil_div(d.n, block_n);
    c.l2_ctrl_a = num_n_blocks <= 2 && num_m_blocks > num_n_blocks ? asc_load_l2_cache_mode::NOTALLOC_KEEP : asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    c.l2_ctrl_b = num_m_blocks <= 2 && num_n_blocks > num_m_blocks ? asc_load_l2_cache_mode::NOTALLOC_KEEP : asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;

    return c;
}

struct GemmComputeCost {
    int64_t mflops, time_ns, num_tail_blocks;
};

static GemmComputeCost get_gemm_compute_cost(const GemmDesc& d, const GemmDtype& dtype, int block_m, int block_n) {
    constexpr int kBlockSwizzleM = 4;
    const auto hw = get_hw_desc();
    const int m = d.m, n = d.n;
    const int64_t k = is_k_grouped(d.gemm_type) ? d.expected_k : d.k;
    const int num_m_blocks = ceil_div(m, block_m);
    const int num_n_blocks = ceil_div(n, block_n);
    const int64_t num_blocks = static_cast<int64_t>(num_m_blocks) * num_n_blocks;
    const int num_batches = is_batched(d.gemm_type) || is_k_grouped(d.gemm_type) ? d.num_groups : 1;
    const int64_t num_waves = ceil_div(num_blocks * num_batches, d.num_cores);

    const int align_m = is_mn_major(d.major_a) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();
    const int align_n = is_mn_major(d.major_b) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();

    // Derive Actual MatMul Shape
    // M-Grouped GEMM always uses BLOCK_M, rather than Expected M
    // TODO: pad zero manually in output buffer for MGroupedLayout
    const int compute_m = is_m_grouped(d.gemm_type) ? block_m : align(std::min(m, block_m), align_m);
    const int compute_n = align(std::min(n, block_n), align_n);

    // Derive MatMul FLOPS and MatMul Time
    int64_t compute_mn = num_waves * compute_m * compute_n;
    int64_t num_tail_blocks = 0;
    const int tail_stride = num_m_blocks % kBlockSwizzleM ? 1 : kBlockSwizzleM;
    // Adjust compute MN for the tail blocks
    if (!is_m_grouped(d.gemm_type) && std::gcd(tail_stride, d.num_cores) == 1) {
        const int num_core_classes = std::gcd(num_blocks, d.num_cores);
        const int num_groups_per_period = d.num_cores / num_core_classes;
        const int num_remaining_groups = num_batches % num_groups_per_period;
        const int64_t num_full_per_period = ceil_div(num_blocks - num_n_blocks, num_core_classes);
        const int tail_remainder = num_n_blocks % d.num_cores;
        const int tail_suffix = tail_stride == 1 ? tail_remainder : std::min(tail_remainder, 1);
        const int64_t num_full_per_group = ceil_div(num_blocks - tail_suffix, d.num_cores) - num_n_blocks / d.num_cores;
        // Bound full blocks on the busiest core over complete periods and the remaining groups.
        const int64_t num_full_remaining = std::min(num_remaining_groups * num_full_per_group, num_full_per_period);
        const int64_t num_full_blocks = num_batches / num_groups_per_period * num_full_per_period + num_full_remaining;
        num_tail_blocks = std::max<int64_t>(0, num_waves - num_full_blocks);
        const int tail_m = align(m - (num_m_blocks - 1) * block_m, align_m);
        compute_mn -= num_tail_blocks * (compute_m - tail_m) * compute_n;
    }
    const int64_t compute_flops = compute_mn * 2 * k;
    const int64_t compute_mflops = static_cast<int64_t>(hw.get_mad_flops_per_cycle(dtype.elem_bits)) * hw.aic_clock_mhz;
    const int64_t compute_time_ns = ceil_div(compute_flops * 1000, compute_mflops);
    return {.mflops=compute_mflops, .time_ns=compute_time_ns, .num_tail_blocks=num_tail_blocks};
}

static auto get_gemm_cost(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c, const GemmComputeCost& compute_cost, int64_t best_cost = std::numeric_limits<int64_t>::max()) {
    constexpr int kBlockSwizzleM = 4;
    const auto hw = get_hw_desc();
    // Single-M scheduling keeps actual extents and traffic; no scalar-cycle discount is modeled.
    const int m = d.m, n = d.n;
    const int64_t k = is_k_grouped(d.gemm_type) ? d.expected_k : d.k;
    const int block_m = c.block_m, block_n = c.block_n;
    const int num_m_blocks = ceil_div(m, block_m);
    const int num_n_blocks = ceil_div(n, block_n);
    const int64_t num_blocks = static_cast<int64_t>(num_m_blocks) * num_n_blocks;
    const int num_batches = is_batched(d.gemm_type) || is_k_grouped(d.gemm_type) ? d.num_groups : 1;
    const int64_t num_waves = ceil_div(num_blocks * num_batches, d.num_cores);
    const int b_elem_bits = dtype.dequant_b ? dtype.elem_bits / 2 : dtype.elem_bits;
    const int64_t sf_row_bytes = dtype.has_sf ? ceil_div(k, hw.sf_elements_per_pair) * sizeof(int16_t) : 0;

    const int align_m = is_mn_major(d.major_a) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();
    const int align_n = is_mn_major(d.major_b) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();

    // Derive Actual MatMul Shape
    // M-Grouped GEMM always uses BLOCK_M, rather than Expected M
    // TODO: pad zero manually in output buffer for MGroupedLayout
    const int compute_m = is_m_grouped(d.gemm_type) ? block_m : align(std::min(m, block_m), align_m);
    const int compute_n = align(std::min(n, block_n), align_n);

    const auto [compute_mflops, compute_time_ns, num_tail_blocks] = compute_cost;

    // Derive the memory loads in a block (BLOCK_M, BLOCK_N)
    const auto gm_load_bytes = [&](int mn, int64_t k, int total_mn, int elem_bits, Major major, int64_t outer_stride) -> int64_t {
        const bool mn_major = is_mn_major(major);
        const int64_t row_bytes = ceil_div((mn_major ? total_mn : k) * elem_bits, 8);
        const int64_t block_bytes = ceil_div((mn_major ? mn : c.block_k) * elem_bits, 8);
        // Packed FP4 strides count bytes, while its logical extents count nibbles.
        const int64_t stride_bytes = outer_stride ? outer_stride * std::max(elem_bits, 8) / 8 : row_bytes;
        const int64_t num_row_blocks = ceil_div(row_bytes, block_bytes);
        const int64_t stride_gcd = std::gcd(stride_bytes, int64_t(hw.l2_sector_bytes));
        const int64_t boundary_period = stride_gcd / std::gcd(block_bytes, stride_gcd);
        // Average covered sectors over row-address phases, including shared block boundaries.
        const int64_t row_load_bytes = align(row_bytes, stride_gcd) + num_row_blocks * hw.l2_sector_bytes
                                    - stride_gcd * (1 + (num_row_blocks - 1) / boundary_period);
        return mn_major ? ceil_div(k * row_load_bytes, num_row_blocks) : mn * row_load_bytes;
    };
    const auto sf_load_bytes = [&](int mn, int64_t k, int total_mn) -> int64_t {
        return dtype.has_sf ? gm_load_bytes(mn, ceil_div(k, hw.sf_elements_per_pair), total_mn, sizeof(int16_t) * 8, Major::MN, total_mn) : 0;
    };
    const int load_m = is_m_grouped(d.gemm_type) ? num_m_blocks * block_m : m;
    const int64_t sfa_block_bytes = sf_load_bytes(block_m, k, load_m);
    const int64_t sfb_block_bytes = sf_load_bytes(block_n, k, n);
    int64_t wa_block_bytes = gm_load_bytes(block_m, k, load_m, dtype.elem_bits, d.major_a, d.outer_stride_a);
    const int64_t wb_block_bytes = gm_load_bytes(block_n, k, n, b_elem_bits, d.major_b, d.outer_stride_b);
    if (dtype.elem_bits == 16 && c.resident_a && is_k_major(d.major_a))
        wa_block_bytes = int64_t(block_m) * align(k * sizeof(uint16_t), hw.l2_sector_bytes);
    const int64_t a_block_bytes = wa_block_bytes + sfa_block_bytes;
    const int64_t b_block_bytes = wb_block_bytes + sfb_block_bytes;

    // L2 residency uses logical bytes within one group; SF inherits its operand's hint.
    const int64_t l2_a_row_bytes = c.l2_ctrl_a == asc_load_l2_cache_mode::NOTALLOC_KEEP ? 0 : sf_row_bytes + ceil_div(k * dtype.elem_bits, 8);
    const int64_t l2_b_row_bytes = c.l2_ctrl_b == asc_load_l2_cache_mode::NOTALLOC_KEEP ? 0 : sf_row_bytes + ceil_div(k * b_elem_bits, 8);

    // Derive the total cold GM loads for A/B and SF of all blocks
    auto [total_gm_a_bytes, total_gm_b_bytes] = [&](){
        const int block_swizzle_m = std::min(num_m_blocks, kBlockSwizzleM);
        const int block_swizzle_n = std::min(num_n_blocks, hw.num_ai_cores / block_swizzle_m);
        const int64_t l2_block_swizzle_a_bytes = std::min(m, block_swizzle_m * block_m) * l2_a_row_bytes;
        const int64_t l2_block_swizzle_b_bytes = std::min(n, block_swizzle_n * block_n) * l2_b_row_bytes;

        // First use loads each A/B block from GM; eviction reloads it at each swizzle boundary.
        int64_t num_a_gm_blocks = static_cast<int64_t>(num_m_blocks) * num_batches;
        int64_t num_b_gm_blocks = static_cast<int64_t>(num_n_blocks) * num_batches;
        if (l2_block_swizzle_a_bytes + l2_block_swizzle_b_bytes > hw.l2_size_bytes) {
            num_a_gm_blocks *= ceil_div(num_n_blocks, block_swizzle_n);
        }
        if (l2_block_swizzle_a_bytes + n * l2_b_row_bytes > hw.l2_size_bytes) {
            num_b_gm_blocks *= ceil_div(num_m_blocks, block_swizzle_m);
        }
        if (is_m_grouped(d.gemm_type)) num_b_gm_blocks = std::min(num_blocks, std::max(num_b_gm_blocks, static_cast<int64_t>(num_n_blocks) * d.num_groups));
        if (c.l2_ctrl_a == asc_load_l2_cache_mode::NOTALLOC_KEEP) num_a_gm_blocks = num_blocks * num_batches;
        if (c.l2_ctrl_b == asc_load_l2_cache_mode::NOTALLOC_KEEP) num_b_gm_blocks = num_blocks * num_batches;
        const int64_t gm_a_bytes = num_a_gm_blocks * a_block_bytes;
        const int64_t gm_b_bytes = num_b_gm_blocks * b_block_bytes;

        return std::tuple{gm_a_bytes, gm_b_bytes};
    }();

    // Derive the total load time for A/B matrix
    const auto get_load_time_ns = [&](int64_t total_gm_bytes, int64_t total_load_bytes) {
        total_gm_bytes = std::min(total_gm_bytes, total_load_bytes);
        const int64_t l2_bytes = total_load_bytes - total_gm_bytes;
        // Not all aicores are active to access GM, we use the wave utilization to estimate the effective GM bandwidth
        //   active_gm_gbps = gm_bw_gbps * (num_blocks * num_batches / (num_waves * num_ai_cores))
        //   gm_time_ns = gm_bytes / active_gm_gbps
        // The above formula is equivalent to the following, which avoids float arithmetic
        const int64_t gm_time_ns = ceil_div(static_cast<__int128>(total_gm_bytes) * num_waves * hw.num_ai_cores, num_blocks * num_batches * hw.gm_bw_gbps);
        const int64_t l2_time_ns = ceil_div(static_cast<__int128>(l2_bytes) * num_waves * hw.num_ai_cores, num_blocks * num_batches * hw.l2_bw_gbps);
        return gm_time_ns + l2_time_ns;
    };
    int64_t a_load_blocks = c.resident_a && num_m_blocks == 1 ? std::min(num_blocks, int64_t(d.num_cores)) * num_batches : num_blocks * num_batches;
    int64_t b_load_blocks = c.resident_b && num_n_blocks == 1 ? std::min(num_blocks, int64_t(d.num_cores) * (is_m_grouped(d.gemm_type) ? d.num_groups : 1)) * num_batches : num_blocks * num_batches;
    const int64_t sfa_load_blocks = c.resident_sf ? a_load_blocks : num_blocks * num_batches;
    const int64_t sfb_load_blocks = c.resident_sf ? b_load_blocks : num_blocks * num_batches;
    // Broadcast data can stay resident across batches; transformed SF remains batch-specific.
    if (dtype.has_sf && is_batched(d.gemm_type)) {
        if (c.resident_a && d.stride_batch_a == 0) a_load_blocks /= num_batches;
        if (c.resident_b && d.stride_batch_b == 0) b_load_blocks /= num_batches;
    }
    const int64_t load_a_time_ns = get_load_time_ns(total_gm_a_bytes, a_load_blocks * wa_block_bytes + sfa_load_blocks * sfa_block_bytes);
    const int64_t load_b_time_ns = get_load_time_ns(total_gm_b_bytes, b_load_blocks * wb_block_bytes + sfb_load_blocks * sfb_block_bytes);

    // UB -> L1 copies half-row B slices.
    auto [dequant_bytes, dequant_time_ns] = [&](){
        const int b_rows = is_k_major(d.major_b) ? block_n : c.block_k;
        const int contiguous_bytes = b_rows / 2 * hw.fractal_row_bytes;
        const int64_t dequant_bytes = dtype.dequant_b ? ceil_div(b_load_blocks, d.num_cores) * block_n * k * dtype.elem_bits / 8
                                   * align(contiguous_bytes, hw.ub_to_l1_bytes_per_cycle) / contiguous_bytes : 0;
        const int64_t dequant_time_ns = ceil_div(dequant_bytes * 1000, hw.ub_to_l1_bytes_per_cycle * hw.aic_clock_mhz);
        return std::tuple{dequant_bytes, dequant_time_ns};
    }();

    const int64_t memory_time_ns = load_a_time_ns + load_b_time_ns + dequant_time_ns;
    if (memory_time_ns > best_cost) return memory_time_ns;

    // Limit the compute discount by the first tail wave's cold-A service.
    const int64_t tail_memory_delay_ns = [&](){
        const int tail_m = m - (num_m_blocks - 1) * block_m;
        const int tail_compute_m = align(tail_m, align_m);
        const int64_t discount_ns = ceil_div(num_tail_blocks * (compute_m - tail_compute_m) * compute_n * 2 * k * 1000, compute_mflops);
        if (!discount_ns) return int64_t(0);

        const auto mn_tail_bytes = [&](int64_t num_rows, int elem_bits, int64_t stride_bytes) {
            const int64_t stride_gcd = std::gcd(stride_bytes, int64_t(hw.l2_sector_bytes));
            const int64_t offset_bytes = int64_t(m - tail_m) * elem_bits / 8;
            const int64_t row_bytes = ceil_div(tail_m * elem_bits, 8);
            return num_rows * (align(offset_bytes % stride_gcd + row_bytes, stride_gcd) + hw.l2_sector_bytes - stride_gcd);
        };
        const int64_t a_stride_bytes = d.outer_stride_a ? d.outer_stride_a * std::max(dtype.elem_bits, 8) / 8 : ceil_div(m * dtype.elem_bits, 8);
        const int64_t tail_a_bytes = is_mn_major(d.major_a)
            ? mn_tail_bytes(k, dtype.elem_bits, a_stride_bytes)
            : gm_load_bytes(tail_m, k, m, dtype.elem_bits, d.major_a, d.outer_stride_a);
        const int64_t tail_sfa_bytes = dtype.has_sf ? mn_tail_bytes(ceil_div(k, hw.sf_elements_per_pair), sizeof(int16_t) * 8, m * sizeof(int16_t)) : 0;
        const bool reuse_b = num_m_blocks > 1 && c.l2_ctrl_b != asc_load_l2_cache_mode::NOTALLOC_KEEP
                          && std::min(m, kBlockSwizzleM * block_m) * l2_a_row_bytes + n * l2_b_row_bytes <= hw.l2_size_bytes;
        const int num_tail_wave_blocks = std::min<int>(num_n_blocks, d.num_cores);
        const int64_t tail_memory_ns = ceil_div((tail_a_bytes + tail_sfa_bytes) * num_tail_wave_blocks, hw.gm_bw_gbps)
                                     + ceil_div(b_block_bytes * num_tail_wave_blocks, reuse_b ? hw.l2_bw_gbps : hw.gm_bw_gbps)
                                     + ceil_div(dequant_time_ns, num_waves);
        const int64_t tail_compute_ns = ceil_div(int64_t(tail_compute_m) * compute_n * 2 * k * 1000, compute_mflops);
        const int64_t num_cold_tail_waves = c.l2_ctrl_a == asc_load_l2_cache_mode::NOTALLOC_KEEP ? num_tail_blocks : std::min<int64_t>(num_batches, num_tail_blocks);
        return std::min(discount_ns, num_cold_tail_waves * std::max<int64_t>(0, tail_memory_ns - tail_compute_ns));
    }();

    // First fill includes the full resident-A load before the K loop; it cannot overlap compute.
    auto first_load_time_ns = [&](){
        const int64_t first_k = std::min<int64_t>(k, c.block_k);
        const int64_t first_sf_k = std::min<int64_t>(k, c.block_k * c.sf_k_blocks);
        const int64_t first_sfa_bytes = sf_load_bytes(block_m, first_sf_k, load_m);
        const int64_t first_sfb_bytes = sf_load_bytes(block_n, first_sf_k, n);
        const int64_t first_a_bytes = dtype.elem_bits == 16 && c.resident_a ? a_block_bytes : gm_load_bytes(block_m, first_k, load_m, dtype.elem_bits, d.major_a, d.outer_stride_a) + first_sfa_bytes;
        const int64_t first_b_bytes = gm_load_bytes(block_n, first_k, n, b_elem_bits, d.major_b, d.outer_stride_b) + first_sfb_bytes;

        // Estimate A, B, Dequant, SF time by the total stage time with the following formula:
        //    first-stage-time = total-stage-time * (first-stage-bytes / total-bytes)
        const int64_t first_dequant_ns = ceil_div(dequant_bytes * first_k / k * 1000, hw.ub_to_l1_bytes_per_cycle * hw.aic_clock_mhz);
        const int64_t first_stage_time_ns = (
            ceil_div(load_a_time_ns * first_a_bytes, a_block_bytes) +
            ceil_div(load_b_time_ns * first_b_bytes, b_block_bytes) +
            first_dequant_ns
        );
        const int64_t sf_startup_time_ns = (
            ceil_div(load_a_time_ns * first_sfa_bytes, a_block_bytes) +
            ceil_div(load_b_time_ns * first_sfb_bytes, b_block_bytes)
        );

        const int64_t prefetch_k = std::min<int64_t>(k, (c.num_l1_stages - 1) * c.block_k);
        const int64_t prefetch_compute_ns = ceil_div(prefetch_k * 2 * compute_m * compute_n * 1000, compute_mflops);
        const int64_t sf_overlap_time_ns = c.num_l1_sf_stages > 1 ? std::min(sf_startup_time_ns / num_waves, prefetch_compute_ns) : 0;
        const int64_t first_load_time_ns = first_stage_time_ns - (num_waves - 1) * sf_overlap_time_ns;
        return first_load_time_ns;
    }();

    // Derive the time to load A/B/SF from L1 to L0, this is overlapped with compute
    const int64_t l0_time_ns = [&](){
        // Measure l0 reuse: when m,k or n,k is small, they can reside in L0
        const int num_l0a_loads = c.block_k <= c.mad_k * c.num_l0_stages ? 1 : block_n / c.mad_n;
        const int num_l0b_loads = block_n == c.mad_n && c.block_k <= c.mad_k * c.num_l0_stages ? 1 : block_m / c.mad_m;
        const int64_t num_l0_rows = num_waves * (block_m * num_l0a_loads + block_n * num_l0b_loads);
        const int64_t a_bytes = num_waves * block_m * num_l0a_loads * k * dtype.elem_bits / 8;
        const int64_t b_bytes = (dtype.elem_bits == 16 && c.resident_b ? 1 : num_waves) * block_n * num_l0b_loads * k * dtype.elem_bits / 8;

        // Measure L0 copy amplification, if the copy is not aligned to the transfer bytes, the bandwidth is wasted
        const int a_contiguous_bytes = (is_k_major(d.major_a) ? c.mad_m : c.mad_k) * hw.fractal_row_bytes;
        const int b_contiguous_bytes = (is_k_major(d.major_b) ? c.mad_n : c.mad_k) * hw.fractal_row_bytes;
        const int64_t ab_bytes = a_bytes * align(a_contiguous_bytes, hw.l1_to_l0_bytes_per_cycle) / a_contiguous_bytes
                               + b_bytes * align(b_contiguous_bytes, hw.l1_to_l0_bytes_per_cycle) / b_contiguous_bytes;
        const int64_t sf_bytes = num_l0_rows * sf_row_bytes;

        const int64_t l0_time_ns = (
            ceil_div(ab_bytes * 1000, hw.l1_to_l0_bytes_per_cycle * hw.aic_clock_mhz) +
            ceil_div(sf_bytes * 1000, hw.l1_to_l0_sf_bytes_per_cycle * hw.aic_clock_mhz)
        );
        return l0_time_ns;
    }();

    // Derive the compute-bound total time
    // TODO: add first_l0_load_ns and last_mte3_copy_ns
    const int64_t compute_bound_time_ns = first_load_time_ns + std::max(l0_time_ns, compute_time_ns + tail_memory_delay_ns);

    // Total loads include the first fill; the last K block still needs its MADs.
    // TODO: add last_mte3_copy_ns
    const int64_t last_k = k % c.block_k ? k % c.block_k : c.block_k;
    const int64_t last_compute_time_ns = ceil_div(last_k * compute_m * compute_n * 2 * 1000, compute_mflops);
    const int64_t memory_bound_time_ns = memory_time_ns + last_compute_time_ns;

    return std::max(compute_bound_time_ns, memory_bound_time_ns);
}

static auto get_gemm_cost(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c) {
    return get_gemm_cost(d, dtype, c, get_gemm_compute_cost(d, dtype, c.block_m, c.block_n));
}

static int64_t get_gemm_output_cost(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c) {
    const auto hw = get_hw_desc();
    const int64_t num_waves = ceil_div(int64_t(ceil_div(d.m, c.block_m)) * ceil_div(d.n, c.block_n), d.num_cores);
    const int64_t num_values = int64_t(std::min(d.m, c.block_m)) * std::min(d.n, c.block_n);
    const int64_t bytes = num_values * dtype.epilogue_elem_bits / 8;
    const bool with_bias = dynamic_cast<const BiasEpilogue*>(d.epilogue_class.get()) != nullptr;
    // AddBias owns one BLOCK_N row per AIV drain; its hooks preserve the existing overlap.
    const int64_t bias_bytes = with_bias ? (dtype.dequant_b ? 1 : 2) *
                               std::min(d.n, c.block_n) * c10::elementSize(d.cd_dtype) : 0;
    const int64_t bias_ns = ceil_div(bias_bytes * hw.num_ai_cores, hw.gm_bw_gbps);
    const int64_t gm_bytes = num_values * c10::elementSize(d.cd_dtype) * (1 + d.acc)
                           + (d.cd_dtype == at::kFloat8_e4m3fn ? num_values / 32 : 0);
    const int64_t fix_ns = ceil_div(bytes * 1000, hw.l0c_to_ub_bytes_per_cycle * hw.aic_clock_mhz);
    const int64_t gm_ns = ceil_div(gm_bytes * hw.num_ai_cores, hw.gm_bw_gbps);
    const int64_t vector_values = int64_t(align(std::min(d.m, c.block_m), c.mad_m)) * align(std::min(d.n, c.block_n), c.mad_n);
    // Pure AddBias performs one vector add per FP32 vector; BF16 packing is fused into its store.
    const int64_t vector_ns = with_bias ? ceil_div(ceil_div(vector_values * sizeof(float), hw.vector_reg_bytes) * 1000, hw.aic_clock_mhz) : 0;
    if (with_bias && c.num_epilogue_stages >= 2 && num_waves >= 2)
        return num_waves * std::max(fix_ns, bias_ns + gm_ns + vector_ns);
    return num_waves * (c.direct_store ? std::max(fix_ns, gm_ns) : std::max(fix_ns, bias_ns) + gm_ns + vector_ns);
}

static int64_t get_gemm_refinement_cost(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c) {
    const auto hw = get_hw_desc();
    int64_t cost = get_gemm_cost(d, dtype, c);
    if (dtype.elem_bits == 16 && d.m < c.block_m) {
        constexpr int kGmLoadIssueCycles = 64;  // Resident-A source calibration, separate from transfer bytes.
        const int64_t num_waves = ceil_div(ceil_div(d.n, c.block_n), d.num_cores);
        const int num_k_blocks = ceil_div(d.k, c.block_k);
        const int64_t num_loads = (c.resident_a ? 1 : num_waves * num_k_blocks) + num_waves * num_k_blocks;
        cost += ceil_div(num_loads * kGmLoadIssueCycles * 1000, hw.aic_clock_mhz);
    }
    if (dtype.dequant_b) {
        const int64_t num_blocks = int64_t(ceil_div(d.m, c.block_m)) * ceil_div(d.n, c.block_n);
        const int num_batches = is_batched(d.gemm_type) || is_k_grouped(d.gemm_type) ? d.num_groups : 1;
        int64_t b_load_blocks = c.resident_b ? std::min(num_blocks, int64_t(d.num_cores) * (is_m_grouped(d.gemm_type) ? d.num_groups : 1)) * num_batches : num_blocks * num_batches;
        if (c.resident_b && is_batched(d.gemm_type) && d.stride_batch_b == 0) b_load_blocks /= num_batches;
        const int num_aivs = c.dual_aiv_dequant ? 2 : 1;
        const int64_t decoded_bytes = ceil_div(b_load_blocks, d.num_cores) * c.block_n * (is_k_grouped(d.gemm_type) ? d.expected_k : d.k);
        const int64_t decode_ns = ceil_div(decoded_bytes * 1000, num_aivs * hw.fp4_decode_bytes_per_cycle * hw.aic_clock_mhz);
        const int64_t store_l1_ns = ceil_div(decoded_bytes * 1000, hw.ub_to_l1_bytes_per_cycle * hw.aic_clock_mhz);
        const int64_t compute_ns = get_gemm_compute_cost(d, dtype, c.block_m, c.block_n).time_ns;
        cost += std::max<int64_t>(0, decode_ns - std::max(compute_ns, store_l1_ns));
    }
    return cost + get_gemm_output_cost(d, dtype, c);
}

static bool can_direct_store(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c) {
    const auto hw = get_hw_desc();
    const auto stride_d = d.outer_stride_d ? d.outer_stride_d : d.n;
    const int align_m = is_mn_major(d.major_a) ? hw.get_mad_alignment_k(dtype.elem_bits) : 1;
    const int align_n = is_mn_major(d.major_b) ? hw.get_mad_alignment_k(dtype.elem_bits) : FRAC_MN;
    return d.gemm_type == GemmType::Normal && dynamic_cast<const IdentityEpilogue*>(d.epilogue_class.get()) &&
           (dtype.elem_bits != 4 || (is_k_major(d.major_a) && is_k_major(d.major_b))) &&
           (!d.acc || d.cd_dtype == at::kFloat) && d.m % align_m == 0 && d.n % align_n == 0 &&
           reinterpret_cast<uintptr_t>(d.d_ptr) % hw.ub_bank_bytes == 0 &&
           stride_d * dtype.epilogue_elem_bits / 8 % hw.ub_bank_bytes == 0 &&
           (int64_t(ceil_div(d.m, c.block_m)) * ceil_div(d.n, c.block_n) <= d.num_cores ||
            (d.m <= c.block_m && c.mad_n * dtype.epilogue_elem_bits / 8 % hw.l2_sector_bytes == 0));
}

static GemmConfig refine_gemm_pipeline(const GemmDesc& d, const GemmDtype& dtype, GemmConfig config) {
    const auto hw = get_hw_desc();
    const int num_m_blocks = ceil_div(d.m, config.block_m), num_n_blocks = ceil_div(d.n, config.block_n);
    const int num_waves = ceil_div(num_m_blocks * num_n_blocks, d.num_cores);
    if (can_direct_store(d, dtype, config)) {
        auto direct = config;
        direct.direct_store = true;
        if (get_gemm_output_cost(d, dtype, direct) <= get_gemm_output_cost(d, dtype, config)) config = direct;
    }
    const int compiled_k = get_compiled_dim(d.k, 'k', d.compiled_dims);
    const int resident_k = is_k_grouped(d.gemm_type) ? d.expected_k : d.k;
    if ((d.gemm_type == GemmType::Normal && compiled_k) || (dtype.has_sf && (compiled_k || is_k_grouped(d.gemm_type)))) {
        auto resident = config;
        if (dtype.elem_bits == 16) {
            const int64_t a_bytes = int64_t(config.block_m) * align(d.k, FRAC_MN) * sizeof(uint16_t);
            const int64_t b_bytes = int64_t(config.block_n) * config.block_k * sizeof(uint16_t);
            const int num_l1_stages = a_bytes < L1SizeBytes ? std::min<int64_t>(hw.num_pipe_events, (L1SizeBytes - a_bytes) / b_bytes) : 0;
            if (d.m <= config.block_m && num_l1_stages > config.num_l0_stages) {
                resident.resident_a = true;
                resident.num_l1_stages = num_l1_stages;
            }
            resident.resident_b = num_n_blocks == 1 && num_waves > 1 && config.block_n == config.mad_n &&
                                  d.k <= config.mad_k * config.num_l0_stages;
        } else if (resident_k <= config.num_l1_stages * config.block_k && num_waves >= 2) {
            // Operands retain their own slots while the load/event rings advance.
            resident.resident_a = num_m_blocks == 1;
            resident.resident_b = num_n_blocks == 1 && (!dtype.dequant_b || !is_m_grouped(d.gemm_type) || ceil_div(resident_k, config.block_k) >= 2);
            resident.resident_sf = (resident.resident_a || resident.resident_b) && resident_k <= config.sf_k_blocks * config.block_k;
        }
        if (get_gemm_refinement_cost(d, dtype, resident) <= get_gemm_refinement_cost(d, dtype, config)) config = resident;
    }
    // Dual decode shares UB->L1 bandwidth but halves the serialized LUT/NZ work.
    if (dtype.dequant_b && config.direct_store) {
        auto dual = config;
        dual.dual_aiv_dequant = true;
        if (get_gemm_refinement_cost(d, dtype, dual) < get_gemm_refinement_cost(d, dtype, config)) config = dual;
    }

    return config;
}

static int64_t get_gemm_k_cost(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c) {
    const auto hw = get_hw_desc();
    const int64_t num_waves = ceil_div(int64_t(ceil_div(d.m, c.block_m)) * ceil_div(d.n, c.block_n), d.num_cores);
    const int64_t mad_cycles = ceil_div(2 * int64_t(c.mad_m) * c.mad_n * c.mad_k, hw.get_mad_flops_per_cycle(dtype.elem_bits));
    const int64_t load_cycles = ceil_div(int64_t(c.mad_m + c.mad_n) * c.mad_k * dtype.elem_bits, 8 * hw.l1_to_l0_bytes_per_cycle)
        + (dtype.has_sf ? ceil_div(int64_t(c.mad_m + c.mad_n) * c.mad_k * sizeof(int16_t), hw.sf_elements_per_pair * hw.l1_to_l0_sf_bytes_per_cycle) : 0);
    const int64_t num_mads = int64_t(c.block_m / c.mad_m) * (c.block_n / c.mad_n);
    // Effective L0/SF issue costs from paired BF16 and native FP8/FP4 measurements.
    const int issue_cycles = dtype.elem_bits == 16 ? 64 : dtype.elem_bits == 4 ? 40 : 116;
    const int64_t issue_ns = ceil_div(num_waves * num_mads * ceil_div(d.k, c.mad_k) * issue_cycles * 1000, hw.aic_clock_mhz);
    const int64_t num_turnovers = d.k / c.block_k * std::max<int64_t>(0, int64_t(c.block_k / c.mad_k) - c.num_l0_stages)
        + std::max<int64_t>(0, int64_t(ceil_div(d.k % c.block_k, c.mad_k)) - c.num_l0_stages);
    const int64_t turnover_ns = ceil_div(num_waves * num_mads * num_turnovers * std::max(mad_cycles, load_cycles) * 1000, hw.aic_clock_mhz);
    return get_gemm_refinement_cost(d, dtype, c) + issue_ns + turnover_ns;
}

static GemmConfig select_gemm_config(const GemmDesc& input_desc) {
    // TODO: add a memorization cache for the selected config

    auto d = input_desc;
    const auto hw = get_hw_desc();
    const auto dtype = get_gemm_dtype(d);
    int grouped_block_m = Runtime::get_mk_alignment_for_contiguous_layout();
    if (is_m_grouped(d.gemm_type) && dtype.has_sf && d.expected_m > 0 &&
        dynamic_cast<const IdentityEpilogue*>(d.epilogue_class.get()) &&
        (d.cd_dtype == at::kBFloat16 || d.cd_dtype == at::kFloat)) {
        while (grouped_block_m > FRAC_MN && d.expected_m <= grouped_block_m / 2)
            grouped_block_m /= 2;
        // Model the scheduled rows, excluding padding skipped by the grouped scheduler.
        if (grouped_block_m < Runtime::get_mk_alignment_for_contiguous_layout())
            d.m = d.num_groups * align(d.expected_m, grouped_block_m);
    }
    const int k = is_k_grouped(d.gemm_type) ? d.num_groups * d.expected_k : d.k;
    const int64_t compute_k = is_k_grouped(d.gemm_type) ? d.expected_k : d.k;
    const int num_batches = is_batched(d.gemm_type) || is_k_grouped(d.gemm_type) ? d.num_groups : 1;
    const int b_elem_bits = dtype.dequant_b ? dtype.elem_bits / 2 : dtype.elem_bits;
    // Logical operand bytes without sector amplification or SF.
    const int64_t a_matrix_bytes = int64_t(d.m) * compute_k * dtype.elem_bits / 8;
    const int64_t b_matrix_bytes = int64_t(d.n) * compute_k * b_elem_bits / 8;
    const int block_cap = dtype.elem_bits == 4 || dtype.dequant_b ? 256 : 512;
    const int m_alignment = is_mn_major(d.major_a) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();
    const bool with_fp8_cast = d.cd_dtype == at::kFloat8_e4m3fn;
    const int n_alignment = with_fp8_cast ? hw.vector_reg_bytes / sizeof(uint16_t) : is_mn_major(d.major_b) ? hw.get_mad_alignment_k(dtype.elem_bits) : hw.get_mad_alignment_mn();
    const int m_limit = align(d.m, m_alignment);
    std::pair<int, int> m_range{m_alignment, std::min(block_cap, m_limit)}, n_range{n_alignment, block_cap};
    if (is_m_grouped(d.gemm_type)) {
        const int block_m = grouped_block_m;
        m_range = {block_m, block_m};
        int block_n = std::min<int>(L0CSizeBytes / (sizeof(float) * block_m), align(d.n, n_alignment));
        if (dtype.dequant_b && is_mn_major(d.major_b)) block_n = block_cap;
        // Small M alignments need narrower N candidates to fit the L1 pipeline.
        n_range = {std::min(block_n, block_cap), block_n};
    }

    // Determine block k
    // 1. good to be multiple of l1-to-l0 transfer size, to avoid partial L1->L0 transfers
    const int min_block_k = hw.l1_to_l0_bytes_per_cycle * 8 / dtype.elem_bits;
    const int max_block_k = 2 * min_block_k;

    GemmConfig config{};
    int64_t best_cost = std::numeric_limits<int64_t>::max();
    // Large blocks establish a useful cost bound early; equal costs still prefer smaller M/N.
    for (int64_t block_m = m_range.second; block_m >= m_range.first; block_m -= m_alignment) {
        const int n_limit = std::min<int>(n_range.second, L0CSizeBytes / (sizeof(float) * block_m));
        for (int64_t block_n = align(n_limit, n_alignment); block_n >= n_range.first; block_n -= n_alignment) {
            const int num_n_blocks = ceil_div(d.n, block_n);
            const int num_m_blocks = ceil_div(d.m, block_m);
            const int64_t num_waves = ceil_div(int64_t(num_m_blocks) * num_n_blocks * num_batches, d.num_cores);

            const int64_t avg_block_bytes = a_matrix_bytes / num_m_blocks + b_matrix_bytes / num_n_blocks;
            // min_memory_time_ns = num_waves * avg_block_bytes / (l2_bw_gbps / hw.num_ai_cores)
            const int64_t min_memory_time_ns = num_waves * hw.num_ai_cores * avg_block_bytes / hw.l2_bw_gbps;
            if (min_memory_time_ns > best_cost) continue;

            // Initial operand loads precede MADs, even when all later loads overlap compute.
            const auto compute_cost = get_gemm_compute_cost(d, dtype, block_m, block_n);
            const int64_t min_first_load_time_ns = min_memory_time_ns * std::min<int64_t>(compute_k, min_block_k) / compute_k;
            if (compute_cost.time_ns + min_first_load_time_ns > best_cost) continue;

            auto fitted = make_gemm_candidate(d, dtype, block_m, block_n, min_block_k);
            if (!fitted) continue;
            // Fewer K iterations are useful while three L1 stages still fit.
            // We do not check the cost of the wider config, because the cost model doesn't consider the issue latency and scalar overhead
            // TODO: rewrite this adhoc code to properly consider issue latency and scalar overhead in the cost model
            if (k % max_block_k == 0) {
                auto wider = make_gemm_candidate(d, dtype, block_m, block_n, max_block_k);
                if (wider && wider->num_l1_stages >= 3) fitted = wider;
            }
            const auto cost = get_gemm_cost(d, dtype, *fitted, compute_cost, best_cost);
            if (cost > best_cost) continue;
            best_cost = cost;
            config = *fitted;
        }
    }
    DJ_HOST_ASSERT(config.block_m, "No legal GEMM configuration for {}", d);

    // Select L2 store hint
    // TODO: smarter heuristics for L2 store hints
    const int num_m_blocks = ceil_div(d.m, config.block_m);
    const int num_n_blocks = ceil_div(d.n, config.block_n);
    config.l2_ctrl_store_cd = asc_store_l2_cache_mode::NOTALLOC_CLEAN;
    const int num_waves = ceil_div(num_m_blocks * num_n_blocks, d.num_cores);
    if (d.acc && num_waves >= 4 && ceil_div(d.k, config.block_k) >= 16)
        config.l2_ctrl_store_cd = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM;

    config = refine_gemm_pipeline(d, dtype, config);
    // Keep M/N fixed and compare legal K tiles after fitting their pipeline and residency.
    // Streaming dequant is limited by repeated decoder startup; refine only a complete L1 working set.
    if (d.gemm_type == GemmType::Normal && get_compiled_dim(d.k, 'k', d.compiled_dims) &&
        d.k % (dtype.has_sf ? MX_SF_DIVISOR : FRAC_MN) == 0 && (!dtype.dequant_b || d.k <= config.num_l1_stages * config.block_k)) {
        const auto primary = config;
        auto best_k_cost = get_gemm_k_cost(d, dtype, config);
        const int k_limit = std::min<int64_t>(d.k, L1SizeBytes * 8 / ((primary.block_m + primary.block_n) * dtype.elem_bits));
        for (uint32_t block_k = min_block_k; block_k <= static_cast<uint32_t>(k_limit); block_k *= 2) {
            if (block_k == primary.block_k || (d.k <= primary.block_k && block_k < d.k)) continue;
            auto candidate = make_gemm_candidate(d, dtype, primary.block_m, primary.block_n, block_k);
            if (!candidate || candidate->mad_m != primary.mad_m || candidate->mad_n != primary.mad_n) continue;
            // Preserve the SF pipeline depth while comparing wider operand transfers.
            if (dtype.has_sf && candidate->num_l1_stages < primary.num_l1_stages) continue;
            // Changing MAD_K only helps issue-bound tiles; keep compute-bound MADs unchanged.
            const int64_t mad_cycles = ceil_div(2 * int64_t(primary.mad_m) * primary.mad_n * primary.mad_k, hw.get_mad_flops_per_cycle(dtype.elem_bits));
            const int64_t load_cycles = ceil_div(int64_t(std::min(primary.mad_m, primary.mad_n)) * primary.mad_k * dtype.elem_bits, 8 * hw.l1_to_l0_bytes_per_cycle);
            if (candidate->mad_k != primary.mad_k && mad_cycles > load_cycles) continue;
            candidate->l2_ctrl_store_cd = primary.l2_ctrl_store_cd;
            *candidate = refine_gemm_pipeline(d, dtype, *candidate);
            const auto cost = get_gemm_k_cost(d, dtype, *candidate);
            if (cost >= best_k_cost) continue;
            best_k_cost = cost;
            config = *candidate;
        }
    }

    // Double-buffer output-bound bias tiles without shortening contiguous operand rows.
    if (dynamic_cast<const BiasEpilogue*>(d.epilogue_class.get()) && config.num_epilogue_stages == 1 &&
        get_gemm_output_cost(d, dtype, config) > get_gemm_cost(d, dtype, config)) {
        const auto primary = config;
        for (auto [block_m, block_n]: {std::pair{primary.block_m - m_alignment, primary.block_n}, std::pair{primary.block_m, primary.block_n - std::max<int>(n_alignment, hw.vector_reg_bytes / sizeof(float))}}) {
            if (block_m < m_range.first || block_n < n_range.first ||
                (block_m != primary.block_m && is_mn_major(d.major_a)) ||
                (block_n != primary.block_n && is_mn_major(d.major_b))) continue;
            auto candidate = make_gemm_candidate(d, dtype, block_m, block_n, primary.block_k);
            if (!candidate || candidate->num_epilogue_stages < 2 || candidate->mad_k != primary.mad_k ||
                candidate->num_l1_stages < primary.num_l1_stages) continue;
            candidate->l2_ctrl_store_cd = primary.l2_ctrl_store_cd;
            *candidate = refine_gemm_pipeline(d, dtype, *candidate);
            if (get_gemm_k_cost(d, dtype, *candidate) < get_gemm_k_cost(d, dtype, config)) config = *candidate;
        }
    }

    // Prefer native FP4 full MADs without reducing operand or epilogue buffers.
    const uint64_t full_epilogue_bytes = uint64_t(config.block_m) * config.block_n * dtype.epilogue_elem_bits / 8 +
                                         (dynamic_cast<const BiasEpilogue*>(d.epilogue_class.get()) ? config.block_n * c10::elementSize(d.cd_dtype) : 0) +
                                         (with_fp8_cast ? uint64_t(config.block_m) * config.block_n / 32 : 0);
    if (dtype.elem_bits == 4 && uint64_t(config.block_m) * config.mad_k * config.num_l0_stages * dtype.elem_bits <= L0ASizeBytes * 8 &&
        uint64_t(config.block_n) * config.mad_k * config.num_l0_stages * dtype.elem_bits <= L0BSizeBytes * 8 &&
        full_epilogue_bytes * config.num_epilogue_stages <= UBSizeBytes) {
        auto full = config;
        full.mad_m = config.block_m;
        full.mad_n = config.block_n;
        if (get_gemm_k_cost(d, dtype, full) <= get_gemm_k_cost(d, dtype, config)) config = full;
    }

    // Admit multiwave dual decoding after final tiling.
    if (dtype.dequant_b && config.direct_store && !config.dual_aiv_dequant && d.gemm_type == GemmType::Normal &&
        num_m_blocks == 1 && num_waves > 1 && !d.acc && config.block_m == config.mad_m &&
        2 * config.block_m * hw.fp4_decode_bytes_per_cycle <= hw.get_mad_flops_per_cycle(dtype.elem_bits) &&
        d.n * dtype.epilogue_elem_bits / 8 % hw.l2_sector_bytes == 0 &&
        config.block_n * dtype.epilogue_elem_bits / 8 % hw.l2_sector_bytes == 0 &&
        2 * int64_t(config.block_m) * config.block_n * sizeof(float) <= L0CSizeBytes &&
        is_k_major(d.major_a) && is_k_major(d.major_b) && get_compiled_dim(d.n, 'n', d.compiled_dims) &&
        get_compiled_dim(d.k, 'k', d.compiled_dims) > config.block_k && d.k % config.block_k == 0 &&
        ceil_div(d.n, config.block_n * d.num_cores) * (d.k / config.block_k) > config.num_l1_stages) {
        config.dual_aiv_dequant = true;
    }

    // Keep shared B sectors in L2 for native FP4 tiles spanning a single K block.
    const int64_t resident_row_bytes = ceil_div(int64_t(d.k) * dtype.elem_bits, 8) + ceil_div(d.k, hw.sf_elements_per_pair) * sizeof(int16_t);
    if (dtype.elem_bits == 4 && d.gemm_type == GemmType::Normal && is_mn_major(d.major_a) && is_mn_major(d.major_b) &&
        config.block_n * dtype.elem_bits == hw.fractal_row_bytes * 8 && d.k <= config.block_k &&
        config.l2_ctrl_b == asc_load_l2_cache_mode::NOTALLOC_KEEP && (int64_t(d.m) + d.n) * resident_row_bytes <= hw.l2_size_bytes) {
        auto cached = config;
        cached.l2_ctrl_b = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
        if (get_gemm_k_cost(d, dtype, cached) <= get_gemm_k_cost(d, dtype, config)) config = cached;
    }

    // A small group consumes each K-major weight tile once; do not allocate it in L2.
    if (is_m_grouped(d.gemm_type) && config.block_m < Runtime::get_mk_alignment_for_contiguous_layout() &&
        d.expected_m > 0 && d.expected_m <= config.block_m && is_k_major(d.major_b)) {
        config.l2_ctrl_b = asc_load_l2_cache_mode::NOTALLOC_KEEP;
        // Direct output frees both AIVs to decode FP4 weights. AIV1 still clears group padding.
        const auto stride_d = d.outer_stride_d ? d.outer_stride_d : d.n;
        if (dtype.dequant_b && config.block_m == FRAC_MN && config.block_m == config.mad_m &&
            is_k_major(d.major_a) && !d.acc && get_compiled_dim(d.n, 'n', d.compiled_dims) && d.n % FRAC_MN == 0 &&
            reinterpret_cast<uintptr_t>(d.d_ptr) % hw.ub_bank_bytes == 0 &&
            stride_d * c10::elementSize(d.cd_dtype) % hw.ub_bank_bytes == 0) {
            config.direct_store = true;
            config.dual_aiv_dequant = true;
            // Narrow streaming tiles increase the number of UB decode buffers per AIV.
            if (config.block_n == 512 && config.block_k == 256 && d.k > config.num_l1_stages * config.block_k) {
                auto narrow = make_gemm_candidate(d, dtype, config.block_m, 256, config.block_k);
                if (narrow && narrow->num_l1_stages >= config.num_l1_stages) {
                    narrow->l2_ctrl_b = config.l2_ctrl_b;
                    narrow->l2_ctrl_store_cd = config.l2_ctrl_store_cd;
                    narrow->direct_store = true;
                    narrow->dual_aiv_dequant = true;
                    config = *narrow;
                }
            }
        }
    }

    if (deep_jit::get_env<int>("DG_JIT_DEBUG") || deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        static std::unordered_set<std::string> printed;
        const auto line = std::format("{} {}", d, config);
        if (printed.insert(line).second) std::puts(line.c_str());
    }
    return config;
}

} // namespace deep_gemm
