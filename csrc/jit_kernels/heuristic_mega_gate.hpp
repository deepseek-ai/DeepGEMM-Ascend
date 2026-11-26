#pragma once

#include <deep_gemm/layout/mega_gate.hpp>

#include "heuristics.hpp"

namespace deep_gemm {

// Cost = AIC service (GEMM producer + split-partial store) pipelined against AIV service
// (per-column ranking + partial traffic) over the gate waves.
static double get_mega_gate_cost(const GemmDesc& d, const GemmConfig& c, uint32_t num_topk, uint32_t num_shared_experts,
                                 bool simple_routing, uint32_t num_fixed_rows) {
    const auto hw = get_hw_desc();
    const auto dtype = get_gemm_dtype(d);
    const uint32_t num_k_splits = d.num_groups;
    const uint32_t num_m_blocks = ceil_div(d.m, c.block_m), num_n_blocks = ceil_div(d.n, c.block_n);
    const uint32_t num_tasks_per_row = num_n_blocks * num_k_splits;
    const uint32_t num_waves = ceil_div(uint64_t(num_m_blocks) * num_tasks_per_row, uint64_t(d.num_cores));

    // AIC: the GEMM service plus the split-partial store.
    const int64_t gemm_ns = get_gemm_cost(d, dtype, c);
    const uint64_t partial_bytes = uint64_t(d.m) * d.n * num_k_splits * sizeof(float);
    const int64_t partial_write_ns = std::max(
        ceil_div(partial_bytes * 1000, uint64_t(hw.l0c_to_ub_bytes_per_cycle) * hw.num_ai_cores * hw.aic_clock_mhz),
        ceil_div(partial_bytes, uint64_t(hw.gm_bw_gbps)));
    const double aic_time_ns = double(gemm_ns + partial_write_ns);

    // AIV: per gate wave, one vector pass per packed column and batch, plus partial traffic.
    const uint32_t num_gate_waves = std::min(num_waves, num_m_blocks);
    const uint32_t num_local_tokens = ceil_div(d.m, 2 * d.num_cores * num_gate_waves);
    const uint32_t num_wave_tokens = std::min(d.m, ceil_div(d.num_cores, num_tasks_per_row) * c.block_m);
    const uint32_t num_rows = std::min(64u / num_k_splits, ceil_div(num_wave_tokens, 2 * d.num_cores));
    // Empirical column cost and fixed batch overhead in equivalent column passes.
    constexpr uint32_t kAivColumnCycles = 80, kAivBatchPasses = 10;
    const uint32_t num_physical_topk = num_topk + num_shared_experts;
    const uint32_t runtime_rows = std::min(num_rows, num_local_tokens);
    const uint32_t packed_rows = num_fixed_rows ? num_fixed_rows :
        mega_gate::layout::get_num_packed_rows(runtime_rows, num_physical_topk, simple_routing);
    const uint32_t packed_width = uint32_t(hw.vector_reg_bytes / sizeof(float)) / packed_rows;
    const uint32_t num_columns = ceil_div(d.n, packed_width);
    const uint32_t num_batches = ceil_div(num_local_tokens, packed_rows);
    const double aiv_cycles = double(num_batches) * kAivColumnCycles * (kAivBatchPasses + num_columns);
    // Approximate AIV service by the larger vector or L2 bandwidth bound.
    const double aiv_vector_ns = aiv_cycles * 1000 / hw.aic_clock_mhz;
    const double aiv_traffic_ns = double(num_local_tokens) * d.n * num_k_splits * sizeof(float) * 2 *
                                  hw.num_ai_cores / hw.l2_bw_gbps;
    const double aiv_time_ns = std::max(aiv_vector_ns, aiv_traffic_ns) * num_gate_waves;

    // AIC and AIV pipeline over the gate waves; the last minor wave is exposed.
    return (std::max(aic_time_ns, aiv_time_ns) + std::min(aic_time_ns, aiv_time_ns) / num_gate_waves) / 1000;
}

static std::pair<GemmConfig, uint32_t> get_mega_gate_config(uint32_t m, uint32_t n, uint32_t k, uint32_t num_topk,
                                                            uint32_t num_shared_experts, bool simple_routing, uint32_t num_fixed_rows) {
    GemmDesc d{};
    d.m = m;
    d.n = n;
    d.a_dtype = d.b_dtype = at::kBFloat16;
    d.d_dtype = d.cd_dtype = at::kFloat;
    d.major_a = d.major_b = Major::K;
    d.num_cores = runtime->get_num_sms();
    d.outer_stride_a = d.outer_stride_b = k;
    d.gemm_type = GemmType::Batched;
    d.epilogue_class = std::make_shared<IdentityEpilogue>();
    const auto hw = get_hw_desc();
    const auto dtype = get_gemm_dtype(d);
    const uint32_t min_block_k = hw.l1_to_l0_bytes_per_cycle / sizeof(uint16_t);
    // The fitter permits two MAD tiles along one MN axis, with two L0 operand buffers.
    const uint32_t m_limit = std::min<uint32_t>(2 * L0ASizeBytes / (min_block_k * sizeof(uint16_t)), align(m, FRAC_MN));
    const uint32_t n_limit = 2 * L0BSizeBytes / (min_block_k * sizeof(uint16_t));
    GemmConfig config{};
    uint32_t num_best_k_splits = 1;
    double best_cost = std::numeric_limits<double>::infinity();
    // Keep split K aligned to the base K block and at least four base blocks long.
    uint32_t max_splits = 1;
    while (2 * max_splits <= d.num_cores and k % (2 * max_splits * min_block_k) == 0 and
           k / (2 * max_splits) >= 4 * min_block_k)
        max_splits *= 2;
    // A batch-independent partition preserves the order of floating-point accumulation.
    if (num_fixed_rows) max_splits = std::min(max_splits, 4u);
    for (uint32_t num_k_splits = num_fixed_rows ? max_splits : 1; num_k_splits <= max_splits; num_k_splits *= 2) {
        d.k = k / num_k_splits;
        d.num_groups = num_k_splits;
        d.outer_stride_d = n * num_k_splits;
        for (uint32_t block_m = FRAC_MN; block_m <= m_limit; block_m += FRAC_MN) {
            for (uint32_t block_n = FRAC_MN; block_n <= n_limit; block_n += FRAC_MN) {
                for (uint32_t block_k : {min_block_k, 2 * min_block_k}) {
                    const auto candidate = make_gemm_candidate(d, dtype, block_m, block_n, block_k);
                    if (not candidate) continue;
                    const auto cost = get_mega_gate_cost(d, *candidate, num_topk, num_shared_experts, simple_routing, num_fixed_rows);
                    if (cost >= best_cost) continue;
                    best_cost = cost;
                    config = *candidate;
                    num_best_k_splits = num_k_splits;
                }
            }
        }
    }
    DJ_HOST_ASSERT(config.block_m);
    return {config, num_best_k_splits};
}

} // namespace deep_gemm
