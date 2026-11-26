#pragma once

#include <deep_jit/utils/exception.hpp>

#include "gemm_config.hpp"
#include "../jit/jit.hpp"
#include "../utils/math.hpp"

namespace deep_gemm {

struct MQALogitsDesc {
    bool is_fp4;
    uint32_t num_q_tokens;
    uint32_t num_kv_tokens;
    uint32_t num_heads;
    uint32_t head_dim;
};

struct MQALogitsConfig {
    uint32_t split_kv;
    uint32_t mad_m;
    uint32_t num_cd_stages;
    uint32_t num_q_stages;
    uint32_t num_kv_stages;
    deep_jit::ascend::LaunchOptions launch_options;
};

static void launch_mqa_logits(
    const MQALogitsDesc& desc, const MQALogitsConfig& config,
    const uint32_t& weights_stride, const uint32_t& logits_stride,
    void* q, void* kv,
    void* sf_q, void* sf_kv,
    void* weights, void* logits,
    void* cu_seq_len_k_start, void* cu_seq_len_k_end
) {
    const auto code = std::format(R"(
#include <deep_gemm/mqa_logits.hpp>

using namespace deep_gemm;
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&mqa_logits_impl<
        {},
        {}, {},
        {}, {}, {},
        {}, {},
        {}>);
}}
)",
        desc.is_fp4 ? "true" : "false",
        desc.num_heads,
        desc.head_dim,
        config.split_kv,
        config.mad_m,
        config.num_cd_stages,
        config.num_q_stages,
        config.num_kv_stages,
        config.launch_options.num_blocks.value());
    if (runtime->get_dry_run()) {
        jit->compile_without_load("mqa_logits", code);
        return;
    }
    const auto kernel = jit->compile("mqa_logits", code);

    const uint32_t qk_stride = desc.is_fp4 ? desc.head_dim / 2u : desc.head_dim;
    jit->launch(kernel, config.launch_options,
                desc.num_q_tokens, desc.num_kv_tokens,
                gm_ptr<uint8_t>{static_cast<uint64_t>(qk_stride), reinterpret_cast<uintptr_t>(q)},
                gm_ptr<uint8_t>{static_cast<uint64_t>(qk_stride), reinterpret_cast<uintptr_t>(kv)},
                sf_q, sf_kv,
                gm_ptr<uint8_t>{static_cast<uint64_t>(weights_stride), reinterpret_cast<uintptr_t>(weights)},
                gm_ptr<uint8_t>{static_cast<uint64_t>(logits_stride), reinterpret_cast<uintptr_t>(logits)},
                cu_seq_len_k_start, cu_seq_len_k_end);
}

struct PagedMQALogitsDesc {
    bool is_fp4;
    uint32_t num_q_tokens;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t page_size;
};

using PagedMQALogitsConfig = MQALogitsConfig;

static PagedMQALogitsConfig select_paged_mqa_logits_config(const PagedMQALogitsDesc& desc) {
    DJ_HOST_ASSERT(desc.num_heads >= 4 and desc.num_heads <= 64 and desc.num_heads % 4 == 0);
    const bool use_fp8_m128n256 = !desc.is_fp4 and desc.num_heads == 32 and desc.head_dim == 128;
    const uint32_t mad_m = 64u / desc.num_heads * desc.num_heads;
    return {
        .split_kv = use_fp8_m128n256 ? 256u : 512u,
        .mad_m = use_fp8_m128n256 ? 128u : mad_m,
        .num_cd_stages = 2u,
        .num_q_stages = 2u,
        .num_kv_stages = 4u,
        .launch_options = {.num_blocks = runtime->get_num_sms()},
    };
}

static void launch_paged_mqa_logits_metadata(
    const uint32_t& num_q_tokens,
    const uint32_t& split_kv,
    const uint32_t& num_cores,
    void* context_lens, void* indices,
    void* metadata
) {
    const auto code = std::format(R"(
#include <deep_gemm/paged_mqa_logits_metadata.hpp>

using namespace deep_gemm;
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&paged_mqa_logits_metadata_simt_impl<
        {}, {}>);
}}
)",
        split_kv,
        num_cores);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("paged_mqa_logits_metadata", code);
        return;
    }
    const auto kernel = jit->compile("paged_mqa_logits_metadata", code);

    constexpr uint32_t kMaxDynamicUBufSize = 248 * 1024;
    constexpr uint32_t kNumWarpSums = 8;
    const uint64_t request_arrays_bytes = 2ull * num_q_tokens * sizeof(uint32_t);
    const uint64_t metadata_offset_bytes = align(request_arrays_bytes + kNumWarpSums * sizeof(uint32_t), 32u);
    const uint64_t metadata_bytes = (num_cores + 1) * 2 * sizeof(uint32_t);
    const uint64_t ubuf_size = align(metadata_offset_bytes + metadata_bytes, 32u);
    DJ_HOST_ASSERT(ubuf_size <= kMaxDynamicUBufSize, "Paged MQA metadata exceeds UB capacity");
    jit->launch(kernel, {.num_blocks = 1, .num_ubuf_bytes = static_cast<int>(ubuf_size)},
                num_q_tokens, context_lens, indices, metadata);
}

static void launch_paged_mqa_logits(
    const PagedMQALogitsDesc& desc, const PagedMQALogitsConfig& config,
    const bool& use_ascend_kv_layout,
    const uint32_t& weights_stride, const uint32_t& logits_stride,
    const uint32_t& block_table_row_stride, const uint64_t& kv_page_stride_bytes,
    void* q, void* sf_q,
    void* fused_kv,
    void* weights, void* logits,
    void* context_lens, void* block_table,
    void* indices, void* metadata
) {
    const auto code = std::format(R"(
#include <deep_gemm/mqa_logits.hpp>

using namespace deep_gemm;
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&paged_mqa_logits_impl<
        {}, {}, {}, {},
        {}, {}, {},
        {}, {},
        {}>);
}}
)",
        desc.is_fp4 ? "true" : "false",
        desc.num_heads,
        desc.head_dim,
        desc.page_size,
        config.split_kv,
        config.mad_m,
        config.num_cd_stages,
        config.num_q_stages,
        config.num_kv_stages,
        use_ascend_kv_layout ? "true" : "false");
    if (runtime->get_dry_run()) {
        jit->compile_without_load("paged_mqa_logits", code);
        return;
    }
    const auto kernel = jit->compile("paged_mqa_logits", code);

    const uint32_t qk_stride = desc.is_fp4 ? desc.head_dim / 2u : desc.head_dim;
    jit->launch(kernel, config.launch_options,
                desc.num_q_tokens,
                block_table_row_stride,
                kv_page_stride_bytes,
                gm_ptr<uint8_t>{qk_stride, reinterpret_cast<uintptr_t>(q)},
                sf_q,
                gm_ptr<uint8_t>{qk_stride, reinterpret_cast<uintptr_t>(fused_kv)},
                gm_ptr<uint8_t>{weights_stride, reinterpret_cast<uintptr_t>(weights)},
                gm_ptr<uint8_t>{logits_stride, reinterpret_cast<uintptr_t>(logits)},
                context_lens, block_table, indices, metadata);
}

} // namespace deep_gemm
