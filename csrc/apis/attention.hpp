#pragma once

#include <c10/core/DeviceGuard.h>
#include <limits>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include <deep_jit/utils/exception.hpp>

#include "../jit_kernels/mqa_logits.hpp"
#include "../jit_kernels/paged_sparse_mqa_logits.hpp"
#include "../jit_kernels/sparse_mqa_logits.hpp"
#include "../utils/dtype.hpp"
#include "../utils/layout.hpp"
#include "../utils/math.hpp"

namespace deep_gemm::attention_api {

namespace py = pybind11;

static uint32_t get_logits_stride(uint32_t num_logits) {
    constexpr uint32_t kStrideAlignBytes = 512;
    const uint32_t align_elems = kStrideAlignBytes / c10::elementSize(torch::kBFloat16);
    return ceil_div(num_logits, align_elems) * align_elems;
}

static void validate_sparse_indices(const torch::Tensor& sparse_kv_block_indices,
                                    at::ScalarType qk_dtype, uint32_t sparse_block_kv) {
    DJ_HOST_ASSERT(runtime->get_dry_run() or sparse_kv_block_indices.device().type() == c10::DeviceType::PrivateUse1);
    DJ_HOST_ASSERT(sparse_kv_block_indices.dim() == 2 and sparse_kv_block_indices.scalar_type() == torch::kInt and sparse_kv_block_indices.is_contiguous());
    DJ_HOST_ASSERT(sparse_kv_block_indices.size(0) > 0 and sparse_kv_block_indices.size(0) <= std::numeric_limits<int32_t>::max());
    DJ_HOST_ASSERT(sparse_kv_block_indices.size(1) > 0 and static_cast<uint64_t>(sparse_kv_block_indices.size(1)) * sparse_block_kv < 65536);
    DJ_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == kPackedFP4);
    DJ_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
}

static void check_sparse_mqa_q_and_weights(const torch::Tensor& q, const std::optional<torch::Tensor>& q_sf,
                                           const torch::Tensor& weights, int64_t num_q_tokens, int64_t num_heads) {
    DJ_HOST_ASSERT(runtime->get_dry_run() or q.device().type() == c10::DeviceType::PrivateUse1);
    DJ_HOST_ASSERT(q.is_contiguous());
    DJ_HOST_ASSERT(q.scalar_type() == kPackedFP4 or q.scalar_type() == torch::kFloat8_e4m3fn);
    DJ_HOST_ASSERT(q.size(-1) == layout::sparse_mqa::kHeadDim / (q.scalar_type() == kPackedFP4 ? 2 : 1));
    DJ_HOST_ASSERT(weights.device() == q.device() and weights.scalar_type() == torch::kBFloat16 and
                   weights.dim() == 2 and weights.size(0) == num_q_tokens and weights.size(1) == num_heads and
                   weights.stride(1) == 1 and weights.stride(0) <= std::numeric_limits<uint32_t>::max());
    DJ_HOST_ASSERT(q_sf.has_value());
    const auto& sf_q = q_sf.value();
    const int64_t num_qh_rows = num_q_tokens * num_heads;
    DJ_HOST_ASSERT(sf_q.scalar_type() == torch::kShort and sf_q.device() == q.device());
    DJ_HOST_ASSERT(sf_q.numel() == num_qh_rows * 2 and sf_q.size(-1) == 2);
    DJ_HOST_ASSERT(sf_q.is_contiguous(), "sparse Q scales must be contiguous");
}

// Only the context-derived prefix is used; its block IDs must be strictly increasing and in range.
// Block i starts at i * sparse_block_kv + ks % sparse_block_kv. Padding may repeat.
// These device-data contracts are caller-owned and are not checked at runtime.
static torch::Tensor get_sparse_mqa_logits_metadata(
    const torch::Tensor& cu_seq_len_k_start, const torch::Tensor& cu_seq_len_k_end,
    uint32_t num_kv_tokens, const torch::Tensor& sparse_kv_block_indices,
    at::ScalarType qk_dtype, uint32_t sparse_block_kv) {
    using namespace layout::sparse_mqa_logits;
    validate_sparse_indices(sparse_kv_block_indices, qk_dtype, sparse_block_kv);
    const auto [num_q_tokens, num_max_sparse_blocks] = get_shape<2>(sparse_kv_block_indices);
    for (const auto& bounds : {cu_seq_len_k_start, cu_seq_len_k_end}) {
        DJ_HOST_ASSERT(bounds.dim() == 1 and bounds.numel() == num_q_tokens and bounds.is_contiguous());
        DJ_HOST_ASSERT(bounds.scalar_type() == torch::kInt and bounds.device() == sparse_kv_block_indices.device());
    }
    DJ_HOST_ASSERT(num_kv_tokens > 0 and num_kv_tokens <= std::numeric_limits<int32_t>::max());
    const SparseMQALayout layout(
        num_q_tokens, num_max_sparse_blocks, std::min<uint32_t>(runtime->get_num_sms(), num_q_tokens), sparse_block_kv);
    auto metadata = torch::empty({static_cast<int64_t>(layout.size_bytes / sizeof(uint32_t))}, sparse_kv_block_indices.options());
    const c10::OptionalDeviceGuard device_guard(runtime->get_dry_run() ? std::nullopt : std::make_optional(sparse_kv_block_indices.device()));
    launch_sparse_mqa_logits_metadata(
        layout, num_kv_tokens,
        sparse_kv_block_indices.data_ptr(), cu_seq_len_k_start.data_ptr(), cu_seq_len_k_end.data_ptr(),
        metadata.data_ptr());
    return metadata;
}

// Only the context-derived prefix is used; its logical block IDs must be strictly increasing and in range.
// Queries of a request must be consecutive and agree on the page mapping; referenced pages must exist.
// These device-data contracts are caller-owned and are not checked at runtime. Padding may repeat.
static torch::Tensor get_paged_sparse_mqa_logits_metadata(
    const torch::Tensor& context_lens, const torch::Tensor& block_table, const torch::Tensor& indices,
    uint32_t page_kv, const torch::Tensor& sparse_kv_block_indices,
    at::ScalarType qk_dtype, uint32_t sparse_block_kv) {
    using namespace layout::paged_sparse_mqa_logits;
    validate_sparse_indices(sparse_kv_block_indices, qk_dtype, sparse_block_kv);
    const auto [num_q_tokens, num_max_sparse_blocks] = get_shape<2>(sparse_kv_block_indices);
    const bool is_fp4 = qk_dtype == kPackedFP4;
    DJ_HOST_ASSERT(page_kv == 64 or page_kv == 128);
    DJ_HOST_ASSERT(num_max_sparse_blocks % (page_kv / sparse_block_kv) == 0);
    const uint32_t num_ai_cores = runtime->get_num_sms();
    DJ_HOST_ASSERT(context_lens.dim() == 1 and context_lens.numel() == num_q_tokens and
                   context_lens.scalar_type() == torch::kInt and context_lens.is_contiguous());
    DJ_HOST_ASSERT(indices.dim() == 1 and indices.numel() == num_q_tokens and
                   indices.scalar_type() == torch::kInt and indices.is_contiguous());
    DJ_HOST_ASSERT(block_table.dim() == 2 and block_table.size(0) == num_q_tokens and block_table.size(1) > 0);
    DJ_HOST_ASSERT(block_table.scalar_type() == torch::kInt and block_table.stride(1) == 1);
    DJ_HOST_ASSERT(block_table.stride(0) <= std::numeric_limits<uint32_t>::max());
    DJ_HOST_ASSERT(context_lens.device() == sparse_kv_block_indices.device() and
                   block_table.device() == sparse_kv_block_indices.device() and indices.device() == sparse_kv_block_indices.device());
    const c10::OptionalDeviceGuard device_guard(runtime->get_dry_run() ? std::nullopt : std::make_optional(sparse_kv_block_indices.device()));
    const PagedSparseMQALayout layout(num_q_tokens, num_max_sparse_blocks, sparse_block_kv, page_kv, num_ai_cores, is_fp4);
    const auto options = sparse_kv_block_indices.options().dtype(torch::kByte);
    auto metadata = torch::empty({static_cast<int64_t>(layout.size_bytes)}, options);
    const uint64_t range_costs_offset = align(num_q_tokens * sizeof(QBlockStats), kMetadataAlignmentBytes);
    auto scratch = torch::empty({static_cast<int64_t>(range_costs_offset + align(2 * num_ai_cores * sizeof(uint64_t), kMetadataAlignmentBytes))}, options);
    // Scratch is allocated and consumed on the current stream; only metadata survives prepare.
    launch_paged_sparse_mqa_metadata(layout, sparse_kv_block_indices.data_ptr(), context_lens.data_ptr(),
        indices.data_ptr(), block_table.data_ptr(), runtime->get_dry_run() ? nullptr : metadata.data_ptr<uint8_t>(), scratch.data_ptr(),
        runtime->get_dry_run() ? nullptr : scratch.data_ptr<uint8_t>() + range_costs_offset, block_table.stride(0));
    return metadata;
}


// NOTES: FIRST caches KV in metadata; REUSE reads this snapshot, not the current raw KV.
// Regenerate metadata after changing KV contents, even if the sparse indices are unchanged.
static torch::Tensor fp8_fp4_paged_sparse_mqa_logits(
    const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
    const torch::Tensor& raw_kv, const torch::Tensor& weights,
    const torch::Tensor& metadata, bool is_first, uint32_t num_max_sparse_blocks, uint32_t sparse_block_kv) {
    using namespace layout::paged_sparse_mqa_logits;
    const auto& q_data = q.first;
    DJ_HOST_ASSERT(q_data.dim() == 4 and q_data.size(1) == 1);
    const uint32_t num_q_tokens = q_data.size(0);
    const uint32_t num_heads = q_data.size(2);
    DJ_HOST_ASSERT(num_q_tokens > 0);
    const bool is_fp4 = q_data.scalar_type() == kPackedFP4;
    const uint32_t qk_data_bytes = is_fp4 ? kHeadDim / 2 : kHeadDim;
    const uint32_t token_bytes = qk_data_bytes + kSparseSFBytes;
    DJ_HOST_ASSERT(num_heads > 0 and num_heads <= kNumMaxHeads and num_heads % 4 == 0);
    DJ_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);

    check_sparse_mqa_q_and_weights(q_data, q.second, weights, num_q_tokens, num_heads);
    const auto& sf_q = q.second.value();

    // MQA pages store [page DATA | page SF], matching use_ascend_kv_layout=False.
    // The 4D shape describes page capacity, not interleaved token strides.
    DJ_HOST_ASSERT(raw_kv.device() == q_data.device() and raw_kv.dim() == 4 and
                   raw_kv.scalar_type() == torch::kByte);
    const uint32_t page_kv = raw_kv.size(1);
    DJ_HOST_ASSERT(page_kv == 64 or page_kv == 128);
    DJ_HOST_ASSERT(num_max_sparse_blocks % (page_kv / sparse_block_kv) == 0);
    DJ_HOST_ASSERT(raw_kv.size(0) > 0 and raw_kv.size(2) == 1 and
                   raw_kv.size(3) == token_bytes);
    DJ_HOST_ASSERT(raw_kv.stride(3) == 1 and raw_kv.stride(1) == token_bytes and
                   raw_kv.stride(0) >= static_cast<int64_t>(page_kv) * token_bytes and
                   raw_kv.stride(0) % kSparseSFBytes == 0);
    DJ_HOST_ASSERT(raw_kv.size(0) <= std::numeric_limits<uint32_t>::max() / page_kv);

    const PagedSparseMQALayout layout(
        num_q_tokens, num_max_sparse_blocks, sparse_block_kv, page_kv,
        static_cast<uint32_t>(runtime->get_num_sms()), is_fp4);
    DJ_HOST_ASSERT(metadata.device() == q_data.device() and metadata.scalar_type() == torch::kByte and
                   metadata.dim() == 1 and metadata.is_contiguous() and
                   metadata.numel() == static_cast<int64_t>(layout.size_bytes),
                   "metadata does not match the current paged sparse MQA configuration");
    const c10::OptionalDeviceGuard device_guard(runtime->get_dry_run() ? std::nullopt : std::make_optional(q_data.device()));
    const uint32_t num_logits = num_max_sparse_blocks * sparse_block_kv;
    const uint32_t logits_stride = get_logits_stride(num_logits);
    auto logits = torch::empty_strided({num_q_tokens, num_logits}, {logits_stride, 1}, q_data.options().dtype(torch::kBFloat16));
    // FIRST/REUSE is host-owned; the storage itself carries no device-visible phase.
    const auto launch = [&](bool first) {
        launch_paged_sparse_mqa_logits(layout, first, num_heads, weights.stride(0), logits_stride,
            raw_kv.stride(0), q_data.data_ptr(), sf_q.data_ptr(), raw_kv.data_ptr(), weights.data_ptr(),
            logits.data_ptr(), runtime->get_dry_run() ? nullptr : metadata.data_ptr<uint8_t>());
    };
    launch(is_first);
    if (runtime->get_dry_run())
        launch(!is_first);
    return logits;
}


// KV is token-major (DATA, SF), without packed tail padding.
// Block i starts at i * sparse_block_kv + ks % sparse_block_kv for each query.
static torch::Tensor fp8_fp4_sparse_mqa_logits(
    const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
    const std::pair<torch::Tensor, torch::Tensor>& kv, const torch::Tensor& weights,
    const torch::Tensor& metadata,
    uint32_t num_max_sparse_blocks, uint32_t sparse_block_kv) {
    using namespace layout::sparse_mqa_logits;
    const auto& q_data = q.first;
    DJ_HOST_ASSERT(q_data.dim() == 3);
    const bool is_fp4 = q_data.scalar_type() == kPackedFP4;
    const uint32_t data_bytes = is_fp4 ? kHeadDim / 2 : kHeadDim;
    const int64_t num_q_tokens = q_data.size(0);
    DJ_HOST_ASSERT(num_q_tokens > 0 and num_q_tokens <= std::numeric_limits<int32_t>::max());
    const auto num_heads = q_data.size(1);
    DJ_HOST_ASSERT(num_heads >= 4 and num_heads <= 32 and num_heads % 4 == 0);
    DJ_HOST_ASSERT(num_max_sparse_blocks > 0 and static_cast<uint64_t>(num_max_sparse_blocks) * sparse_block_kv < 65536);
    check_sparse_mqa_q_and_weights(q_data, q.second, weights, num_q_tokens, num_heads);
    const auto& sf_q = q.second.value();

    const auto& [kv_data, kv_sf] = kv;
    DJ_HOST_ASSERT(kv_data.device() == q_data.device() and kv_data.scalar_type() == q_data.scalar_type() and
                   kv_data.dim() == 2 and kv_data.is_contiguous() and kv_data.size(1) == data_bytes);
    DJ_HOST_ASSERT(kv_data.size(0) > 0 and kv_data.size(0) <= std::numeric_limits<int32_t>::max());
    DJ_HOST_ASSERT(kv_sf.device() == q_data.device() and kv_sf.scalar_type() == torch::kShort and
                   kv_sf.dim() == 2 and kv_sf.is_contiguous() and kv_sf.size(0) == kv_data.size(0) and
                   kv_sf.size(1) == kHeadDim / MX_SF_DIVISOR);
    const uint32_t num_kv_tokens = kv_data.size(0);

    // NOTES: The caller must match prepare's Q/KV sizes, sparse capacity, block size and active core count.
    // Only the allocation size is checked here; reading back the header would synchronize the stream.
    const SparseMQALayout layout(
        num_q_tokens, num_max_sparse_blocks, std::min<uint32_t>(runtime->get_num_sms(), num_q_tokens), sparse_block_kv);
    DJ_HOST_ASSERT(metadata.device() == q_data.device() and metadata.scalar_type() == torch::kInt and
                   metadata.dim() == 1 and metadata.is_contiguous() and
                   metadata.numel() == static_cast<int64_t>(layout.size_bytes / sizeof(uint32_t)));

    const uint32_t num_logits = num_max_sparse_blocks * sparse_block_kv;
    const uint32_t logits_stride = get_logits_stride(num_logits);
    auto logits = torch::empty_strided(
        {num_q_tokens, num_logits}, {logits_stride, 1}, q_data.options().dtype(torch::kBFloat16));
    const c10::OptionalDeviceGuard device_guard(runtime->get_dry_run() ? std::nullopt : std::make_optional(q_data.device()));
    launch_sparse_mqa_logits(
        num_q_tokens, num_heads, weights.stride(0), logits_stride,
        q_data.data_ptr(), kv_data.data_ptr(), kv_sf.data_ptr(), sf_q.data_ptr(),
        weights.data_ptr(), logits.data_ptr(), metadata.data_ptr(), num_kv_tokens,
        sparse_block_kv, is_fp4);
    return logits;
}

static torch::Tensor fp8_fp4_mqa_logits(
    const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
    const std::pair<torch::Tensor, torch::Tensor>& kv,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    uint32_t max_seqlen_k
) {
    const auto& q_data = q.first;
    const auto& sf_q = q.second;
    const auto& kv_data = kv.first;
    const auto& sf_kv = kv.second;

    DJ_HOST_ASSERT(q_data.dim() == 3);
    DJ_HOST_ASSERT(kv_data.dim() == 2);
    DJ_HOST_ASSERT(weights.dim() == 2);
    DJ_HOST_ASSERT(cu_seq_len_k_start.is_contiguous() and cu_seq_len_k_end.is_contiguous());
    DJ_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt and cu_seq_len_k_end.scalar_type() == torch::kInt);
    DJ_HOST_ASSERT(weights.scalar_type() == torch::kBFloat16, "Ascend MQA logits expects bf16 weights");

    const auto num_q_tokens = static_cast<uint32_t>(q_data.size(0));
    const auto num_heads = static_cast<uint32_t>(q_data.size(1));
    const uint32_t head_dim = q_data.scalar_type() == kPackedFP4 ? static_cast<uint32_t>(q_data.size(2)) * 2u : static_cast<uint32_t>(q_data.size(2));
    const auto num_kv_tokens = static_cast<uint32_t>(kv_data.size(0));
    const bool is_fp4 = q_data.scalar_type() == kPackedFP4;
    const uint32_t head_dim_storage = is_fp4 ? head_dim / 2u : head_dim;
    DJ_HOST_ASSERT(num_heads >= 4 and num_heads <= 64 and num_heads % 4 == 0);
    DJ_HOST_ASSERT(head_dim == 64 or head_dim == 128,
                   "Initial Ascend MQA logits kernel supports head_dim=64/128 only");
    DJ_HOST_ASSERT(max_seqlen_k > 0);
    const MQALogitsDesc desc = {
        .is_fp4 = is_fp4,
        .num_q_tokens = num_q_tokens,
        .num_kv_tokens = num_kv_tokens,
        .num_heads = num_heads,
        .head_dim = head_dim,
    };
    const MQALogitsConfig config = {
        .split_kv = 512u,
        .mad_m = 64u / num_heads * num_heads,
        .num_cd_stages = 2u,
        .num_q_stages = 2u,
        .num_kv_stages = 4u,
        .launch_options = {.num_blocks = runtime->get_num_sms()},
    };
    DJ_HOST_ASSERT(weights.size(0) == num_q_tokens and weights.size(1) == num_heads and weights.stride(1) == 1);
    DJ_HOST_ASSERT(cu_seq_len_k_start.numel() == num_q_tokens and cu_seq_len_k_end.numel() == num_q_tokens);
    DJ_HOST_ASSERT(q_data.is_contiguous() and kv_data.is_contiguous());
    DJ_HOST_ASSERT(kv_data.size(1) == head_dim_storage);

    DJ_HOST_ASSERT((q_data.scalar_type() == torch::kFloat8_e4m3fn and kv_data.scalar_type() == torch::kFloat8_e4m3fn) or
                   (q_data.scalar_type() == kPackedFP4 and kv_data.scalar_type() == kPackedFP4));
    DJ_HOST_ASSERT(sf_q.has_value(), "MQA logits requires Q scale factors");

    // TODO (important): refactor SF
    auto sf_q_view = sf_q.value();
    if (sf_q_view.dim() == 3) {
        DJ_HOST_ASSERT(sf_q_view.size(0) == num_q_tokens and sf_q_view.size(1) == num_heads);
        sf_q_view = sf_q_view.reshape({static_cast<int64_t>(num_q_tokens) * num_heads, sf_q_view.size(2)});
    }
    DJ_HOST_ASSERT(sf_q_view.scalar_type() == torch::kShort and sf_kv.scalar_type() == torch::kShort);
    DJ_HOST_ASSERT(sf_q_view.dim() == 2 and sf_q_view.size(0) == num_q_tokens * num_heads and
                   sf_q_view.size(1) == head_dim / 64);
    DJ_HOST_ASSERT(sf_kv.dim() == 2 and sf_kv.size(0) == num_kv_tokens and sf_kv.size(1) == head_dim / 64);
    auto sf_q_packed = sf_q_view;
    auto sf_kv_packed = sf_kv;
    if (sf_q_packed.stride(0) != 1 or sf_q_packed.stride(1) != static_cast<int64_t>(num_q_tokens) * num_heads)
        sf_q_packed = sf_q_packed.t().contiguous().t();
    if (sf_kv_packed.stride(0) != 1 or sf_kv_packed.stride(1) != static_cast<int64_t>(num_kv_tokens))
        sf_kv_packed = sf_kv_packed.t().contiguous().t();

    const uint32_t logits_stride = get_logits_stride(max_seqlen_k);
    auto logits = torch::empty_strided({num_q_tokens, max_seqlen_k}, {logits_stride, 1}, q_data.options().dtype(torch::kBFloat16));

    deep_gemm::launch_mqa_logits(desc, config, static_cast<uint32_t>(weights.stride(0)), logits_stride,
                                 q_data.data_ptr(), kv_data.data_ptr(), sf_q_packed.data_ptr(), sf_kv_packed.data_ptr(),
                                 weights.data_ptr(), logits.data_ptr(), cu_seq_len_k_start.data_ptr(),
                                 cu_seq_len_k_end.data_ptr());
    return logits;
}

static torch::Tensor get_paged_mqa_logits_metadata(
    const torch::Tensor& context_lens,
    uint32_t num_heads,
    const torch::Tensor& indices
) {
    DJ_HOST_ASSERT(context_lens.dim() == 2 and context_lens.size(1) == 1 and context_lens.is_contiguous() and
                   context_lens.scalar_type() == torch::kInt);
    const uint32_t num_q_tokens = static_cast<uint32_t>(context_lens.size(0));
    DJ_HOST_ASSERT(indices.dim() == 1 and indices.size(0) == num_q_tokens and
                   indices.scalar_type() == torch::kInt and indices.is_contiguous());

    DJ_HOST_ASSERT(num_heads >= 4 and num_heads <= 64 and num_heads % 4 == 0);
    // TODO: Revisit the fixed 512-token metadata split; it currently supports both N256 and N512 kernels.
    constexpr uint32_t SPLIT_KV = 512;
    const uint32_t num_cores = runtime->get_num_sms();
    auto metadata = torch::empty({num_cores + 1, 2}, context_lens.options());
    launch_paged_mqa_logits_metadata(num_q_tokens, SPLIT_KV, num_cores,
                                    context_lens.data_ptr(), indices.data_ptr(), metadata.data_ptr());
    return metadata;
}

static torch::Tensor fp8_fp4_paged_mqa_logits(
    const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const torch::Tensor& schedule_meta,
    uint32_t max_context_len,
    const torch::Tensor& indices,
    bool use_ascend_kv_layout
) {
    const auto& q_data = q.first;
    const auto& sf_q = q.second;

    DJ_HOST_ASSERT(q_data.dim() == 4 and fused_kv_cache.dim() == 4 and weights.dim() == 2);
    DJ_HOST_ASSERT(context_lens.dim() == 2 and context_lens.is_contiguous());
    DJ_HOST_ASSERT(block_table.dim() == 2 and block_table.stride(1) == 1);
    DJ_HOST_ASSERT(context_lens.scalar_type() == torch::kInt and block_table.scalar_type() == torch::kInt);
    DJ_HOST_ASSERT(schedule_meta.dim() == 2 and schedule_meta.is_contiguous() and
                   schedule_meta.scalar_type() == torch::kInt);
    DJ_HOST_ASSERT(weights.scalar_type() == torch::kBFloat16, "Ascend MQA logits expects bf16 weights");

    const uint32_t num_q_tokens = q_data.size(0);
    const uint32_t num_heads = q_data.size(2);
    const bool is_fp4 = q_data.scalar_type() == kPackedFP4;
    const uint32_t head_dim_storage = static_cast<uint32_t>(q_data.size(3));
    const uint32_t head_dim = is_fp4 ? head_dim_storage * 2u : head_dim_storage;
    const uint32_t page_size = fused_kv_cache.size(1);
    DJ_HOST_ASSERT(head_dim == 64 or head_dim == 128, "Ascend MQA logits kernel supports head_dim=64/128 only");
    DJ_HOST_ASSERT(page_size == 64 or page_size == 128, "Ascend paged MQA logits supports page_size=64/128");
    const uint32_t sf_pairs = head_dim / 64u;
    const uint32_t sf_bytes = sf_pairs * sizeof(int16_t);
    // NOTES: Shape is [num_pages, page_size, 1, head_dim_storage + sf_bytes].
    // Each physical page is [KV data | SF]; only stride(0) spans pages.
    DJ_HOST_ASSERT(fused_kv_cache.size(2) == 1 and fused_kv_cache.size(3) == head_dim_storage + sf_bytes and
                   fused_kv_cache.scalar_type() == torch::kByte);
    DJ_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_storage + sf_bytes and fused_kv_cache.stride(3) == 1 and
                   fused_kv_cache.stride(0) % sf_bytes == 0);
    const PagedMQALogitsDesc desc = {
        .is_fp4 = is_fp4,
        .num_q_tokens = num_q_tokens,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .page_size = page_size,
    };
    const PagedMQALogitsConfig config = select_paged_mqa_logits_config(desc);

    DJ_HOST_ASSERT(q_data.size(1) == 1);
    DJ_HOST_ASSERT(q_data.scalar_type() == torch::kFloat8_e4m3fn or q_data.scalar_type() == kPackedFP4);
    DJ_HOST_ASSERT(q_data.is_contiguous());
    DJ_HOST_ASSERT(sf_q.has_value() and sf_q.value().scalar_type() == torch::kShort);
    DJ_HOST_ASSERT(indices.dim() == 1 and indices.size(0) == num_q_tokens and
                   indices.scalar_type() == torch::kInt and indices.is_contiguous());
    DJ_HOST_ASSERT(weights.size(0) == num_q_tokens and weights.size(1) == num_heads and weights.stride(1) == 1);
    DJ_HOST_ASSERT(context_lens.size(0) == num_q_tokens and context_lens.size(1) == 1);
    DJ_HOST_ASSERT(block_table.size(0) == num_q_tokens);

    auto sf_q_view = sf_q.value();
    if (sf_q_view.dim() == 4)
        sf_q_view = sf_q_view.reshape({static_cast<int64_t>(num_q_tokens) * num_heads,
                                       sf_q_view.size(3)});
    DJ_HOST_ASSERT(sf_q_view.dim() == 2 and sf_q_view.size(0) == num_q_tokens * num_heads and
                   sf_q_view.size(1) == sf_pairs);
    if (sf_q_view.stride(0) != 1 or
        sf_q_view.stride(1) != static_cast<int64_t>(num_q_tokens) * num_heads)
        sf_q_view = sf_q_view.t().contiguous().t();

    const uint32_t num_cores = runtime->get_num_sms();
    DJ_HOST_ASSERT(schedule_meta.size(0) == static_cast<int64_t>(num_cores) + 1 and schedule_meta.size(1) == 2);
    const uint32_t logits_stride = get_logits_stride(max_context_len);
    auto logits = torch::empty_strided({num_q_tokens, max_context_len}, {logits_stride, 1}, q_data.options().dtype(torch::kBFloat16));
    deep_gemm::launch_paged_mqa_logits(desc, config, use_ascend_kv_layout,
                                       static_cast<uint32_t>(weights.stride(0)), logits_stride,
                                       static_cast<uint32_t>(block_table.stride(0)),
                                       static_cast<uint64_t>(fused_kv_cache.stride(0)), q_data.data_ptr(),
                                       sf_q_view.data_ptr(), fused_kv_cache.data_ptr(), weights.data_ptr(),
                                       logits.data_ptr(), context_lens.data_ptr(), block_table.data_ptr(),
                                       indices.data_ptr(), schedule_meta.data_ptr());
    return logits;
}


static void register_apis(pybind11::module_& m) {
    m.def("get_sparse_mqa_logits_metadata", get_sparse_mqa_logits_metadata, R"doc(
Prepare metadata for token-major sparse MQA logits.

For each query, let B = sparse_block_kv, capacity = sparse_kv_block_indices.size(1),
ks = cu_seq_len_k_start[q], and end = min(cu_seq_len_k_end[q], num_kv_tokens).
The bounds must be nonnegative. The first min(capacity, ceil(max(0, end - ks) / B))
indices must be unique and strictly increasing. Each block ID i must satisfy
ks <= i * B + ks % B < end. The remaining entries are unused padding and may repeat.

The caller must guarantee these data constraints; no device-value validation is
performed. Violations can produce incorrect results or out-of-bounds accesses.
The metadata must be consumed with the same Q/KV sizes, sparse capacity, block size
and active core count used to prepare it.
)doc",
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("num_kv_tokens"),
          py::arg("sparse_kv_block_indices"), py::arg("qk_dtype"), py::arg("sparse_block_kv"));
    // Internal paged bindings: deep_gemm Python wrappers own configuration and FIRST/REUSE state.
    m.def("_get_paged_sparse_mqa_logits_metadata", get_paged_sparse_mqa_logits_metadata,
          py::arg("context_lens"), py::arg("block_table"), py::arg("indices"), py::arg("page_kv"),
          py::arg("sparse_kv_block_indices"), py::arg("qk_dtype"), py::arg("sparse_block_kv"));
    m.def("_fp8_fp4_paged_sparse_mqa_logits", fp8_fp4_paged_sparse_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"), py::arg("metadata"), py::arg("is_first"),
          py::arg("num_max_sparse_blocks"), py::arg("sparse_block_kv"));
    m.def("fp8_fp4_sparse_mqa_logits", fp8_fp4_sparse_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"), py::arg("metadata"), py::arg("num_max_sparse_blocks"),
          py::arg("sparse_block_kv"));
    m.def("fp8_fp4_mqa_logits", fp8_fp4_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("max_seqlen_k"));
    m.def("get_paged_mqa_logits_metadata", get_paged_mqa_logits_metadata,
          py::arg("context_lens"), py::arg("num_heads"), py::arg("indices"));
    m.def("fp8_fp4_paged_mqa_logits", fp8_fp4_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"), py::arg("indices"),
          py::arg("use_ascend_kv_layout") = false);
}

} // namespace deep_gemm::attention_api
