#pragma once

#include <deep_gemm/ascend.hpp>
#include <hcomm/hcomm.h>
#include <utils/debug/asc_assert.h>

namespace deep_gemm::urma {

// CANN 9.2 UBC_CTP descriptors, matching the validated DeepEP WRITE/NOP path
constexpr uint32_t kNumWQEBBBytes = 64;
constexpr uint64_t kWriteHeader = 0x0000030030010000ULL;
constexpr uint64_t kNopHeader = 0x0000110000000000ULL;

struct Peer {
    uint64_t tp_token, eid[2], token_value, base, size;
};

template <uint32_t kNumRanks>
struct State {
    uint64_t sq, sq_head, sq_doorbell, sq_tail;
    uint64_t cq, cq_tail, cq_doorbell;
    uint32_t sq_depth, cq_depth, cqe_bytes;
    uint32_t head, committed_head, start_head, completions;
    uint32_t completed;
    Peer peers[kNumRanks];
};

template <uint32_t kNumRanks, uint32_t kNumWQEStages, uint32_t kNumWQEPerStage, const auto& kWQEDone>
struct Jetty {
    static_assert(kNumWQEStages > 0 and kNumWQEPerStage > 0);
    static_assert(kWQEDone.base + kNumWQEStages <= kNumPipeEvents);

    __ubuf__ State<kNumRanks>* state;
    __ubuf__ uint64_t* wqe_stages;
    uint32_t rank_idx;
    uint32_t stage_idx = 0, stage_head = 0;
    uint32_t num_written_wqes = 0;
    bool has_nop = false;

    // Prime stage credits without touching UB; initialize contents separately.
    __aicore__ Jetty(__ubuf__ State<kNumRanks>* state, __ubuf__ uint64_t* wqe_stages, uint32_t rank_idx):
        state(state), wqe_stages(wqe_stages), rank_idx(rank_idx) {
        if constexpr (kNumRanks > 1)
            notify<PIPE_MTE3, PIPE_S, kNumWQEStages>(kWQEDone);
    }

    __aicore__ Jetty(__ubuf__ State<kNumRanks>* state, __ubuf__ uint64_t* wqe_stages, __gm__ void* channels,
                     uint32_t rank_idx, uint32_t core_idx): Jetty(state, wqe_stages, rank_idx) {
        init(channels, core_idx);
    }

    // Each caller-owned stage has kNumWQEPerStage slots, with NOP in the last slot.
    __aicore__ void init(__gm__ void* channels, uint32_t core_idx) {
        if constexpr (kNumRanks == 1) {
            state->head = state->committed_head = state->start_head = 0;
            state->completions = state->completed = 0;
            return;
        }
        // Initialize each reusable NOP template once.
        for (uint32_t stage_idx = 0; stage_idx < kNumWQEStages; ++ stage_idx) {
            const auto nop = wqe_stages + (static_cast<uint64_t>(stage_idx + 1) * kNumWQEPerStage - 1) * kNumWQEBBBytes / sizeof(uint64_t);
            nop[0] = kNopHeader | (5ULL << 16) | (1ULL << 19) | (1ULL << 21);
            #pragma unroll
            for (uint32_t word_idx = 1; word_idx < kNumWQEBBBytes / sizeof(uint64_t); ++ word_idx)
                nop[word_idx] = 0;
        }
        assert(channels != nullptr);
        const auto table = static_cast<__gm__ uint64_t*>(channels);
        const auto channel = reinterpret_cast<__gm__ AscendC::ChannelEntity*>(asc_load_dev(table + core_idx));
        assert(channel->sqNum > 0 and channel->cqNum > 0);
        const auto sq = channel->sqContextAddr[AscendC::HCOMM_URMA_DEFAULT_QP_IDX].contextInfo.ubJfs;
        const auto cq = channel->cqContextAddr[AscendC::HCOMM_URMA_DEFAULT_QP_IDX].contextInfo.ubJfc;
        state->sq = sq.sqVa;
        state->sq_head = sq.headAddr;
        state->sq_doorbell = sq.dbVa;
        state->sq_tail = sq.tailAddr;
        state->cq = cq.scqVa;
        state->cq_tail = cq.tailAddr;
        state->cq_doorbell = cq.dbVa;
        state->sq_depth = sq.sqDepth;
        state->cq_depth = cq.cqDepth;
        state->cqe_bytes = cq.cqeSize;
        assert(sq.wqeSize == kNumWQEBBBytes);
        assert(state->sq_depth > 4 and (state->sq_depth & (state->sq_depth - 1)) == 0);
        assert(state->cq_depth > 0 and (state->cq_depth & (state->cq_depth - 1)) == 0);
        const auto packed_head = asc_load_dev(reinterpret_cast<__gm__ uint64_t*>(state->sq_head));
        state->head = state->committed_head = state->start_head = static_cast<uint32_t>(packed_head);
        state->completions = static_cast<uint32_t>(packed_head >> 32);
        state->completed = asc_load_dev(reinterpret_cast<__gm__ uint32_t*>(state->cq_tail));

        for (uint32_t peer_idx = 0; peer_idx < kNumRanks; ++ peer_idx) {
            if (peer_idx == rank_idx)
                continue;
            const uint32_t peer_slot = peer_idx < rank_idx ? peer_idx : peer_idx - 1;
            const auto peer_channel = reinterpret_cast<__gm__ AscendC::ChannelEntity*>(asc_load_dev(table + peer_slot));
            assert(peer_channel->remoteBufferNum == 1);
            const auto remote = peer_channel->remoteBufferAddr[0].bufferInfo.rma;
            const auto context = peer_channel->sqContextAddr[AscendC::HCOMM_URMA_DEFAULT_QP_IDX].contextInfo.ubJfs;
            auto& peer = state->peers[peer_idx];
            peer.tp_token = static_cast<uint64_t>(context.tpID) | (1ULL << 24) |
                            (static_cast<uint64_t>(remote.protectionInfo.memInfo.ub.tokenId) << 32);
            peer.token_value = remote.protectionInfo.memInfo.ub.tokenValue;
            peer.base = remote.addr;
            peer.size = remote.size;
            peer.eid[0] = peer.eid[1] = 0;
            for (uint32_t byte_idx = 0; byte_idx < 8; ++ byte_idx) {
                peer.eid[0] |= static_cast<uint64_t>(context.remoteEID[byte_idx]) << (8 * byte_idx);
                peer.eid[1] |= static_cast<uint64_t>(context.remoteEID[byte_idx + 8]) << (8 * byte_idx);
            }
        }
    }

    // Acquire the stage before filling WRITEs from its first slot.
    __aicore__ void begin_stage(uint32_t head) {
        if constexpr (kNumRanks > 1)
            wait<PIPE_MTE3, PIPE_S>(kWQEDone, stage_idx);
        stage_head = head;
        num_written_wqes = 0;
        has_nop = false;
    }

    __aicore__ uint32_t write_at(uint32_t head, uint32_t peer_idx, uint64_t dst_offset, __gm__ const void* src, uint32_t num_bytes) {
        assert(peer_idx < kNumRanks and peer_idx != rank_idx);
        assert(head - state->start_head < state->sq_depth - 4);
        const auto& peer = state->peers[peer_idx];
        assert(dst_offset <= peer.size and num_bytes <= peer.size - dst_offset);
        const auto wqe = wqe_stages + (static_cast<uint64_t>(stage_idx) * kNumWQEPerStage + num_written_wqes) * kNumWQEBBBytes / sizeof(uint64_t);
        const uint32_t slot_idx = head & (state->sq_depth - 1);
        const uint64_t owner = static_cast<uint64_t>((head & state->sq_depth) == 0) << 31;
        wqe[0] = kWriteHeader | owner | slot_idx;
        // Words 1..7: token, EID[0..1], token value, destination, byte count, source
        wqe[1] = peer.tp_token;
        wqe[2] = peer.eid[0];
        wqe[3] = peer.eid[1];
        wqe[4] = peer.token_value;
        wqe[5] = peer.base + dst_offset;
        wqe[6] = num_bytes;
        wqe[7] = reinterpret_cast<uint64_t>(src);
        ++ num_written_wqes;
        return head + 1;
    }

    // The caller retires descriptor and payload stores before submitting this prefix.
    __aicore__ void commit() {
        if (kNumRanks == 1 or state->head == state->committed_head)
            return;
        asc_store_dev(reinterpret_cast<__gm__ uint64_t*>(state->sq_head),
                      static_cast<uint64_t>(state->head) | (static_cast<uint64_t>(state->completions) << 32));
        asc_sync_data_barrier(DSB_DDR);
        asc_store_dev(reinterpret_cast<__gm__ uint32_t*>(state->sq_doorbell), state->head);
        state->committed_head = state->head;
    }

    __aicore__ uint32_t append_checkpoint(uint32_t head) {
        if constexpr (kNumRanks == 1)
            return head;
        // One NOP fences prior WRITEs and requests completion.
        assert(head - state->start_head < state->sq_depth);
        const auto wqe = wqe_stages + (static_cast<uint64_t>(stage_idx + 1) * kNumWQEPerStage - 1) * kNumWQEBBBytes / sizeof(uint64_t);
        const uint64_t position_mask = (1ULL << 31) | (state->sq_depth - 1);
        const uint64_t owner = static_cast<uint64_t>((head & state->sq_depth) == 0) << 31;
        wqe[0] = (wqe[0] & ~position_mask) | owner | (head & (state->sq_depth - 1));
        has_nop = true;
        return head + 1;
    }

    // Copy WRITEs across at most one SQ wrap, then copy the optional NOP separately.
    __aicore__ void flush_stage() {
        const uint32_t num_wqes = num_written_wqes + has_nop;
        if (num_wqes == 0) {
            if constexpr (kNumRanks > 1)
                notify<PIPE_MTE3, PIPE_S>(kWQEDone, stage_idx);
            return;
        }
        assert(num_wqes <= state->sq_depth);
        const auto src = reinterpret_cast<__ubuf__ uint8_t*>(wqe_stages) +
                         static_cast<uint64_t>(stage_idx) * kNumWQEPerStage * kNumWQEBBBytes;
        asc_sync_data_barrier(DSB_UB);
        if (num_written_wqes != 0) {
            const uint32_t slot_idx = stage_head & (state->sq_depth - 1);
            const uint32_t num_first_wqes = min(num_written_wqes, state->sq_depth - slot_idx);
            const uint32_t num_first_bytes = num_first_wqes * kNumWQEBBBytes;
            asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(state->sq + static_cast<uint64_t>(slot_idx) * kNumWQEBBBytes),
                                src, 1, num_first_bytes, static_cast<asc_store_l2_cache_mode>(12), num_first_bytes, num_first_bytes);
            if (num_first_wqes < num_written_wqes) {
                const uint32_t num_remaining_bytes = (num_written_wqes - num_first_wqes) * kNumWQEBBBytes;
                asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(state->sq), src + num_first_bytes,
                                    1, num_remaining_bytes, static_cast<asc_store_l2_cache_mode>(12), num_remaining_bytes, num_remaining_bytes);
            }
        }
        if (has_nop) {
            const uint32_t slot_idx = (stage_head + num_written_wqes) & (state->sq_depth - 1);
            asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(state->sq + static_cast<uint64_t>(slot_idx) * kNumWQEBBBytes),
                                src + (kNumWQEPerStage - 1) * kNumWQEBBBytes, 1, kNumWQEBBBytes,
                                static_cast<asc_store_l2_cache_mode>(12), kNumWQEBBBytes, kNumWQEBBBytes);
        }
        notify<PIPE_MTE3, PIPE_S>(kWQEDone, stage_idx);
        stage_idx = (stage_idx + 1) % kNumWQEStages;
        num_written_wqes = 0;
        has_nop = false;
    }

    // Consume the NOP already prepared by pull; the caller commits this ready prefix
    __aicore__ void advance_checkpoint() {
        state->head += kNumRanks == 1 ? 0 : 1;
        ++ state->completions;
        if constexpr (kNumRanks == 1)
            state->completed = state->completions;
    }

    // Consume at most one completion so the caller can keep advancing other coroutines
    __aicore__ uint32_t poll() {
        if (state->completed == state->completions)
            return state->completed;
        const uint32_t tail = state->completed;
        const auto cqe = reinterpret_cast<__gm__ uint32_t*>(state->cq +
            static_cast<uint64_t>(tail & (state->cq_depth - 1)) * state->cqe_bytes);
        const uint32_t header = asc_load_dev(cqe);
        if (((header >> 2) & 1) == ((tail & state->cq_depth) != 0))
            return tail;
        assert((header >> 16) == 0);
        const auto sq_tail_ptr = reinterpret_cast<__gm__ uint32_t*>(state->sq_tail);
        const uint32_t sq_tail = asc_load_dev(sq_tail_ptr);
        // CQEs count checkpoints; recover the completed SQ WQEBB from the CQE
        const uint32_t completed_slot = asc_load_dev(cqe + 1) & (state->sq_depth - 1);
        const uint32_t completed_head = sq_tail + ((completed_slot - sq_tail) & (state->sq_depth - 1)) + 1;
        ++ state->completed;
        asc_store_dev(reinterpret_cast<__gm__ uint32_t*>(state->cq_tail), state->completed);
        asc_store_dev(reinterpret_cast<__gm__ uint32_t*>(state->cq_doorbell), state->completed & 0x00ffffffu);
        asc_store_dev(sq_tail_ptr, completed_head);
        return state->completed;
    }

};

} // namespace deep_gemm::urma
