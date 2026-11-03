#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <acl/acl.h>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res.h>
#include <hcomm/hcomm_res.h>

#include <deep_jit/backend/ascend/driver.hpp>

#define DG_HCCL_CHECK(cmd) \
do { \
    const auto result = (cmd); \
    DJ_HOST_ASSERT(result == HCCL_SUCCESS, "{} failed: {} ({})", \
                   #cmd, static_cast<int>(result), HcclGetErrorString(result)); \
} while (0)

namespace deep_gemm {

// Leases one process-lifetime HCCL-registered segment and resolves every peer's UBMEM VA.
class HCCLSymmetricBuffer {
public:
    void* buffer = nullptr;
    std::vector<uint64_t> peer_ptrs;

private:
    std::string group_name;
    HcclComm comm = nullptr;
    HcclMemHandle memory_handle = nullptr;
    uint32_t rank_idx;
    uint32_t num_ranks;
    uint64_t num_bytes;
    bool leased = false;

    struct Pool {
        std::mutex mutex;
        std::vector<HCCLSymmetricBuffer*> buffers;
    };

public:
    static std::shared_ptr<HCCLSymmetricBuffer> acquire(
        const std::string& group_name,
        uint32_t rank_idx,
        uint32_t num_ranks,
        uint64_t num_bytes) {
        DJ_HOST_ASSERT(num_bytes > 0);
        DJ_HOST_ASSERT(num_ranks > 0 and rank_idx < num_ranks);

        HcclComm comm = nullptr;
        DG_HCCL_CHECK(HcclCommGetHandleWithName(group_name.c_str(), &comm));
        auto& pool = get_pool();
        std::lock_guard lock(pool.mutex);
        HCCLSymmetricBuffer* selected = nullptr;
        for (auto candidate : pool.buffers) {
            if (candidate->leased or candidate->group_name != group_name or candidate->comm != comm or
                candidate->rank_idx != rank_idx or candidate->num_ranks != num_ranks or
                candidate->num_bytes < num_bytes)
                continue;
            if (selected == nullptr or candidate->num_bytes < selected->num_bytes)
                selected = candidate;
        }
        if (selected == nullptr) {
            selected = new HCCLSymmetricBuffer(group_name, comm, rank_idx, num_ranks, num_bytes);
            pool.buffers.push_back(selected);
        } else {
            DJ_ACL_CHECK(aclrtMemset(selected->buffer, num_bytes, 0, num_bytes));
        }
        selected->leased = true;
        return std::shared_ptr<HCCLSymmetricBuffer>(selected, [](HCCLSymmetricBuffer* symmetric_buffer) {
            auto& pool = HCCLSymmetricBuffer::get_pool();
            std::lock_guard lock(pool.mutex);
            DJ_HOST_ASSERT(symmetric_buffer->leased);
            symmetric_buffer->leased = false;
        });
    }

    HCCLSymmetricBuffer(const HCCLSymmetricBuffer&) = delete;
    HCCLSymmetricBuffer& operator=(const HCCLSymmetricBuffer&) = delete;

private:
    HCCLSymmetricBuffer(
        const std::string& group_name,
        HcclComm comm,
        uint32_t rank_idx,
        uint32_t num_ranks,
        uint64_t num_bytes):
        peer_ptrs(num_ranks, 0),
        group_name(group_name),
        comm(comm),
        rank_idx(rank_idx),
        num_ranks(num_ranks),
        num_bytes(num_bytes) {
        DJ_ACL_CHECK(aclrtMalloc(&buffer, num_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        DJ_ACL_CHECK(aclrtMemset(buffer, num_bytes, 0, num_bytes));

        CommMem memory{};
        memory.type = COMM_MEM_TYPE_DEVICE;
        memory.addr = buffer;
        memory.size = num_bytes;

        // NOTES: Ranks must match process-wide registration order, including subgroup allocations
        static std::atomic<uint32_t> next_tag_idx = 0;
        const auto memory_tag = "deep_gemm_mega_moe_" + std::to_string(next_tag_idx.fetch_add(1));
        DG_HCCL_CHECK(HcclCommMemReg(comm, memory_tag.c_str(), &memory, &memory_handle));

        peer_ptrs[rank_idx] = reinterpret_cast<uint64_t>(buffer);
        if (num_ranks == 1)
            return;

        uint32_t* layers = nullptr;
        uint32_t num_layers = 0;
        DG_HCCL_CHECK(HcclRankGraphGetLayers(comm, &layers, &num_layers));
        DJ_HOST_ASSERT(num_layers > 0, "UBMEM requires a network layer; got {} layers", num_layers);
        // Single-server CANN 9.2 topologies expose UB_MEM on the only layer.
        // make_peer_descriptor still verifies every peer has a UB_MEM link.
        const uint32_t layer = layers[num_layers > 1 ? 1 : 0];

        std::vector<HcclChannelDesc> descriptors;
        descriptors.reserve(num_ranks - 1);
        for (uint32_t peer_rank_idx = 0; peer_rank_idx < num_ranks; ++peer_rank_idx) {
            if (peer_rank_idx == rank_idx)
                continue;
            descriptors.push_back(make_peer_descriptor(
                comm, layer, rank_idx, peer_rank_idx, &memory_handle));
        }

        std::vector<ChannelHandle> channels(descriptors.size(), 0);
        DG_HCCL_CHECK(HcclChannelAcquire(
            comm,
            CommEngine::COMM_ENGINE_AIV,
            descriptors.data(),
            static_cast<uint32_t>(descriptors.size()),
            channels.data()));

        for (uint32_t idx = 0; idx < descriptors.size(); ++idx) {
            uint32_t num_remote_memories = 0;
            CommMem* remote_memories = nullptr;
            char** memory_tags = nullptr;
            DG_HCCL_CHECK(HcclChannelGetRemoteMems(
                comm,
                channels[idx],
                &num_remote_memories,
                &remote_memories,
                &memory_tags));
            DJ_HOST_ASSERT(num_remote_memories > 0 and remote_memories != nullptr and memory_tags != nullptr);
            bool found = false;
            for (uint32_t memory_idx = 0; memory_idx < num_remote_memories; ++memory_idx) {
                if (memory_tags[memory_idx] == nullptr or memory_tag != memory_tags[memory_idx])
                    continue;
                DJ_HOST_ASSERT(remote_memories[memory_idx].size == num_bytes);
                peer_ptrs[descriptors[idx].remoteRank] =
                    reinterpret_cast<uint64_t>(remote_memories[memory_idx].addr);
                found = true;
                break;
            }
            DJ_HOST_ASSERT(found);
        }
    }

    // HcclCommMemReg has no matching HCCL unregister API. Keep registered allocations alive
    // for the process lifetime and reuse them; freeing one leaves stale remote mappings.
    static Pool& get_pool() {
        static auto* pool = new Pool;
        return *pool;
    }

    static HcclChannelDesc make_peer_descriptor(
        HcclComm comm,
        uint32_t layer,
        uint32_t rank_idx,
        uint32_t peer_rank_idx,
        HcclMemHandle* memory_handle) {
        CommLink* links = nullptr;
        uint32_t num_links = 0;
        DG_HCCL_CHECK(HcclRankGraphGetLinks(
            comm, layer, rank_idx, peer_rank_idx, &links, &num_links));

        for (uint32_t link_idx = 0; link_idx < num_links; ++link_idx) {
            const auto& link = links[link_idx];
            if (link.linkAttr.linkProtocol != COMM_PROTOCOL_UB_MEM)
                continue;

            HcclChannelDesc descriptor;
            DG_HCCL_CHECK(HcclChannelDescInit(&descriptor, 1));
            descriptor.remoteRank = peer_rank_idx;
            descriptor.channelProtocol = link.linkAttr.linkProtocol;
            descriptor.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            descriptor.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            descriptor.localEndpoint.loc = link.srcEndpointDesc.loc;
            descriptor.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            descriptor.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            descriptor.remoteEndpoint.loc = link.dstEndpointDesc.loc;
            descriptor.notifyNum = 3;
            descriptor.memHandles = memory_handle;
            descriptor.memHandleNum = 1;
            return descriptor;
        }

        DJ_HOST_ASSERT(false, "No UBMEM link from rank {} to peer {} on layer {} ({} links)",
                       rank_idx, peer_rank_idx, layer, num_links);
        return {};
    }
};

} // namespace deep_gemm
