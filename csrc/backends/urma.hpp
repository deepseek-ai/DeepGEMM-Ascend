#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res.h>
#include <hcomm/hcomm_res.h>
#include <hcomm/hcomm_channel.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <deep_jit/backend/ascend/driver.hpp>

#define DG_HCOMM_CHECK(cmd) \
do { \
    const auto dg_hcomm_status = (cmd); \
    DJ_HOST_ASSERT(static_cast<int>(dg_hcomm_status) == 0, "{} failed: {}", #cmd, static_cast<int>(dg_hcomm_status)); \
} while (0)

namespace deep_gemm {

// UBC_CTP queues over a separate registration of the MegaMoE allocation
struct UrmaChannels {
    EndpointHandle endpoint = nullptr;
    HcommMemHandle memory = nullptr;
    std::vector<ChannelHandle> channels;
    std::vector<std::string> names;
    void* device_channels = nullptr;

    UrmaChannels(HcclComm comm, void* buffer, uint64_t num_bytes, uint32_t rank_idx,
                 uint32_t num_ranks, uint32_t num_jetties, const std::string& buffer_name) {
        if (num_ranks == 1)
            return;
        uint32_t* layers = nullptr;
        uint32_t num_layers = 0;
        DG_HCOMM_CHECK(HcclRankGraphGetLayers(comm, &layers, &num_layers));
        DJ_HOST_ASSERT(num_layers >= 2);
        const auto layer = layers[1];
        std::vector<CommLink> peers;
        for (uint32_t peer_idx = 0; peer_idx < num_ranks; ++ peer_idx) {
            if (peer_idx == rank_idx)
                continue;
            CommLink* links = nullptr;
            uint32_t num_links = 0;
            DG_HCOMM_CHECK(HcclRankGraphGetLinks(comm, layer, rank_idx, peer_idx, &links, &num_links));
            uint32_t link_idx = 0;
            while (link_idx < num_links and links[link_idx].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP)
                ++ link_idx;
            DJ_HOST_ASSERT(link_idx < num_links, "No UBC_CTP link to rank {}", peer_idx);
            peers.push_back(links[link_idx]);
        }
        const auto& local = peers.front().srcEndpointDesc;
        for (const auto& peer: peers) {
            DJ_HOST_ASSERT(peer.srcEndpointDesc.protocol == local.protocol and
                           peer.srcEndpointDesc.commAddr.type == local.commAddr.type and
                           std::memcmp(peer.srcEndpointDesc.commAddr.raws, local.commAddr.raws, sizeof(local.commAddr.raws)) == 0 and
                           peer.srcEndpointDesc.loc.locType == local.loc.locType and
                           std::memcmp(peer.srcEndpointDesc.loc.raws, local.loc.raws, sizeof(local.loc.raws)) == 0);
        }
        DG_HCOMM_CHECK(HcommEndpointCreate(&local, &endpoint));
        CommMem region{};
        region.type = COMM_MEM_TYPE_DEVICE;
        region.addr = buffer;
        region.size = num_bytes;
        DG_HCOMM_CHECK(HcommMemReg(endpoint, buffer_name.c_str(), &region, &memory));

        uint32_t port = 0;
        DG_HCOMM_CHECK(HcommEndpointGetListenPort(endpoint, &port));
        DJ_HOST_ASSERT(port > 0 and port <= UINT16_MAX);
        std::vector<uint16_t> ports(num_ranks);
        void* device_ports = nullptr;
        const size_t num_port_bytes = num_ranks * sizeof(uint16_t);
        DJ_ACL_CHECK(aclrtMalloc(&device_ports, num_port_bytes + sizeof(uint16_t), ACL_MEM_MALLOC_NORMAL_ONLY));
        auto local_port = static_cast<uint8_t*>(device_ports) + num_port_bytes;
        const auto port_value = static_cast<uint16_t>(port);
        DJ_ACL_CHECK(aclrtMemcpy(local_port, sizeof(port_value), &port_value, sizeof(port_value), ACL_MEMCPY_HOST_TO_DEVICE));
        const auto stream = c10_npu::getCurrentNPUStream();
        DG_HCOMM_CHECK(HcclAllGather(local_port, device_ports, 1, HCCL_DATA_TYPE_UINT16, comm, stream));
        DJ_ACL_CHECK(aclrtSynchronizeStream(stream));
        DJ_ACL_CHECK(aclrtMemcpy(ports.data(), num_port_bytes, device_ports, num_port_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        DJ_ACL_CHECK(aclrtFree(device_ports));

        const uint32_t num_channels = std::max(num_jetties, num_ranks - 1);
        // Paired peers must create matching channels; avoid allocating an unused full group
        const uint32_t num_extra_channels = num_channels % (num_ranks - 1);
        DJ_HOST_ASSERT(num_ranks % 2 == 0 or num_extra_channels % 2 == 0);
        const uint32_t num_groups = (num_channels + num_ranks - 2) / (num_ranks - 1);
        std::vector<HcommChannelDesc> descs(num_channels);
        DG_HCOMM_CHECK(HcommChannelDescInit(descs.data(), num_channels));
        names.resize(num_channels);
        channels.resize(num_channels);
        uint32_t channel_idx = 0;
        for (uint32_t group_idx = 0; group_idx < num_groups; ++ group_idx) {
            for (uint32_t peer_idx = 0; peer_idx < num_ranks; ++ peer_idx) {
                if (peer_idx == rank_idx)
                    continue;
                const uint32_t distance = (peer_idx + num_ranks - rank_idx) % num_ranks;
                if (group_idx + 1 == num_groups and num_extra_channels != 0 and
                    std::min(distance, num_ranks - distance) > num_extra_channels / 2 and
                    not (num_extra_channels % 2 != 0 and distance == num_ranks / 2))
                    continue;
                const uint32_t peer_slot = peer_idx < rank_idx ? peer_idx : peer_idx - 1;
                names[channel_idx] = buffer_name + "/" + std::to_string(std::min(rank_idx, peer_idx)) + "-" +
                                     std::to_string(std::max(rank_idx, peer_idx)) + "/" + std::to_string(group_idx);
                DJ_HOST_ASSERT(names[channel_idx].size() <= HCOMM_CHANNEL_NAME_MAX_LEN);
                auto& desc = descs[channel_idx];
                desc.remoteEndpoint = peers[peer_slot].dstEndpointDesc;
                desc.notifyNum = 3;
                desc.memHandles = &memory;
                desc.memHandleNum = 1;
                desc.role = rank_idx < peer_idx ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;
                desc.port = ports[std::min(rank_idx, peer_idx)];
                desc.channelName = names[channel_idx].c_str();
                ++ channel_idx;
            }
        }
        DJ_HOST_ASSERT(channel_idx == num_channels);
        DG_HCOMM_CHECK(HcommChannelCreate(endpoint, COMM_ENGINE_AIV, descs.data(), num_channels, channels.data()));
        std::vector<int32_t> statuses(num_channels, 1);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
        while (true) {
            const auto result = HcommChannelGetStatus(channels.data(), num_channels, statuses.data());
            if (result != HCCL_E_AGAIN)
                DG_HCOMM_CHECK(result);
            if (result == 0) {
                bool ready = true;
                for (auto status: statuses) {
                    DJ_HOST_ASSERT(status == 0 or status == 1, "URMA channel status {}", status);
                    ready &= status == 0;
                }
                if (ready)
                    break;
            }
            DJ_HOST_ASSERT(std::chrono::steady_clock::now() < deadline, "URMA channels did not connect");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        static_assert(sizeof(ChannelHandle) == sizeof(uint64_t));
        const size_t num_table_bytes = num_channels * sizeof(ChannelHandle);
        DJ_ACL_CHECK(aclrtMalloc(&device_channels, num_table_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        DJ_ACL_CHECK(aclrtMemcpy(device_channels, num_table_bytes, channels.data(), num_table_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    ~UrmaChannels() {
        if (device_channels != nullptr) aclrtFree(device_channels);
        if (not channels.empty()) HcommChannelDestroy(channels.data(), channels.size());
        if (memory != nullptr) HcommMemUnreg(endpoint, memory);
        if (endpoint != nullptr) HcommEndpointDestroy(endpoint);
    }
};

} // namespace deep_gemm
