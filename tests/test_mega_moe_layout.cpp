// Host-only regression: c++ -std=c++20 -Ideep_gemm/include tests/test_mega_moe_layout.cpp -o /tmp/test_mega_moe_layout
#include <cstdio>

#include <deep_gemm/layout/mega_moe.hpp>

using namespace deep_gemm::mega_moe::layout;

template <const auto& workspace, uint32_t kExpectedCapacity>
constexpr bool check_ring() {
    constexpr auto capacity = workspace.get_num_ring_blocks();
    if (capacity != kExpectedCapacity or workspace.config.num_aivs != 2 * workspace.config.num_ai_cores)
        return false;

    // Every live window must have distinct physical slots, including windows
    // crossing the wrap boundary. A mask with capacity 28/36/112/144 fails here.
    for (uint32_t begin = 0; begin < 2 * capacity; ++ begin) {
        bool occupied[capacity]{};
        for (uint32_t i = begin; i < begin + capacity; ++ i) {
            const auto slot = get_ring_block_idx<workspace>(i);
            if (slot >= capacity or occupied[slot])
                return false;
            occupied[slot] = true;
            if (get_ring_block_idx<workspace>(i + capacity) != slot)
                return false;
            if (get_ring_m_idx<workspace>(i, BLOCK_M - 1) >= workspace.num_ring_rows)
                return false;
        }
    }
    return true;
}

template <uint32_t kNumCores>
constexpr bool check_full_pool() {
    constexpr auto& workspace = kWorkspace<8, 8, 0, 1, 1, 5120, 2304, kNumCores>;
    if (workspace.get_num_ring_blocks() != workspace.get_num_max_m_blocks())
        return false;
    for (uint32_t i = 0; i < workspace.get_num_max_m_blocks(); ++ i)
        if (get_ring_block_idx<workspace>(i) != i)
            return false;
    return true;
}

// H=5120 selects one overlap window; H=2048 selects four.
static_assert(check_ring<kWorkspace<8, 384, 1, 4096, 6, 5120, 2304, 28>, 28>());
static_assert(check_ring<kWorkspace<8, 384, 1, 4096, 6, 5120, 2304, 32>, 32>());
static_assert(check_ring<kWorkspace<8, 384, 1, 4096, 6, 5120, 2304, 36>, 36>());
static_assert(check_ring<kWorkspace<8, 384, 1, 4096, 6, 2048, 2304, 28>, 112>());
static_assert(check_ring<kWorkspace<8, 384, 1, 4096, 6, 2048, 2304, 32>, 128>());
static_assert(check_ring<kWorkspace<8, 384, 1, 4096, 6, 2048, 2304, 36>, 144>());
static_assert(check_full_pool<28>() and check_full_pool<32>() and check_full_pool<36>());

int main() {
    std::puts("PASS: 28/32/36-core single-window, four-window and full-pool layouts");
}
