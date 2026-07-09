#ifndef PC_ROLE_TLB_STRIDE_H
#define PC_ROLE_TLB_STRIDE_H

#include <array>
#include <cstdint>
#include <deque>
#include <unordered_set>

#include "address.h"
#include "champsim.h"
#include "modules.h"

struct pc_role_tlb_stride : public champsim::modules::prefetcher {
  struct tracker_entry {
    uint64_t tag = 0;
    uint64_t last_vpn = 0;
    int64_t last_stride = 0;
    int confidence = 0;
    bool valid = false;
  };

  static constexpr std::size_t TRACKER_ENTRIES = 1024;
  static constexpr std::size_t SHADOW_BUFFER_ENTRIES = 128;
  static constexpr int CONFIDENCE_THRESHOLD = 2;
  static constexpr int PREFETCH_DEGREE = 1;

  std::array<tracker_entry, TRACKER_ENTRIES> table{};
  std::deque<uint64_t> shadow_fifo{};
  std::unordered_set<uint64_t> shadow_buffer{};

  uint64_t issued = 0;
  uint64_t useful = 0;
  uint64_t trained = 0;
  uint64_t predictions = 0;
  uint64_t shadow_useful_on_miss = 0;
  uint64_t shadow_redundant_on_hit = 0;
  uint64_t shadow_evictions = 0;

  explicit pc_role_tlb_stride(CACHE* cache) : champsim::modules::prefetcher(cache) {}

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
  void prefetcher_initialize();
  void prefetcher_final_stats();
};

#endif
