#ifndef B_NEXT_VPN_REALFILL_H
#define B_NEXT_VPN_REALFILL_H

#include <array>
#include <cstdint>
#include <deque>
#include <unordered_map>

#include "address.h"
#include "champsim.h"
#include "modules.h"

struct b_next_vpn_realfill : public champsim::modules::prefetcher {
  struct pending_prefetch {
    uint64_t issue_seq = 0;
  };

  static constexpr uint8_t B_ROLE = 1;
  static constexpr std::size_t PENDING_ENTRIES = 4096;

  std::deque<uint64_t> pending_fifo{};
  std::unordered_map<uint64_t, pending_prefetch> pending{};

  uint64_t demand_seq = 0;
  uint64_t demand_access = 0;
  uint64_t b_demand_access = 0;
  uint64_t b_demand_miss = 0;
  uint64_t b_candidates = 0;
  uint64_t b_pending_inserted = 0;
  uint64_t b_pending_duplicate = 0;
  uint64_t b_issued_real = 0;
  uint64_t b_real_rejected = 0;
  uint64_t b_prefetch_useful_callback = 0;
  uint64_t b_real_useful_with_pending = 0;
  uint64_t b_pending_evictions = 0;
  uint64_t b_timeliness_sum = 0;
  uint64_t b_timeliness_max = 0;
  std::array<uint64_t, 6> b_timeliness_buckets{};

  explicit b_next_vpn_realfill(CACHE* cache) : champsim::modules::prefetcher(cache) {}

  static uint8_t role_from_ip(champsim::address ip);
  static std::size_t timeliness_bucket(uint64_t distance);
  void record_timeliness(uint64_t distance);

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
  void prefetcher_initialize();
  void prefetcher_final_stats();
};

#endif
