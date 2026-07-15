#ifndef LOOP_BOUNDARY_TLB_REALFILL_H
#define LOOP_BOUNDARY_TLB_REALFILL_H

#include <array>
#include <cstdint>
#include <deque>
#include <string_view>
#include <unordered_map>

#include "address.h"
#include "champsim.h"
#include "modules.h"

struct loop_boundary_tlb_realfill : public champsim::modules::prefetcher {
  struct tracker_entry {
    uint64_t tag = 0;
    uint64_t last_vpn = 0;
    int64_t stride = 0;
    uint8_t confidence = 0;
    bool valid = false;
  };

  struct pending_entry {
    uint64_t issue_seq = 0;
    uint8_t role = 0;
    uint8_t phase = 0;
  };

  struct stream_stats {
    uint64_t accesses = 0;
    uint64_t misses = 0;
    uint64_t trained = 0;
    uint64_t predictions = 0;
    uint64_t issued = 0;
    uint64_t rejected = 0;
    uint64_t pending_hits = 0;
    uint64_t real_useful = 0;
    uint64_t lead_sum = 0;
    uint64_t lead_max = 0;
  };

  static constexpr std::size_t TRACKER_ENTRIES = 4096;
  static constexpr std::size_t PENDING_ENTRIES = 256;
  static constexpr uint8_t CONFIDENCE_THRESHOLD = 2;
  static constexpr uint8_t ROLE_COUNT = 3;
  static constexpr uint8_t PHASE_COUNT = 7;

  std::array<tracker_entry, TRACKER_ENTRIES> table{};
  std::array<std::array<stream_stats, ROLE_COUNT>, PHASE_COUNT> stats{};
  std::unordered_map<uint64_t, pending_entry> pending{};
  std::deque<uint64_t> pending_fifo{};

  uint64_t demand_seq = 0;
  uint64_t duplicate_vpn = 0;
  uint64_t duplicate_prediction = 0;
  uint64_t pending_evictions = 0;
  uint64_t champ_useful = 0;

  explicit loop_boundary_tlb_realfill(CACHE* cache) : champsim::modules::prefetcher(cache) {}

  static uint8_t role_from_ip(champsim::address ip);
  static uint8_t phase_from_ip(champsim::address ip);
  static std::string_view role_name(uint8_t role);
  static std::string_view phase_name(uint8_t phase);

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr,
                                 uint32_t metadata_in);
  void prefetcher_initialize();
  void prefetcher_final_stats();
};

#endif
