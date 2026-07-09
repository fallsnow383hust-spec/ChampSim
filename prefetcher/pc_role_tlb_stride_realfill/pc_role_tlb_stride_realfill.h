#ifndef PC_ROLE_TLB_STRIDE_REALFILL_H
#define PC_ROLE_TLB_STRIDE_REALFILL_H

#include <array>
#include <cstdint>
#include <deque>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "address.h"
#include "champsim.h"
#include "modules.h"

struct pc_role_tlb_stride_realfill : public champsim::modules::prefetcher {
  struct tracker_entry {
    uint64_t tag = 0;
    uint64_t last_vpn = 0;
    int64_t last_stride = 0;
    int confidence = 0;
    bool valid = false;
  };

  struct pending_prefetch {
    uint64_t issue_seq = 0;
    uint8_t role = 0;
  };

  struct role_stats {
    uint64_t demand_access = 0;
    uint64_t demand_miss = 0;
    uint64_t trained = 0;
    uint64_t predictions = 0;
    uint64_t issued_shadow = 0;
    uint64_t issued_real = 0;
    uint64_t real_rejected = 0;
    uint64_t shadow_useful_on_miss = 0;
    uint64_t shadow_timely_on_hit = 0;
    uint64_t shadow_evictions = 0;
    uint64_t champ_prefetch_useful_callback = 0;
    uint64_t real_useful_with_pending = 0;
    uint64_t timeliness_sum = 0;
    uint64_t timeliness_max = 0;
    std::array<uint64_t, 6> timeliness_buckets{};
    uint64_t real_timeliness_sum = 0;
    uint64_t real_timeliness_max = 0;
    std::array<uint64_t, 6> real_timeliness_buckets{};
  };

  static constexpr std::size_t TRACKER_ENTRIES = 1024;
  static constexpr std::size_t SHADOW_BUFFER_ENTRIES = 128;
  static constexpr int CONFIDENCE_THRESHOLD = 2;
  static constexpr int PREFETCH_DEGREE = 1;
  static constexpr int ROLE_COUNT = 3;

  std::array<tracker_entry, TRACKER_ENTRIES> table{};
  std::deque<uint64_t> shadow_fifo{};
  std::unordered_map<uint64_t, pending_prefetch> shadow_buffer{};
  std::array<role_stats, ROLE_COUNT> per_role{};

  uint64_t demand_seq = 0;
  uint64_t trained = 0;
  uint64_t predictions = 0;
  uint64_t issued_shadow = 0;
  uint64_t issued_real = 0;
  uint64_t real_rejected = 0;
  uint64_t shadow_useful_on_miss = 0;
  uint64_t shadow_redundant_on_hit = 0;
  uint64_t shadow_evictions = 0;
  uint64_t champ_prefetch_useful_callback = 0;

  explicit pc_role_tlb_stride_realfill(CACHE* cache) : champsim::modules::prefetcher(cache) {}

  static uint8_t role_from_ip(champsim::address ip);
  static std::string_view role_name(uint8_t role);
  static std::size_t timeliness_bucket(uint64_t distance);
  void record_timeliness(uint8_t role, uint64_t distance);
  void record_real_timeliness(uint8_t role, uint64_t distance);

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
  void prefetcher_initialize();
  void prefetcher_final_stats();
};

#endif
