#ifndef LOOP_BOUNDARY_TLB_REALFILL_H
#define LOOP_BOUNDARY_TLB_REALFILL_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "address.h"
#include "champsim.h"
#include "modules.h"

struct loop_boundary_tlb_realfill : public champsim::modules::prefetcher {
  enum class pattern_kind : uint8_t { steady = 0, boundary_post = 1, boundary_recurrence = 2 };

  struct stride_state {
    int64_t stride = 0;
    uint8_t confidence = 0;
    bool valid = false;
  };

  struct tracker_entry {
    uint64_t tag = 0;
    uint64_t last_address = 0;
    uint8_t last_phase = 0;
    bool valid = false;
    bool have_last = false;
    stride_state steady{};
    std::array<stride_state, 7> post_boundary{};
    std::array<stride_state, 7> boundary_recurrence{};
    std::array<uint64_t, 7> last_phase_address{};
    std::array<bool, 7> phase_seen{};
  };

  struct pending_entry {
    uint64_t prediction_id = 0;
    uint64_t trigger_address = 0;
    uint64_t issue_cycle = 0;
    uint64_t issue_seq = 0;
    uint64_t demand_cycle = 0;
    uint64_t fill_cycle = 0;
    uint8_t role = 0;
    uint8_t phase = 0;
    pattern_kind pattern = pattern_kind::steady;
    bool demand_seen = false;
    bool fill_seen = false;
    bool evicted_before_demand = false;
  };

  struct event_stats {
    uint64_t demand_access = 0;
    uint64_t demand_miss = 0;
    uint64_t useful_callback = 0;
    uint64_t candidates = 0;
    uint64_t cross_page = 0;
    uint64_t filtered_resident = 0;
    uint64_t filtered_inflight = 0;
    uint64_t filtered_pending = 0;
    uint64_t issued = 0;
    uint64_t rejected = 0;
    uint64_t demanded_after_issue = 0;
    uint64_t timely = 0;
    uint64_t late_at_demand = 0;
    uint64_t late_completed = 0;
    uint64_t nonuseful_hit = 0;
    uint64_t too_early = 0;
    uint64_t never_demanded = 0;
    uint64_t unresolved_late = 0;
    uint64_t issue_to_demand_sum = 0;
    uint64_t issue_to_demand_max = 0;
    uint64_t ready_lead_sum = 0;
    uint64_t ready_lead_max = 0;
    uint64_t late_by_sum = 0;
    uint64_t late_by_max = 0;
  };

  static constexpr std::size_t TRACKER_ENTRIES = 4096;
  static constexpr uint8_t ROLE_COUNT = 3;
  static constexpr uint8_t PHASE_COUNT = 7;
  static constexpr uint8_t PATTERN_COUNT = 3;
  static constexpr uint8_t STEADY_CONFIDENCE = 2;
  static constexpr uint8_t BOUNDARY_CONFIDENCE = 1;
  static constexpr uint64_t PIM_PC_BEGIN = 0x400000;
  static constexpr uint64_t PIM_PC_END = 0x500000;

  std::array<tracker_entry, TRACKER_ENTRIES> table{};
  std::array<std::array<event_stats, ROLE_COUNT>, PHASE_COUNT> stats{};
  std::array<event_stats, PATTERN_COUNT> pattern_stats{};
  std::unordered_map<uint64_t, pending_entry> pending{};
  std::unordered_set<uint64_t> observed_requests{};
  std::ofstream event_log{};

  uint64_t current_cycle = 0;
  uint64_t demand_seq = 0;
  uint64_t prediction_seq = 0;
  uint64_t ignored_non_pim = 0;
  uint64_t duplicate_callback = 0;
  uint64_t missing_runtime_context = 0;
  bool runtime_context_enabled = true;
  bool finalized = false;

  explicit loop_boundary_tlb_realfill(CACHE* cache) : champsim::modules::prefetcher(cache) {}

  static uint8_t role_from_ip(champsim::address ip);
  static uint64_t base_key_from_ip(champsim::address ip);
  static uint64_t vpn_from_address(champsim::address addr);
  static std::string_view role_name(uint8_t role);
  static std::string_view phase_name(uint8_t phase);
  static std::string_view pattern_name(pattern_kind pattern);
  static void update_stride(stride_state& state, int64_t stride);

  bool local_tlb_contains(uint64_t vpn) const;
  bool local_tlb_inflight(uint64_t vpn) const;
  bool issue_candidate(uint64_t current_address, int64_t byte_stride, uint8_t role, uint8_t phase, pattern_kind pattern,
                       uint32_t metadata_in);
  void record_demand_for_prediction(uint64_t vpn, bool cache_hit, bool useful_prefetch);
  void write_event(uint64_t vpn, const pending_entry& prediction, std::string_view outcome);
  void finalize_unresolved();

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address full_addr, champsim::address ip, uint8_t cache_hit,
                                    bool useful_prefetch, access_type type, uint64_t instr_id, uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr,
                                 uint32_t metadata_in);
  void prefetcher_cycle_operate();
  void prefetcher_initialize();
  void prefetcher_final_stats();
};

#endif
