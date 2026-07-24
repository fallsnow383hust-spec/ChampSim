#ifndef G_LBTP_H
#define G_LBTP_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string_view>
#include <unordered_map>

#include "address.h"
#include "champsim.h"
#include "modules.h"

// G-LBTP: Graph-based Loop-Boundary Translation Prefetcher.
//
// A stream node represents one static PIM site plus one operand role. Directed
// graph edges connect two runtime loop-boundary contexts. Each edge learns the
// signed byte delta between consecutive base addresses. A predicted taken
// backedge supplies the target context, selects the strongest matching edge
// from each stream's last context, and injects one page-aligned STLB prefetch
// only when that byte delta crosses a 4-KiB page.
struct g_lbtp : public champsim::modules::prefetcher {
  enum class lookup_result : uint8_t { selected = 0, no_edge = 1, low_confidence = 2, ambiguous = 3 };

  struct stream_node {
    uint64_t tag = 0;
    uint64_t last_address = 0;
    uint64_t last_instr_id = 0;
    uint8_t last_context = 0;
    bool valid = false;
    bool have_last = false;
    bool have_instr_id = false;
  };

  struct graph_edge {
    uint64_t stream_tag = 0;
    int64_t byte_delta = 0;
    uint16_t generation = 0;
    uint8_t occurrences = 0;
    int8_t usefulness = 0;
    uint8_t source_context = 0;
    uint8_t target_context = 0;
    uint8_t confidence = 0;
    uint8_t lru = 0;
    bool valid = false;
  };

  struct edge_selection {
    int64_t byte_delta = 0;
    uint16_t generation = 0;
    uint16_t set = 0;
    uint16_t score = 0;
    uint8_t way = 0;
    uint8_t source_context = 0;
    uint8_t target_context = 0;
    uint8_t confidence = 0;
    bool valid = false;
  };

  struct pending_entry {
    uint64_t prediction_id = 0;
    uint64_t trigger_address = 0;
    uint64_t issue_cycle = 0;
    uint64_t issue_seq = 0;
    uint64_t demand_cycle = 0;
    uint64_t fill_cycle = 0;
    int64_t byte_delta = 0;
    uint16_t edge_generation = 0;
    uint16_t edge_set = 0;
    uint16_t edge_score = 0;
    uint8_t edge_way = 0;
    uint8_t role = 0;
    uint8_t source_context = 0;
    uint8_t target_context = 0;
    uint8_t edge_confidence = 0;
    bool demand_seen = false;
    bool fill_seen = false;
    bool evicted_before_demand = false;
  };

  struct event_stats {
    uint64_t demand_access = 0;
    uint64_t demand_miss = 0;
    uint64_t useful_callback = 0;
    uint64_t graph_lookups = 0;
    uint64_t graph_no_edge = 0;
    uint64_t graph_low_confidence = 0;
    uint64_t graph_ambiguous = 0;
    uint64_t graph_selected = 0;
    uint64_t candidates = 0;
    uint64_t cross_page = 0;
    uint64_t filtered_resident = 0;
    uint64_t filtered_inflight = 0;
    uint64_t filtered_pending = 0;
    uint64_t filtered_capacity = 0;
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

  struct graph_counters {
    uint64_t boundary_triggers = 0;
    uint64_t transitions = 0;
    uint64_t edge_allocations = 0;
    uint64_t edge_reinforcements = 0;
    uint64_t edge_evictions = 0;
    uint64_t stream_collisions = 0;
    uint64_t positive_feedback = 0;
    uint64_t negative_feedback = 0;
    uint64_t stale_feedback = 0;
  };

  static constexpr std::size_t STREAM_ENTRIES = 128;
  static constexpr std::size_t EDGE_SETS = 128;
  static constexpr std::size_t EDGE_WAYS = 4;
  static constexpr std::size_t MAX_PENDING = 64;
  static constexpr uint64_t PENDING_MAX_DEMAND_DISTANCE = 4096;
  static constexpr uint8_t ROLE_COUNT = 3;
  static constexpr uint8_t CONTEXT_COUNT = 7;
  static constexpr uint8_t EDGE_CONFIDENCE_THRESHOLD = 2;
  static constexpr uint16_t EDGE_SCORE_MARGIN = 8;
  static constexpr uint64_t PIM_PC_BEGIN = 0x400000;
  static constexpr uint64_t PIM_PC_END = 0x500000;

  static_assert((STREAM_ENTRIES & (STREAM_ENTRIES - 1)) == 0);
  static_assert((EDGE_SETS & (EDGE_SETS - 1)) == 0);

  std::array<stream_node, STREAM_ENTRIES> stream_table{};
  std::array<std::array<graph_edge, EDGE_WAYS>, EDGE_SETS> edge_table{};
  std::array<std::array<event_stats, ROLE_COUNT>, CONTEXT_COUNT> stats{};
  graph_counters graph{};
  std::unordered_map<uint64_t, pending_entry> pending{};
  std::ofstream event_log{};

  uint64_t current_cycle = 0;
  uint64_t demand_seq = 0;
  uint64_t prediction_seq = 0;
  uint16_t edge_generation = 1;
  uint64_t ignored_non_pim = 0;
  uint64_t duplicate_callback = 0;
  uint64_t missing_runtime_context = 0;
  bool runtime_context_enabled = true;
  bool finalized = false;

  static inline g_lbtp* active_instance = nullptr;

  explicit g_lbtp(CACHE* cache) : champsim::modules::prefetcher(cache) {}

  static uint8_t role_from_ip(champsim::address ip);
  static uint64_t stream_key_from_ip(champsim::address ip);
  static uint64_t vpn_from_address(champsim::address addr);
  static std::string_view role_name(uint8_t role);
  static std::string_view context_name(uint8_t context);
  static bool signed_delta(uint64_t newer, uint64_t older, int64_t& result);
  static bool add_delta(uint64_t address, int64_t delta, uint64_t& result);
  static int8_t saturating_utility(int8_t value, int adjustment);

  std::size_t stream_index(uint64_t stream_key) const;
  std::size_t edge_set_index(uint64_t stream_key, uint8_t source_context) const;
  uint16_t edge_score(const graph_edge& edge) const;
  uint64_t context_pc(uint8_t context) const;
  void touch_edge(std::size_t set, std::size_t way);
  void train_edge(uint64_t stream_key, uint8_t source_context, uint8_t target_context, int64_t byte_delta);
  edge_selection select_edge(uint64_t stream_key, uint8_t source_context, lookup_result& result,
                             uint8_t required_target = CONTEXT_COUNT);
  void feedback_edge(const pending_entry& prediction, int adjustment);
  static void boundary_callback(uint8_t target_context);
  void on_loop_boundary(uint8_t target_context);

  bool local_tlb_contains(uint64_t vpn) const;
  bool local_tlb_inflight(uint64_t vpn) const;
  bool issue_candidate(uint64_t current_address, uint8_t role, const edge_selection& selection, uint32_t metadata_in);
  void record_demand_for_prediction(uint64_t vpn, bool cache_hit, bool useful_prefetch);
  void expire_pending();
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
