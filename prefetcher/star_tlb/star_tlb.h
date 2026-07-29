#ifndef STAR_TLB_H
#define STAR_TLB_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "address.h"
#include "champsim.h"
#include "gemm_runtime_loop_context.h"
#include "modules.h"

// STAR-TLB: Semantic Tile-footprint and Affine-Reuse Translation Prefetcher.
//
// The simulator receives PIMCFG/PIMGEMM descriptors through the canonical CSV
// named by STAR_TLB_DESCRIPTOR_CSV. This sideband models fields that are
// architecturally visible to a real implementation. The binary trace still
// supplies all demand translations and all dynamic loop branches.
//
// RPRG edges are trained atomically for the A/B/C address tuple:
//
//   (PIM site, source loop phase)
//       -> (target loop phase, delta_A, delta_B, delta_C, next tile shape)
//
// A selected edge predicts a complete descriptor. The Page Footprint
// Generator expands the descriptor into exact A/B/C VPN sets. The scheduler
// uses an adaptive PTW-latency/PIM-interval estimate to issue translations
// close to, but before, their predicted first use.
struct star_tlb : public champsim::modules::prefetcher {
  static constexpr uint8_t ROLE_A = 0;
  static constexpr uint8_t ROLE_B = 1;
  static constexpr uint8_t ROLE_C = 2;
  static constexpr uint8_t ROLE_COUNT = 3;
  static constexpr uint8_t CONTEXT_COUNT = 7;
  static constexpr uint8_t ANY_CONTEXT = 0xff;

  struct descriptor {
    uint64_t raw_site_pc = 0;
    uint64_t a_base = 0;
    uint64_t b_base = 0;
    uint64_t c_base = 0;
    uint64_t lda = 0;
    uint64_t ldb = 0;
    uint64_t ldc = 0;
    uint8_t valid_m = 0;
    uint8_t valid_n = 0;
    uint8_t valid_k = 0;
    uint8_t flags = 0;
  };

  struct graph_edge {
    uint64_t site_tag = 0;
    std::array<int64_t, ROLE_COUNT> byte_delta{};
    uint64_t lda = 0;
    uint64_t ldb = 0;
    uint64_t ldc = 0;
    uint16_t generation = 0;
    uint8_t occurrences = 0;
    int8_t usefulness = 0;
    uint8_t source_context = 0;
    uint8_t target_context = 0;
    uint8_t confidence = 0;
    uint8_t lru = 0;
    uint8_t valid_m = 0;
    uint8_t valid_n = 0;
    uint8_t valid_k = 0;
    uint8_t flags = 0;
    bool valid = false;
  };

  struct edge_selection {
    std::array<int64_t, ROLE_COUNT> byte_delta{};
    uint64_t lda = 0;
    uint64_t ldb = 0;
    uint64_t ldc = 0;
    uint16_t generation = 0;
    uint16_t set = 0;
    uint16_t score = 0;
    uint8_t way = 0;
    uint8_t source_context = 0;
    uint8_t target_context = 0;
    uint8_t confidence = 0;
    uint8_t valid_m = 0;
    uint8_t valid_n = 0;
    uint8_t valid_k = 0;
    uint8_t flags = 0;
    bool valid = false;
  };

  struct candidate_entry {
    uint64_t vpn = 0;
    uint64_t created_cycle = 0;
    uint64_t ready_cycle = 0;
    uint64_t target_cycle = 0;
    uint16_t edge_generation = 0;
    uint16_t edge_set = 0;
    uint16_t edge_score = 0;
    uint8_t edge_way = 0;
    uint8_t role = 0;
    uint8_t distance = 0;
    uint8_t source_context = 0;
    uint8_t target_context = 0;
    uint8_t edge_confidence = 0;
    uint8_t reuse_count = 1;
  };

  struct pending_entry : candidate_entry {
    uint64_t prediction_id = 0;
    uint64_t issue_cycle = 0;
    uint64_t demand_cycle = 0;
    uint64_t fill_cycle = 0;
    bool demand_seen = false;
    bool fill_seen = false;
    bool evicted_before_demand = false;
  };

  // Side-effect-free Acc@1 record. It is created before the target descriptor
  // is visible and resolved only when that exact dynamic descriptor retires.
  struct accuracy_record {
    descriptor predicted{};
    uint64_t prediction_id = 0;
    uint64_t source_descriptor_index = 0;
    uint64_t target_descriptor_index = 0;
    uint64_t source_instr_id = 0;
    uint64_t branch_instr_id = 0;
    uint64_t loop_branch_pc = 0;
    uint64_t prediction_cycle = 0;
    uint16_t edge_generation = 0;
    uint16_t edge_set = 0;
    uint16_t edge_score = 0;
    uint8_t edge_way = 0;
    uint8_t source_context = 0;
    uint8_t predicted_context = 0;
    uint8_t edge_confidence = 0;
    bool primary_for_target = false;
    bool source_committed = false;
  };

  struct accuracy_bucket {
    uint64_t predictions = 0;
    uint64_t resolved = 0;
    uint64_t context_correct = 0;
    std::array<uint64_t, ROLE_COUNT> byte_correct{};
    std::array<uint64_t, ROLE_COUNT> vpn_correct{};
    uint64_t triplet_byte_correct = 0;
    uint64_t triplet_vpn_correct = 0;
    uint64_t descriptor_exact = 0;
  };

  struct accuracy_stats {
    accuracy_bucket all{};
    accuracy_bucket primary{};
    std::array<accuracy_bucket, CONTEXT_COUNT> by_actual_context{};
    uint64_t duplicate_targets = 0;
    uint64_t source_not_committed = 0;
    uint64_t out_of_range = 0;
    uint64_t unresolved = 0;
  };

  enum class edge_lookup_status : uint8_t { selected, no_edge, low_confidence, ambiguous };

  struct oracle_graph_stats {
    accuracy_bucket all{};
    std::array<accuracy_bucket, CONTEXT_COUNT> by_actual_context{};
    std::array<uint64_t, CONTEXT_COUNT> eligible_by_actual_context{};
    uint64_t eligible = 0;
    uint64_t selected = 0;
    uint64_t no_edge = 0;
    uint64_t low_confidence = 0;
    uint64_t ambiguous = 0;
    uint64_t address_overflow = 0;
  };

  struct branch_audit_stats {
    uint64_t events = 0;
    uint64_t predicted = 0;
    uint64_t actual = 0;
    uint64_t exact = 0;
    uint64_t false_positive = 0;
    uint64_t false_negative = 0;
    uint64_t wrong_target = 0;
    std::array<uint64_t, CONTEXT_COUNT> events_by_context{};
    std::array<uint64_t, CONTEXT_COUNT> predicted_by_context{};
    std::array<uint64_t, CONTEXT_COUNT> actual_by_context{};
    std::array<uint64_t, CONTEXT_COUNT> exact_by_context{};
    std::array<uint64_t, CONTEXT_COUNT> false_positive_by_context{};
    std::array<uint64_t, CONTEXT_COUNT> false_negative_by_context{};
    std::array<uint64_t, CONTEXT_COUNT> wrong_target_by_context{};
  };

  // Allocated before retirement so a loop boundary can use the nearest older
  // dynamic PIM descriptor rather than a stale global committed descriptor.
  struct pdq_entry {
    descriptor value{};
    uint64_t descriptor_index = 0;
    uint64_t instr_id = 0;
    uint64_t dispatch_cycle = 0;
    uint8_t context = 0;
    bool committed = false;
  };

  // Holds a predicted loop event until it can be paired by program order.
  struct lbq_entry {
    uint64_t branch_instr_id = 0;
    uint64_t created_cycle = 0;
    uint8_t target_context = 0;
  };

  struct role_stats {
    uint64_t demand_access = 0;
    uint64_t demand_miss = 0;
    uint64_t footprint_pages = 0;
    uint64_t footprint_reused = 0;
    uint64_t candidates = 0;
    uint64_t candidate_merged = 0;
    uint64_t filtered_resident = 0;
    uint64_t filtered_inflight = 0;
    uint64_t filtered_pending = 0;
    uint64_t filtered_capacity = 0;
    uint64_t issued = 0;
    uint64_t rejected = 0;
    uint64_t demanded_after_issue = 0;
    uint64_t timely = 0;
    uint64_t late = 0;
    uint64_t late_completed = 0;
    uint64_t redundant = 0;
    uint64_t too_early = 0;
    uint64_t never_demanded = 0;
    uint64_t unresolved_late = 0;
    uint64_t issue_to_demand_sum = 0;
    uint64_t ready_lead_sum = 0;
    uint64_t late_by_sum = 0;
  };

  struct graph_stats {
    uint64_t boundary_triggers = 0;
    uint64_t descriptors_seen = 0;
    uint64_t descriptor_mismatch = 0;
    uint64_t transitions = 0;
    uint64_t edge_allocations = 0;
    uint64_t edge_reinforcements = 0;
    uint64_t edge_evictions = 0;
    uint64_t edge_selected = 0;
    uint64_t no_edge = 0;
    uint64_t low_confidence = 0;
    uint64_t ambiguous = 0;
    uint64_t prediction_chains = 0;
    uint64_t predicted_descriptors = 0;
    uint64_t positive_feedback = 0;
    uint64_t negative_feedback = 0;
    uint64_t stale_feedback = 0;
    uint64_t pdq_dispatches = 0;
    uint64_t pdq_commits = 0;
    uint64_t pdq_capacity_stalls = 0;
    uint64_t pdq_high_watermark = 0;
    uint64_t lbq_allocations = 0;
    uint64_t lbq_pairs = 0;
    uint64_t lbq_capacity_drops = 0;
    uint64_t pair_sequence_conflicts = 0;
    uint64_t committed_base_lag_sum = 0;
    uint64_t committed_base_lag_max = 0;
  };

  // Associativity-only ablation: edge_set_index intentionally remains
  // hash(PIM site, source context), so 4-way versus 8-way is causal.
  static constexpr std::size_t EDGE_SETS = 64;
  static constexpr std::size_t EDGE_WAYS = 8;
  static constexpr std::size_t MAX_CANDIDATES = 512;
  static constexpr std::size_t MAX_PENDING = 256;
  // The measured full-trace frontend lead is at most 103 PIM descriptors.
  // Keep a bounded power-of-two margin; any overflow is reported as an
  // evaluation failure instead of being silently hidden.
  static constexpr std::size_t PDQ_ENTRIES = 128;
  static constexpr std::size_t LBQ_ENTRIES = 16;
  static constexpr uint8_t EDGE_CONFIDENCE_THRESHOLD = 2;
  static constexpr uint16_t EDGE_SCORE_MARGIN = 8;
  static constexpr uint8_t MAX_LOOKAHEAD = 4;
  static constexpr uint64_t FRONTEND_TO_PIM_CYCLES = 8;
  static constexpr uint64_t PIM_PC_BEGIN = 0x400000;
  static constexpr uint64_t PIM_PC_END = 0x500000;

  static_assert((EDGE_SETS & (EDGE_SETS - 1)) == 0);

  std::vector<descriptor> descriptors{};
  std::size_t descriptor_cursor = 0;
  descriptor last_committed_descriptor{};
  uint64_t last_committed_site_tag = 0;
  uint64_t last_committed_descriptor_cycle = 0;
  uint64_t last_committed_descriptor_index = 0;
  uint8_t last_committed_context = 0;
  bool have_last_committed_descriptor = false;
  bool descriptor_sideband_ready = false;

  std::deque<pdq_entry> pdq{};
  std::deque<lbq_entry> lbq{};
  std::array<std::array<graph_edge, EDGE_WAYS>, EDGE_SETS> edge_table{};
  std::unordered_map<uint64_t, candidate_entry> candidates{};
  std::unordered_map<uint64_t, pending_entry> pending{};
  std::deque<accuracy_record> accuracy_pending{};
  std::unordered_set<uint64_t> accuracy_primary_targets{};
  std::unordered_set<uint64_t> accuracy_resolved_targets{};
  std::unordered_set<uint64_t> accuracy_correct_targets{};
  std::array<role_stats, ROLE_COUNT> stats{};
  graph_stats graph{};
  accuracy_stats accuracy{};
  oracle_graph_stats oracle_graph{};
  branch_audit_stats branch_audit{};
  std::ofstream event_log{};
  std::ofstream accuracy_log{};
  std::ofstream graph_log{};
  std::ofstream oracle_graph_log{};
  std::ofstream branch_log{};

  uint64_t current_cycle = 0;
  uint64_t demand_seq = 0;
  uint64_t prediction_seq = 0;
  uint64_t accuracy_prediction_seq = 0;
  uint64_t pim_interval_ema = 128;
  uint64_t walk_latency_ema = 256;
  uint16_t edge_generation = 1;
  uint64_t ignored_non_pim = 0;
  uint64_t missing_runtime_context = 0;
  std::size_t descriptor_limit = 0;
  bool accuracy_only = false;
  bool graph_oracle_only = false;
  bool finalized = false;

  static inline star_tlb* active_instance = nullptr;

  explicit star_tlb(CACHE* cache) : champsim::modules::prefetcher(cache) {}

  static uint8_t role_from_ip(champsim::address ip);
  static uint64_t site_key_from_ip(champsim::address ip);
  static uint64_t vpn_from_address(champsim::address addr);
  static std::string_view role_name(uint8_t role);
  static std::string_view context_name(uint8_t context);
  static bool signed_delta(uint64_t newer, uint64_t older, int64_t& result);
  static bool add_delta(uint64_t address, int64_t delta, uint64_t& result);
  static int8_t saturating_utility(int8_t value, int adjustment);
  static uint64_t ema(uint64_t old_value, uint64_t sample);

  bool load_descriptors(const char* path);
  std::size_t edge_set_index(uint64_t site_tag, uint8_t source_context) const;
  uint16_t edge_score(const graph_edge& edge) const;
  void touch_edge(std::size_t set, std::size_t way);
  void train_edge(uint64_t site_tag, uint8_t source_context, uint8_t target_context, const descriptor& previous,
                  const descriptor& current);
  edge_selection find_edge(uint64_t site_tag, uint8_t source_context, uint8_t required_target,
                           edge_lookup_status& status) const;
  edge_selection select_edge(uint64_t site_tag, uint8_t source_context, uint8_t required_target);
  void feedback_edge(const pending_entry& prediction, int adjustment);

  static void boundary_callback(uint64_t branch_instr_id, uint8_t target_context);
  static void resolved_branch_callback(const gemm_runtime_loop_context::resolved_backedge_event& event);
  static void descriptor_dispatch_callback(uint64_t descriptor_index, uint64_t instr_id, uint8_t context);
  static void descriptor_callback(uint64_t descriptor_index, uint64_t instr_id, uint8_t context);
  void on_loop_boundary(uint64_t branch_instr_id, uint8_t target_context);
  void on_resolved_branch(const gemm_runtime_loop_context::resolved_backedge_event& event);
  void on_descriptor_dispatch(uint64_t descriptor_index, uint64_t instr_id, uint8_t context);
  void on_descriptor_marker(uint64_t descriptor_index, uint64_t instr_id, uint8_t context);
  void pair_boundaries();
  void predict_from_pair(const pdq_entry& source, uint64_t branch_instr_id, uint8_t target_context);
  void evaluate_oracle_graph(uint64_t descriptor_index, uint64_t site_tag, uint8_t context, const descriptor& current);
  void observe_descriptor(uint64_t descriptor_index, uint64_t site_tag, uint8_t context, const descriptor& current);
  std::optional<descriptor> apply_edge(const descriptor& current, const edge_selection& edge) const;
  void record_accuracy_prediction(const pdq_entry& source, uint64_t branch_instr_id, const descriptor& predicted,
                                  const edge_selection& edge);
  void mark_accuracy_source_committed(uint64_t descriptor_index, uint64_t instr_id);
  void resolve_accuracy(uint64_t descriptor_index, uint8_t actual_context, const descriptor& actual);
  void update_accuracy_bucket(accuracy_bucket& bucket, const accuracy_record& prediction, uint8_t actual_context,
                              const descriptor& actual);
  void write_accuracy_event(const accuracy_record& prediction, uint8_t actual_context, const descriptor* actual,
                            std::string_view outcome);
  void finalize_accuracy();
  void write_graph_event(std::string_view operation, std::size_t set, int way, const graph_edge* edge, uint64_t site_tag,
                         uint8_t source_context, uint8_t target_context, uint16_t score = 0);

  std::vector<uint64_t> footprint(const descriptor& desc, uint8_t role) const;
  void enqueue_footprint(const descriptor& predicted, uint8_t distance, const edge_selection& edge);
  void enqueue_vpn(uint64_t vpn, uint8_t role, uint8_t distance, const edge_selection& edge);
  uint8_t lookahead_distance() const;
  bool local_tlb_contains(uint64_t vpn) const;
  bool local_tlb_inflight(uint64_t vpn) const;
  bool issue_one_candidate(uint32_t metadata_in = 0);
  void record_demand(uint64_t vpn, bool cache_hit, bool useful_prefetch);
  void expire_entries();
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
