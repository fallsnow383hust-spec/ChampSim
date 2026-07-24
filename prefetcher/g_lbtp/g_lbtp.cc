#include "g_lbtp.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>

#include <fmt/core.h>

#include "cache.h"
#include "gemm_runtime_loop_context.h"

namespace
{
template <typename AddressSlice>
uint64_t as_u64(AddressSlice addr)
{
  return addr.template to<uint64_t>();
}

double average(uint64_t sum, uint64_t count) { return count == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(count); }
} // namespace

uint8_t g_lbtp::role_from_ip(champsim::address ip) { return static_cast<uint8_t>(as_u64(ip) & 0x3ULL); }

uint64_t g_lbtp::stream_key_from_ip(champsim::address ip)
{
  // The trace encodes only the static PIM site and the A/B/C role in the IP.
  return as_u64(ip);
}

uint64_t g_lbtp::vpn_from_address(champsim::address addr) { return as_u64(champsim::page_number{addr}); }

std::string_view g_lbtp::role_name(uint8_t role)
{
  constexpr std::array<std::string_view, ROLE_COUNT> names{"A", "B", "C"};
  return role < names.size() ? names[role] : "unknown";
}

std::string_view g_lbtp::context_name(uint8_t context)
{
  constexpr std::array<std::string_view, CONTEXT_COUNT> names{
      "NO_BACKEDGE", "BACKEDGE_1", "BACKEDGE_2", "BACKEDGE_3", "BACKEDGE_4", "BACKEDGE_5", "BACKEDGE_6"};
  return context < names.size() ? names[context] : "unknown";
}

bool g_lbtp::signed_delta(uint64_t newer, uint64_t older, int64_t& result)
{
  if (newer >= older) {
    const auto magnitude = newer - older;
    if (magnitude > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return false;
    result = static_cast<int64_t>(magnitude);
    return true;
  }

  const auto magnitude = older - newer;
  if (magnitude > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  result = -static_cast<int64_t>(magnitude);
  return true;
}

bool g_lbtp::add_delta(uint64_t address, int64_t delta, uint64_t& result)
{
  if (delta >= 0) {
    const auto magnitude = static_cast<uint64_t>(delta);
    if (address > std::numeric_limits<uint64_t>::max() - magnitude)
      return false;
    result = address + magnitude;
    return true;
  }

  const auto magnitude = static_cast<uint64_t>(-(delta + 1)) + 1;
  if (address < magnitude)
    return false;
  result = address - magnitude;
  return true;
}

int8_t g_lbtp::saturating_utility(int8_t value, int adjustment)
{
  const auto updated = std::clamp<int>(static_cast<int>(value) + adjustment, -4, 3);
  return static_cast<int8_t>(updated);
}

std::size_t g_lbtp::stream_index(uint64_t stream_key) const
{
  const auto mixed = stream_key ^ (stream_key >> 11) ^ (stream_key >> 23);
  return static_cast<std::size_t>(mixed) & (STREAM_ENTRIES - 1);
}

std::size_t g_lbtp::edge_set_index(uint64_t stream_key, uint8_t source_context) const
{
  const auto context_hash = static_cast<uint64_t>(source_context + 1) * 0x9e3779b97f4a7c15ULL;
  const auto mixed = stream_key ^ (stream_key >> 17) ^ context_hash ^ (context_hash >> 29);
  return static_cast<std::size_t>(mixed) & (EDGE_SETS - 1);
}

uint16_t g_lbtp::edge_score(const graph_edge& edge) const
{
  const auto utility = static_cast<uint16_t>(static_cast<int>(edge.usefulness) + 4);
  return static_cast<uint16_t>(static_cast<uint16_t>(edge.confidence) * 64U + static_cast<uint16_t>(edge.occurrences)
                               + utility * 8U);
}

uint64_t g_lbtp::context_pc(uint8_t context) const
{
  if (context == 0 || context >= gemm_runtime_loop_context::state.context_branch_pc.size())
    return 0;
  return gemm_runtime_loop_context::state.context_branch_pc[context];
}

void g_lbtp::touch_edge(std::size_t set, std::size_t way)
{
  for (std::size_t index = 0; index < EDGE_WAYS; ++index) {
    auto& edge = edge_table[set][index];
    if (!edge.valid)
      continue;
    if (index == way)
      edge.lru = 0;
    else
      edge.lru = std::min<uint8_t>(static_cast<uint8_t>(EDGE_WAYS - 1), static_cast<uint8_t>(edge.lru + 1));
  }
}

void g_lbtp::train_edge(uint64_t stream_key, uint8_t source_context, uint8_t target_context, int64_t byte_delta)
{
  ++graph.transitions;
  const auto set = edge_set_index(stream_key, source_context);
  auto& ways = edge_table[set];

  for (std::size_t way = 0; way < EDGE_WAYS; ++way) {
    auto& edge = ways[way];
    if (!edge.valid || edge.stream_tag != stream_key || edge.source_context != source_context || edge.target_context != target_context
        || edge.byte_delta != byte_delta)
      continue;

    if (edge.occurrences == std::numeric_limits<uint8_t>::max()) {
      for (auto& peer : ways) {
        if (peer.valid && peer.stream_tag == stream_key && peer.source_context == source_context)
          peer.occurrences = std::max<uint8_t>(1, static_cast<uint8_t>(peer.occurrences / 2));
      }
    }
    edge.occurrences = std::min<uint8_t>(std::numeric_limits<uint8_t>::max(), static_cast<uint8_t>(edge.occurrences + 1));
    edge.confidence = std::min<uint8_t>(7, static_cast<uint8_t>(edge.confidence + 1));
    if (edge.usefulness < 0)
      edge.usefulness = saturating_utility(edge.usefulness, 1);
    ++graph.edge_reinforcements;
    touch_edge(set, way);
    return;
  }

  // Preserve multiple byte-delta modes for one loop transition. A mismatching
  // mode loses confidence, while a free/LRU way learns the new directed edge.
  for (auto& edge : ways) {
    if (edge.valid && edge.stream_tag == stream_key && edge.source_context == source_context && edge.target_context == target_context
        && edge.confidence > 0)
      --edge.confidence;
  }

  std::size_t victim = 0;
  bool found_invalid = false;
  for (std::size_t way = 0; way < EDGE_WAYS; ++way) {
    if (!ways[way].valid) {
      victim = way;
      found_invalid = true;
      break;
    }
    const auto victim_score = edge_score(ways[victim]);
    const auto candidate_score = edge_score(ways[way]);
    if (ways[way].lru > ways[victim].lru || (ways[way].lru == ways[victim].lru && candidate_score < victim_score))
      victim = way;
  }

  if (!found_invalid && ways[victim].valid)
    ++graph.edge_evictions;

  if (++edge_generation == 0)
    edge_generation = 1;
  ways[victim] = graph_edge{stream_key, byte_delta, edge_generation, 1, 0, source_context, target_context, 1, 0, true};
  ++graph.edge_allocations;
  touch_edge(set, victim);
}

g_lbtp::edge_selection g_lbtp::select_edge(uint64_t stream_key, uint8_t source_context, lookup_result& result,
                                           uint8_t required_target)
{
  const auto set = edge_set_index(stream_key, source_context);
  const auto& ways = edge_table[set];
  edge_selection best{};
  uint16_t runner_score = 0;
  bool any_edge = false;
  bool any_confident = false;
  bool have_runner = false;

  for (std::size_t way = 0; way < EDGE_WAYS; ++way) {
    const auto& edge = ways[way];
    if (!edge.valid || edge.stream_tag != stream_key || edge.source_context != source_context
        || (required_target < CONTEXT_COUNT && edge.target_context != required_target))
      continue;
    any_edge = true;
    if (edge.confidence < EDGE_CONFIDENCE_THRESHOLD)
      continue;
    any_confident = true;

    edge_selection candidate{edge.byte_delta, edge.generation, static_cast<uint16_t>(set), edge_score(edge), static_cast<uint8_t>(way),
                             edge.source_context, edge.target_context, edge.confidence, true};
    if (!best.valid || candidate.score > best.score) {
      if (best.valid && best.byte_delta != candidate.byte_delta) {
        runner_score = std::max(runner_score, best.score);
        have_runner = true;
      }
      best = candidate;
    } else if (candidate.byte_delta != best.byte_delta) {
      runner_score = std::max(runner_score, candidate.score);
      have_runner = true;
    }
  }

  if (!any_edge) {
    result = lookup_result::no_edge;
    return {};
  }
  if (!any_confident) {
    result = lookup_result::low_confidence;
    return {};
  }
  if (have_runner && best.score < static_cast<uint16_t>(runner_score + EDGE_SCORE_MARGIN)) {
    result = lookup_result::ambiguous;
    return {};
  }

  result = lookup_result::selected;
  touch_edge(best.set, best.way);
  return best;
}

void g_lbtp::boundary_callback(uint8_t target_context)
{
  if (active_instance != nullptr)
    active_instance->on_loop_boundary(target_context);
}

void g_lbtp::on_loop_boundary(uint8_t target_context)
{
  if (!runtime_context_enabled || target_context == 0 || target_context >= CONTEXT_COUNT)
    return;

  ++graph.boundary_triggers;
  for (const auto& node : stream_table) {
    if (!node.valid || !node.have_last || node.last_context >= CONTEXT_COUNT)
      continue;

    const auto role = static_cast<uint8_t>(node.tag & 0x3ULL);
    if (role >= ROLE_COUNT)
      continue;

    auto& value = stats[node.last_context][role];
    ++value.graph_lookups;
    lookup_result result = lookup_result::no_edge;
    const auto selection = select_edge(node.tag, node.last_context, result, target_context);
    switch (result) {
    case lookup_result::selected:
      ++value.graph_selected;
      issue_candidate(node.last_address, role, selection, 0);
      break;
    case lookup_result::no_edge:
      ++value.graph_no_edge;
      break;
    case lookup_result::low_confidence:
      ++value.graph_low_confidence;
      break;
    case lookup_result::ambiguous:
      ++value.graph_ambiguous;
      break;
    }
  }
}

void g_lbtp::feedback_edge(const pending_entry& prediction, int adjustment)
{
  auto& edge = edge_table[prediction.edge_set][prediction.edge_way];
  if (!edge.valid || edge.generation != prediction.edge_generation || edge.source_context != prediction.source_context
      || edge.target_context != prediction.target_context || edge.byte_delta != prediction.byte_delta) {
    ++graph.stale_feedback;
    return;
  }

  edge.usefulness = saturating_utility(edge.usefulness, adjustment);
  if (adjustment > 0)
    ++graph.positive_feedback;
  else if (adjustment < 0)
    ++graph.negative_feedback;
}

bool g_lbtp::local_tlb_contains(uint64_t vpn) const
{
  return std::any_of(std::cbegin(intern_->block), std::cend(intern_->block),
                     [vpn](const auto& block) { return block.valid && g_lbtp::vpn_from_address(block.address) == vpn; });
}

bool g_lbtp::local_tlb_inflight(uint64_t vpn) const
{
  const auto matches = [vpn](const auto& entry) { return g_lbtp::vpn_from_address(entry.address) == vpn; };
  return std::any_of(std::cbegin(intern_->MSHR), std::cend(intern_->MSHR), matches)
      || std::any_of(std::cbegin(intern_->inflight_fills), std::cend(intern_->inflight_fills), matches);
}

bool g_lbtp::issue_candidate(uint64_t current_address, uint8_t role, const edge_selection& selection, uint32_t metadata_in)
{
  auto& value = stats[selection.source_context][role];
  ++value.candidates;

  uint64_t predicted_address_u64 = 0;
  if (selection.byte_delta == 0 || !add_delta(current_address, selection.byte_delta, predicted_address_u64))
    return false;

  const auto current_vpn = current_address >> LOG2_PAGE_SIZE;
  const auto predicted_vpn = predicted_address_u64 >> LOG2_PAGE_SIZE;
  if (predicted_vpn == current_vpn)
    return false;
  ++value.cross_page;

  if (local_tlb_contains(predicted_vpn)) {
    ++value.filtered_resident;
    return false;
  }
  if (local_tlb_inflight(predicted_vpn)) {
    ++value.filtered_inflight;
    return false;
  }
  if (pending.find(predicted_vpn) != pending.end()) {
    ++value.filtered_pending;
    return false;
  }
  if (pending.size() >= MAX_PENDING) {
    ++value.filtered_capacity;
    return false;
  }

  const auto predicted_address = champsim::address{predicted_vpn << LOG2_PAGE_SIZE};
  if (!prefetch_line(predicted_address, true, metadata_in)) {
    ++value.rejected;
    return false;
  }

  pending_entry prediction{};
  prediction.prediction_id = prediction_seq++;
  prediction.trigger_address = current_address;
  prediction.issue_cycle = current_cycle;
  prediction.issue_seq = demand_seq;
  prediction.byte_delta = selection.byte_delta;
  prediction.edge_generation = selection.generation;
  prediction.edge_set = selection.set;
  prediction.edge_score = selection.score;
  prediction.edge_way = selection.way;
  prediction.role = role;
  prediction.source_context = selection.source_context;
  prediction.target_context = selection.target_context;
  prediction.edge_confidence = selection.confidence;
  pending.emplace(predicted_vpn, prediction);
  ++value.issued;
  return true;
}

void g_lbtp::record_demand_for_prediction(uint64_t vpn, bool cache_hit, bool useful_prefetch)
{
  const auto found = pending.find(vpn);
  if (found == pending.end() || found->second.demand_seen)
    return;

  auto& prediction = found->second;
  auto& value = stats[prediction.source_context][prediction.role];
  const auto issue_to_demand = current_cycle >= prediction.issue_cycle ? current_cycle - prediction.issue_cycle : 0;
  ++value.demanded_after_issue;
  value.issue_to_demand_sum += issue_to_demand;
  value.issue_to_demand_max = std::max(value.issue_to_demand_max, issue_to_demand);

  if (useful_prefetch) {
    ++value.timely;
    if (prediction.fill_seen) {
      const auto ready_lead = current_cycle >= prediction.fill_cycle ? current_cycle - prediction.fill_cycle : 0;
      value.ready_lead_sum += ready_lead;
      value.ready_lead_max = std::max(value.ready_lead_max, ready_lead);
    }
    feedback_edge(prediction, 1);
    write_event(vpn, prediction, "timely");
    pending.erase(found);
    return;
  }

  if (prediction.evicted_before_demand) {
    ++value.too_early;
    prediction.demand_seen = true;
    prediction.demand_cycle = current_cycle;
    feedback_edge(prediction, -1);
    write_event(vpn, prediction, "too_early");
    pending.erase(found);
    return;
  }

  if (cache_hit) {
    ++value.nonuseful_hit;
    prediction.demand_seen = true;
    prediction.demand_cycle = current_cycle;
    feedback_edge(prediction, -1);
    write_event(vpn, prediction, "nonuseful_hit");
    pending.erase(found);
    return;
  }

  prediction.demand_seen = true;
  prediction.demand_cycle = current_cycle;
  ++value.late_at_demand;
}

void g_lbtp::expire_pending()
{
  for (auto current = pending.begin(); current != pending.end();) {
    auto& prediction = current->second;
    const auto distance = demand_seq >= prediction.issue_seq ? demand_seq - prediction.issue_seq : 0;
    if (prediction.demand_seen || distance <= PENDING_MAX_DEMAND_DISTANCE) {
      ++current;
      continue;
    }

    auto& value = stats[prediction.source_context][prediction.role];
    if (prediction.fill_seen)
      ++value.too_early;
    else
      ++value.never_demanded;
    feedback_edge(prediction, -1);
    write_event(current->first, prediction, prediction.fill_seen ? "expired_too_early" : "expired_never");
    current = pending.erase(current);
  }
}

void g_lbtp::prefetcher_initialize()
{
  stream_table = {};
  edge_table = {};
  stats = {};
  graph = {};
  pending.clear();
  current_cycle = 0;
  demand_seq = 0;
  prediction_seq = 0;
  edge_generation = 1;
  ignored_non_pim = 0;
  duplicate_callback = 0;
  missing_runtime_context = 0;
  finalized = false;

  const auto* runtime_setting = std::getenv("GEMM_RUNTIME_LOOP_CONTEXT");
  runtime_context_enabled = runtime_setting == nullptr || std::string_view{runtime_setting} != "0";
  gemm_runtime_loop_context::state.reset();
  active_instance = this;
  gemm_runtime_loop_context::predicted_backedge_observer =
      runtime_context_enabled ? &g_lbtp::boundary_callback : nullptr;

  if (event_log.is_open())
    event_log.close();
  if (const auto* path = std::getenv("GEMM_TLB_EVENT_LOG"); path != nullptr && *path != '\0') {
    event_log.open(path, std::ios::out | std::ios::trunc);
    if (event_log)
      event_log << "prediction_id,role,source_context,target_context,source_loop_pc,target_loop_pc,byte_delta,edge_confidence,"
                   "edge_score,trigger_address,predicted_vpn,issue_cycle,demand_cycle,fill_cycle,outcome,issue_to_demand,"
                   "ready_lead,late_by\n";
  }
}

uint32_t g_lbtp::prefetcher_cache_operate(champsim::address addr, champsim::address full_addr, champsim::address ip, uint8_t cache_hit,
                                          bool useful_prefetch, access_type type, uint64_t instr_id, uint32_t metadata_in)
{
  const bool demand = type == access_type::LOAD || type == access_type::WRITE || type == access_type::RFO;
  if (!demand)
    return metadata_in;

  const auto raw_ip = as_u64(ip);
  if (raw_ip < PIM_PC_BEGIN || raw_ip >= PIM_PC_END) {
    ++ignored_non_pim;
    return metadata_in;
  }

  const auto role = role_from_ip(ip);
  uint8_t context = 0;
  if (runtime_context_enabled) {
    const auto found = gemm_runtime_loop_context::state.context_for(instr_id);
    if (!found.has_value())
      ++missing_runtime_context;
    else
      context = *found;
  }
  if (role >= ROLE_COUNT || context >= CONTEXT_COUNT)
    return metadata_in;

  const auto stream_key = stream_key_from_ip(ip);
  auto& node = stream_table[stream_index(stream_key)];
  if (node.valid && node.tag == stream_key && node.have_instr_id && node.last_instr_id == instr_id) {
    ++duplicate_callback;
    return metadata_in;
  }
  if (!node.valid || node.tag != stream_key) {
    if (node.valid)
      ++graph.stream_collisions;
    node = stream_node{};
    node.tag = stream_key;
    node.valid = true;
  }
  node.last_instr_id = instr_id;
  node.have_instr_id = true;

  ++demand_seq;
  expire_pending();
  auto& value = stats[context][role];
  ++value.demand_access;
  if (!cache_hit)
    ++value.demand_miss;
  if (useful_prefetch)
    ++value.useful_callback;

  // Only the instruction-carried A/B/C base is consumed. addr is page-aligned
  // by the STLB; full_addr retains the base's byte offset for signed byte-delta
  // training.
  const auto raw_address = as_u64(full_addr);
  const auto vpn = vpn_from_address(addr);
  record_demand_for_prediction(vpn, cache_hit != 0, useful_prefetch);

  if (!node.have_last) {
    node.have_last = true;
    node.last_address = raw_address;
    node.last_context = context;
    return metadata_in;
  }

  const auto previous_address = node.last_address;
  const auto previous_context = node.last_context;
  node.last_address = raw_address;
  node.last_context = context;

  int64_t byte_delta = 0;
  if (signed_delta(raw_address, previous_address, byte_delta))
    train_edge(stream_key, previous_context, context, byte_delta);

  // With runtime loop contexts, prediction is triggered at the taken loop
  // boundary, before the following A/B/C requests arrive. Context-disabled
  // mode remains a PC+role-only, base-triggered control experiment.
  if (runtime_context_enabled)
    return metadata_in;

  ++value.graph_lookups;
  lookup_result result = lookup_result::no_edge;
  const auto selection = select_edge(stream_key, context, result);
  switch (result) {
  case lookup_result::selected:
    ++value.graph_selected;
    issue_candidate(raw_address, role, selection, metadata_in);
    break;
  case lookup_result::no_edge:
    ++value.graph_no_edge;
    break;
  case lookup_result::low_confidence:
    ++value.graph_low_confidence;
    break;
  case lookup_result::ambiguous:
    ++value.graph_ambiguous;
    break;
  }

  return metadata_in;
}

uint32_t g_lbtp::prefetcher_cache_fill(champsim::address addr, long, long, uint8_t, champsim::address evicted_addr, uint32_t metadata_in)
{
  if (as_u64(evicted_addr) != 0) {
    const auto evicted_vpn = vpn_from_address(evicted_addr);
    const auto evicted = pending.find(evicted_vpn);
    if (evicted != pending.end() && !evicted->second.demand_seen)
      evicted->second.evicted_before_demand = true;
  }

  const auto vpn = vpn_from_address(addr);
  const auto found = pending.find(vpn);
  if (found == pending.end())
    return metadata_in;

  auto& prediction = found->second;
  prediction.fill_seen = true;
  prediction.fill_cycle = current_cycle;
  if (!prediction.demand_seen)
    return metadata_in;

  const auto late_by = current_cycle >= prediction.demand_cycle ? current_cycle - prediction.demand_cycle : 0;
  auto& value = stats[prediction.source_context][prediction.role];
  ++value.late_completed;
  value.late_by_sum += late_by;
  value.late_by_max = std::max(value.late_by_max, late_by);
  feedback_edge(prediction, 1);
  write_event(vpn, prediction, "late");
  pending.erase(found);
  return metadata_in;
}

void g_lbtp::prefetcher_cycle_operate() { ++current_cycle; }

void g_lbtp::write_event(uint64_t vpn, const pending_entry& prediction, std::string_view outcome)
{
  if (!event_log)
    return;

  const auto issue_to_demand = prediction.demand_seen && prediction.demand_cycle >= prediction.issue_cycle
      ? prediction.demand_cycle - prediction.issue_cycle
      : 0;
  const auto ready_lead = prediction.demand_seen && prediction.fill_seen && prediction.demand_cycle >= prediction.fill_cycle
      ? prediction.demand_cycle - prediction.fill_cycle
      : 0;
  const auto late_by = prediction.demand_seen && prediction.fill_seen && prediction.fill_cycle >= prediction.demand_cycle
      ? prediction.fill_cycle - prediction.demand_cycle
      : 0;
  event_log << prediction.prediction_id << ',' << role_name(prediction.role) << ',' << context_name(prediction.source_context) << ','
            << context_name(prediction.target_context) << ',' << context_pc(prediction.source_context) << ','
            << context_pc(prediction.target_context) << ',' << prediction.byte_delta << ',' << static_cast<unsigned>(prediction.edge_confidence)
            << ',' << prediction.edge_score << ',' << prediction.trigger_address << ',' << vpn << ',' << prediction.issue_cycle << ','
            << prediction.demand_cycle << ',' << prediction.fill_cycle << ',' << outcome << ',' << issue_to_demand << ',' << ready_lead << ','
            << late_by << '\n';
}

void g_lbtp::finalize_unresolved()
{
  if (finalized)
    return;
  for (const auto& [vpn, prediction] : pending) {
    auto& value = stats[prediction.source_context][prediction.role];
    if (prediction.demand_seen)
      ++value.unresolved_late;
    else
      ++value.never_demanded;
    write_event(vpn, prediction, prediction.demand_seen ? "unresolved_late" : "never_demanded");
  }
  if (event_log)
    event_log.flush();
  finalized = true;
}

void g_lbtp::prefetcher_final_stats()
{
  finalize_unresolved();
  const auto valid_edges = std::count_if(edge_table.begin(), edge_table.end(), [](const auto& set) {
    return std::any_of(set.begin(), set.end(), [](const auto& edge) { return edge.valid; });
  });
  const auto exact_valid_edges = std::accumulate(edge_table.begin(), edge_table.end(), uint64_t{0}, [](uint64_t total, const auto& set) {
    return total + static_cast<uint64_t>(std::count_if(set.begin(), set.end(), [](const auto& edge) { return edge.valid; }));
  });

  fmt::print(
      "g_lbtp_v1 runtime_context:{} ignored_non_pim:{} duplicate_callback:{} missing_context:{} outstanding:{} cycles:{} "
      "predicted_backedges:{} actual_backedges:{} correct_backedges:{} missed_backedges:{} false_backedges:{} context_overflow:{}\n",
      runtime_context_enabled, ignored_non_pim, duplicate_callback, missing_runtime_context, pending.size(), current_cycle,
      gemm_runtime_loop_context::state.predicted_backedges, gemm_runtime_loop_context::state.actual_backedges,
      gemm_runtime_loop_context::state.correctly_predicted_backedges, gemm_runtime_loop_context::state.missed_backedges,
      gemm_runtime_loop_context::state.false_backedges, gemm_runtime_loop_context::state.context_overflow);
  fmt::print(
      "g_lbtp_v1 graph boundary_triggers:{} transitions:{} edge_allocations:{} edge_reinforcements:{} edge_evictions:{} valid_edge_sets:{} valid_edges:{} "
      "edge_capacity:{} stream_collisions:{} positive_feedback:{} negative_feedback:{} stale_feedback:{} pending_capacity:{}\n",
      graph.boundary_triggers, graph.transitions, graph.edge_allocations, graph.edge_reinforcements, graph.edge_evictions, valid_edges, exact_valid_edges,
      EDGE_SETS * EDGE_WAYS, graph.stream_collisions, graph.positive_feedback, graph.negative_feedback, graph.stale_feedback, MAX_PENDING);

  for (uint8_t context = 1; context < gemm_runtime_loop_context::state.next_context; ++context)
    fmt::print("g_lbtp_v1 context {} branch_pc:0x{:x}\n", context_name(context),
               gemm_runtime_loop_context::state.context_branch_pc[context]);

  for (uint8_t context = 0; context < CONTEXT_COUNT; ++context) {
    for (uint8_t role = 0; role < ROLE_COUNT; ++role) {
      const auto& value = stats[context][role];
      if (value.demand_access == 0 && value.graph_lookups == 0 && value.issued == 0)
        continue;
      fmt::print(
          "g_lbtp_v1 source {} role {} access:{} miss:{} useful_callback:{} graph_lookup:{} no_edge:{} low_confidence:{} ambiguous:{} "
          "selected:{} candidate:{} cross_page:{} same_page:{} resident_filter:{} inflight_filter:{} pending_filter:{} capacity_filter:{} "
          "issued:{} rejected:{} demanded:{} timely:{} late:{} late_completed:{} nonuseful_hit:{} too_early:{} never:{} unresolved_late:{} "
          "issue_to_demand_avg:{:.2f} issue_to_demand_max:{} ready_lead_avg:{:.2f} ready_lead_max:{} late_by_avg:{:.2f} late_by_max:{}\n",
          context_name(context), role_name(role), value.demand_access, value.demand_miss, value.useful_callback, value.graph_lookups,
          value.graph_no_edge, value.graph_low_confidence, value.graph_ambiguous, value.graph_selected, value.candidates, value.cross_page,
          value.candidates - value.cross_page, value.filtered_resident, value.filtered_inflight, value.filtered_pending, value.filtered_capacity,
          value.issued, value.rejected, value.demanded_after_issue, value.timely, value.late_at_demand, value.late_completed,
          value.nonuseful_hit, value.too_early, value.never_demanded, value.unresolved_late,
          average(value.issue_to_demand_sum, value.demanded_after_issue), value.issue_to_demand_max,
          average(value.ready_lead_sum, value.timely), value.ready_lead_max, average(value.late_by_sum, value.late_completed),
          value.late_by_max);
    }
  }

  if (active_instance == this) {
    gemm_runtime_loop_context::predicted_backedge_observer = nullptr;
    active_instance = nullptr;
  }
}
