#include "star_tlb.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

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

std::vector<std::string> split_csv(const std::string& line)
{
  std::vector<std::string> result;
  std::string field;
  bool quoted = false;
  for (char ch : line) {
    if (ch == '"') {
      quoted = !quoted;
    } else if (ch == ',' && !quoted) {
      result.push_back(field);
      field.clear();
    } else {
      field.push_back(ch);
    }
  }
  result.push_back(field);
  return result;
}

uint64_t parse_u64(const std::string& value)
{
  std::size_t consumed = 0;
  const auto result = std::stoull(value, &consumed, 0);
  if (consumed != value.size())
    throw std::invalid_argument{"trailing characters"};
  return result;
}

double average(uint64_t sum, uint64_t count) { return count == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(count); }
} // namespace

uint8_t star_tlb::role_from_ip(champsim::address ip) { return static_cast<uint8_t>(as_u64(ip) & 0x3ULL); }

uint64_t star_tlb::site_key_from_ip(champsim::address ip) { return as_u64(ip) & ~0x3ULL; }

uint64_t star_tlb::vpn_from_address(champsim::address addr) { return as_u64(champsim::page_number{addr}); }

std::string_view star_tlb::role_name(uint8_t role)
{
  constexpr std::array<std::string_view, ROLE_COUNT> names{"A", "B", "C"};
  return role < names.size() ? names[role] : "unknown";
}

std::string_view star_tlb::context_name(uint8_t context)
{
  constexpr std::array<std::string_view, CONTEXT_COUNT> names{
      "NO_BACKEDGE", "K_PROGRESS", "K_TO_IR", "IR_TO_JR", "JR_TO_IC", "IC_TO_PC", "PC_TO_JC"};
  return context < names.size() ? names[context] : "unknown";
}

bool star_tlb::signed_delta(uint64_t newer, uint64_t older, int64_t& result)
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

bool star_tlb::add_delta(uint64_t address, int64_t delta, uint64_t& result)
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

int8_t star_tlb::saturating_utility(int8_t value, int adjustment)
{
  return static_cast<int8_t>(std::clamp<int>(static_cast<int>(value) + adjustment, -4, 3));
}

uint64_t star_tlb::ema(uint64_t old_value, uint64_t sample)
{
  sample = std::max<uint64_t>(1, sample);
  return std::max<uint64_t>(1, (7 * old_value + sample) / 8);
}

bool star_tlb::load_descriptors(const char* path)
{
  std::ifstream input{path};
  if (!input) {
    fmt::print("star_tlb_v1 error unable_to_open_descriptor_csv:{}\n", path);
    return false;
  }

  std::string line;
  if (!std::getline(input, line))
    return false;
  auto header = split_csv(line);
  if (!header.empty() && header.front().size() >= 3 && static_cast<unsigned char>(header.front()[0]) == 0xef
      && static_cast<unsigned char>(header.front()[1]) == 0xbb && static_cast<unsigned char>(header.front()[2]) == 0xbf)
    header.front().erase(0, 3);

  std::unordered_map<std::string, std::size_t> column;
  for (std::size_t index = 0; index < header.size(); ++index)
    column.emplace(header[index], index);

  const char* pc_column = column.find("pim_pc") != column.end()
      ? "pim_pc"
      : (column.find("pim_site_pc") != column.end() ? "pim_site_pc" : nullptr);
  if (pc_column == nullptr) {
    fmt::print("star_tlb_v1 error missing_descriptor_column:pim_pc_or_pim_site_pc\n");
    return false;
  }
  const std::array required{"a_tile_base", "b_tile_base", "c_tile_base", "a_row_stride_bytes",
                            "b_row_stride_bytes", "c_row_stride_bytes", "valid_m", "valid_n", "valid_k", "flags"};
  for (const auto* name : required) {
    if (column.find(name) == column.end()) {
      fmt::print("star_tlb_v1 error missing_descriptor_column:{}\n", name);
      return false;
    }
  }

  std::size_t row = 1;
  while (std::getline(input, line)) {
    ++row;
    if (line.empty())
      continue;
    const auto values = split_csv(line);
    const auto value = [&](const char* name) -> const std::string& {
      const auto index = column.at(name);
      if (index >= values.size())
        throw std::out_of_range{"short CSV row"};
      return values[index];
    };
    try {
      descriptor desc{};
      desc.raw_site_pc = parse_u64(value(pc_column));
      desc.a_base = parse_u64(value("a_tile_base"));
      desc.b_base = parse_u64(value("b_tile_base"));
      desc.c_base = parse_u64(value("c_tile_base"));
      desc.lda = parse_u64(value("a_row_stride_bytes"));
      desc.ldb = parse_u64(value("b_row_stride_bytes"));
      desc.ldc = parse_u64(value("c_row_stride_bytes"));
      desc.valid_m = static_cast<uint8_t>(parse_u64(value("valid_m")));
      desc.valid_n = static_cast<uint8_t>(parse_u64(value("valid_n")));
      desc.valid_k = static_cast<uint8_t>(parse_u64(value("valid_k")));
      desc.flags = static_cast<uint8_t>(parse_u64(value("flags")));
      if (desc.lda == 0 || desc.ldb == 0 || desc.ldc == 0 || desc.valid_m == 0 || desc.valid_m > 32 || desc.valid_n == 0
          || desc.valid_n > 32 || desc.valid_k == 0 || desc.valid_k > 32)
        throw std::invalid_argument{"invalid PIM descriptor"};
      descriptors.push_back(desc);
    } catch (const std::exception& error) {
      fmt::print("star_tlb_v1 error descriptor_row:{} reason:{}\n", row, error.what());
      return false;
    }
  }
  return !descriptors.empty();
}

std::size_t star_tlb::edge_set_index(uint64_t site_tag, uint8_t source_context) const
{
  const auto mixed = site_tag ^ (site_tag >> 17) ^ (static_cast<uint64_t>(source_context + 1) * 0x9e3779b97f4a7c15ULL);
  return static_cast<std::size_t>(mixed) & (EDGE_SETS - 1);
}

uint16_t star_tlb::edge_score(const graph_edge& edge) const
{
  const auto utility = static_cast<uint16_t>(static_cast<int>(edge.usefulness) + 4);
  return static_cast<uint16_t>(static_cast<uint16_t>(edge.confidence) * 64U + edge.occurrences + utility * 8U);
}

void star_tlb::touch_edge(std::size_t set, std::size_t way)
{
  for (std::size_t index = 0; index < EDGE_WAYS; ++index) {
    auto& edge = edge_table[set][index];
    if (!edge.valid)
      continue;
    edge.lru = index == way ? 0 : std::min<uint8_t>(EDGE_WAYS - 1, static_cast<uint8_t>(edge.lru + 1));
  }
}

void star_tlb::train_edge(uint64_t site_tag, uint8_t source_context, uint8_t target_context, const descriptor& previous,
                          const descriptor& current)
{
  std::array<int64_t, ROLE_COUNT> delta{};
  if (!signed_delta(current.a_base, previous.a_base, delta[ROLE_A])
      || !signed_delta(current.b_base, previous.b_base, delta[ROLE_B])
      || !signed_delta(current.c_base, previous.c_base, delta[ROLE_C]))
    return;

  ++graph.transitions;
  const auto set = edge_set_index(site_tag, source_context);
  auto& ways = edge_table[set];
  const auto same_mode = [&](const graph_edge& edge) {
    return edge.valid && edge.site_tag == site_tag && edge.source_context == source_context && edge.target_context == target_context
        && edge.byte_delta == delta && edge.lda == current.lda && edge.ldb == current.ldb && edge.ldc == current.ldc
        && edge.valid_m == current.valid_m && edge.valid_n == current.valid_n && edge.valid_k == current.valid_k;
  };
  for (std::size_t way = 0; way < EDGE_WAYS; ++way) {
    auto& edge = ways[way];
    if (!same_mode(edge))
      continue;
    edge.occurrences = std::min<uint8_t>(255, static_cast<uint8_t>(edge.occurrences + 1));
    edge.confidence = std::min<uint8_t>(7, static_cast<uint8_t>(edge.confidence + 1));
    ++graph.edge_reinforcements;
    touch_edge(set, way);
    return;
  }

  for (auto& edge : ways) {
    if (edge.valid && edge.site_tag == site_tag && edge.source_context == source_context && edge.target_context == target_context
        && edge.confidence > 0)
      --edge.confidence;
  }

  std::size_t victim = 0;
  for (std::size_t way = 0; way < EDGE_WAYS; ++way) {
    if (!ways[way].valid) {
      victim = way;
      break;
    }
    if (ways[way].lru > ways[victim].lru
        || (ways[way].lru == ways[victim].lru && edge_score(ways[way]) < edge_score(ways[victim])))
      victim = way;
  }
  if (ways[victim].valid)
    ++graph.edge_evictions;
  if (++edge_generation == 0)
    edge_generation = 1;

  auto& edge = ways[victim];
  edge = {};
  edge.site_tag = site_tag;
  edge.byte_delta = delta;
  edge.lda = current.lda;
  edge.ldb = current.ldb;
  edge.ldc = current.ldc;
  edge.generation = edge_generation;
  edge.occurrences = 1;
  edge.source_context = source_context;
  edge.target_context = target_context;
  edge.confidence = 1;
  edge.valid_m = current.valid_m;
  edge.valid_n = current.valid_n;
  edge.valid_k = current.valid_k;
  edge.flags = current.flags;
  edge.valid = true;
  ++graph.edge_allocations;
  touch_edge(set, victim);
}

star_tlb::edge_selection star_tlb::select_edge(uint64_t site_tag, uint8_t source_context, uint8_t required_target)
{
  const auto set = edge_set_index(site_tag, source_context);
  const auto& ways = edge_table[set];
  edge_selection best{};
  uint16_t runner_score = 0;
  bool any = false;
  bool confident = false;
  bool runner = false;

  for (std::size_t way = 0; way < EDGE_WAYS; ++way) {
    const auto& edge = ways[way];
    if (!edge.valid || edge.site_tag != site_tag || edge.source_context != source_context
        || (required_target != ANY_CONTEXT && edge.target_context != required_target))
      continue;
    any = true;
    if (edge.confidence < EDGE_CONFIDENCE_THRESHOLD)
      continue;
    confident = true;
    const auto score = edge_score(edge);
    if (!best.valid || score > best.score) {
      if (best.valid && best.byte_delta != edge.byte_delta) {
        runner_score = std::max(runner_score, best.score);
        runner = true;
      }
      best.byte_delta = edge.byte_delta;
      best.lda = edge.lda;
      best.ldb = edge.ldb;
      best.ldc = edge.ldc;
      best.generation = edge.generation;
      best.set = static_cast<uint16_t>(set);
      best.score = score;
      best.way = static_cast<uint8_t>(way);
      best.source_context = edge.source_context;
      best.target_context = edge.target_context;
      best.confidence = edge.confidence;
      best.valid_m = edge.valid_m;
      best.valid_n = edge.valid_n;
      best.valid_k = edge.valid_k;
      best.flags = edge.flags;
      best.valid = true;
    } else if (best.byte_delta != edge.byte_delta) {
      runner_score = std::max(runner_score, score);
      runner = true;
    }
  }

  if (!any) {
    ++graph.no_edge;
    return {};
  }
  if (!confident) {
    ++graph.low_confidence;
    return {};
  }
  if (runner && best.score < static_cast<uint16_t>(runner_score + EDGE_SCORE_MARGIN)) {
    ++graph.ambiguous;
    return {};
  }
  ++graph.edge_selected;
  touch_edge(best.set, best.way);
  return best;
}

void star_tlb::feedback_edge(const pending_entry& prediction, int adjustment)
{
  auto& edge = edge_table[prediction.edge_set][prediction.edge_way];
  if (!edge.valid || edge.generation != prediction.edge_generation || edge.source_context != prediction.source_context
      || edge.target_context != prediction.target_context) {
    ++graph.stale_feedback;
    return;
  }
  edge.usefulness = saturating_utility(edge.usefulness, adjustment);
  if (adjustment > 0)
    ++graph.positive_feedback;
  else if (adjustment < 0)
    ++graph.negative_feedback;
}

void star_tlb::boundary_callback(uint8_t target_context)
{
  if (active_instance != nullptr)
    active_instance->on_loop_boundary(target_context);
}

std::optional<star_tlb::descriptor> star_tlb::apply_edge(const descriptor& current, const edge_selection& edge) const
{
  descriptor predicted = current;
  if (!add_delta(current.a_base, edge.byte_delta[ROLE_A], predicted.a_base)
      || !add_delta(current.b_base, edge.byte_delta[ROLE_B], predicted.b_base)
      || !add_delta(current.c_base, edge.byte_delta[ROLE_C], predicted.c_base))
    return std::nullopt;
  predicted.lda = edge.lda;
  predicted.ldb = edge.ldb;
  predicted.ldc = edge.ldc;
  predicted.valid_m = edge.valid_m;
  predicted.valid_n = edge.valid_n;
  predicted.valid_k = edge.valid_k;
  predicted.flags = edge.flags;
  return predicted;
}

uint8_t star_tlb::lookahead_distance() const
{
  const auto intervals = (walk_latency_ema + pim_interval_ema - 1) / pim_interval_ema;
  // A loop boundary is observed immediately before the next PIM descriptor, so
  // distance 1 offers only frontend-to-PIM lead time. Add one full descriptor.
  return static_cast<uint8_t>(std::clamp<uint64_t>(intervals + 1, 2, MAX_LOOKAHEAD));
}

void star_tlb::on_loop_boundary(uint8_t target_context)
{
  if (!have_last_descriptor || target_context == 0 || target_context >= CONTEXT_COUNT)
    return;
  ++graph.boundary_triggers;
  ++graph.prediction_chains;

  descriptor predicted = last_descriptor;
  uint8_t source_context = last_context;
  const auto horizon = lookahead_distance();
  for (uint8_t distance = 1; distance <= horizon; ++distance) {
    const auto required = distance == 1 ? target_context : ANY_CONTEXT;
    const auto edge = select_edge(last_site_tag, source_context, required);
    if (!edge.valid)
      break;
    const auto next = apply_edge(predicted, edge);
    if (!next.has_value())
      break;
    predicted = *next;
    enqueue_footprint(predicted, distance, edge);
    source_context = edge.target_context;
    ++graph.predicted_descriptors;
  }
}

void star_tlb::observe_descriptor(uint64_t site_tag, uint8_t context, const descriptor& current)
{
  ++graph.descriptors_seen;
  if (have_last_descriptor && last_site_tag == site_tag)
    train_edge(site_tag, last_context, context, last_descriptor, current);

  if (have_last_descriptor && current_cycle > last_descriptor_cycle)
    pim_interval_ema = ema(pim_interval_ema, current_cycle - last_descriptor_cycle);
  last_descriptor = current;
  last_site_tag = site_tag;
  last_context = context;
  last_descriptor_cycle = current_cycle;
  have_last_descriptor = true;
}

std::vector<uint64_t> star_tlb::footprint(const descriptor& desc, uint8_t role) const
{
  uint64_t base = 0;
  uint64_t stride = 0;
  uint64_t width = 0;
  uint8_t rows = 0;
  switch (role) {
  case ROLE_A:
    base = desc.a_base;
    stride = desc.lda;
    rows = desc.valid_m;
    width = static_cast<uint64_t>(desc.valid_k) * 2;
    break;
  case ROLE_B:
    base = desc.b_base;
    stride = desc.ldb;
    rows = desc.valid_k;
    width = static_cast<uint64_t>(desc.valid_n) * 2;
    break;
  case ROLE_C:
    base = desc.c_base;
    stride = desc.ldc;
    rows = desc.valid_m;
    width = static_cast<uint64_t>(desc.valid_n) * 4;
    break;
  default:
    return {};
  }

  std::vector<uint64_t> pages;
  std::unordered_set<uint64_t> seen;
  for (uint64_t row = 0; row < rows; ++row) {
    if (row != 0 && stride > (std::numeric_limits<uint64_t>::max() - base) / row)
      break;
    const auto begin = base + row * stride;
    if (width == 0 || begin > std::numeric_limits<uint64_t>::max() - (width - 1))
      break;
    const auto first = begin >> LOG2_PAGE_SIZE;
    const auto last = (begin + width - 1) >> LOG2_PAGE_SIZE;
    for (uint64_t vpn = first; vpn <= last; ++vpn) {
      if (seen.insert(vpn).second)
        pages.push_back(vpn);
      if (vpn == std::numeric_limits<uint64_t>::max())
        break;
    }
  }
  return pages;
}

void star_tlb::enqueue_footprint(const descriptor& predicted, uint8_t distance, const edge_selection& edge)
{
  for (uint8_t role = 0; role < ROLE_COUNT; ++role) {
    const auto pages = footprint(predicted, role);
    stats[role].footprint_pages += pages.size();
    for (const auto vpn : pages)
      enqueue_vpn(vpn, role, distance, edge);
  }
}

bool star_tlb::local_tlb_contains(uint64_t vpn) const
{
  return std::any_of(std::cbegin(intern_->block), std::cend(intern_->block),
                     [vpn](const auto& block) { return block.valid && star_tlb::vpn_from_address(block.address) == vpn; });
}

bool star_tlb::local_tlb_inflight(uint64_t vpn) const
{
  const auto matches = [vpn](const auto& entry) { return star_tlb::vpn_from_address(entry.address) == vpn; };
  return std::any_of(std::cbegin(intern_->MSHR), std::cend(intern_->MSHR), matches)
      || std::any_of(std::cbegin(intern_->inflight_fills), std::cend(intern_->inflight_fills), matches);
}

void star_tlb::enqueue_vpn(uint64_t vpn, uint8_t role, uint8_t distance, const edge_selection& edge)
{
  auto& value = stats[role];
  ++value.candidates;
  if (local_tlb_contains(vpn)) {
    ++value.filtered_resident;
    ++value.footprint_reused;
    return;
  }
  if (local_tlb_inflight(vpn)) {
    ++value.filtered_inflight;
    ++value.footprint_reused;
    return;
  }
  if (pending.find(vpn) != pending.end()) {
    ++value.filtered_pending;
    ++value.footprint_reused;
    return;
  }

  const auto target_cycle = current_cycle + FRONTEND_TO_PIM_CYCLES + static_cast<uint64_t>(distance - 1) * pim_interval_ema;
  const auto ready_cycle = target_cycle > walk_latency_ema ? target_cycle - walk_latency_ema : current_cycle;
  const auto found = candidates.find(vpn);
  if (found != candidates.end()) {
    ++value.candidate_merged;
    ++value.footprint_reused;
    auto& existing = found->second;
    existing.reuse_count = std::min<uint8_t>(15, static_cast<uint8_t>(existing.reuse_count + 1));
    if (target_cycle < existing.target_cycle) {
      existing.target_cycle = target_cycle;
      existing.ready_cycle = ready_cycle;
      existing.distance = distance;
      existing.role = role;
    }
    return;
  }
  if (candidates.size() >= MAX_CANDIDATES) {
    ++value.filtered_capacity;
    return;
  }

  candidate_entry candidate{};
  candidate.vpn = vpn;
  candidate.created_cycle = current_cycle;
  candidate.ready_cycle = ready_cycle;
  candidate.target_cycle = target_cycle;
  candidate.edge_generation = edge.generation;
  candidate.edge_set = edge.set;
  candidate.edge_score = edge.score;
  candidate.edge_way = edge.way;
  candidate.role = role;
  candidate.distance = distance;
  candidate.source_context = edge.source_context;
  candidate.target_context = edge.target_context;
  candidate.edge_confidence = edge.confidence;
  candidates.emplace(vpn, candidate);
}

bool star_tlb::issue_one_candidate(uint32_t metadata_in)
{
  if (pending.size() >= MAX_PENDING || candidates.empty())
    return false;

  auto best = candidates.end();
  for (auto current = candidates.begin(); current != candidates.end(); ++current) {
    auto& candidate = current->second;
    if (candidate.ready_cycle > current_cycle)
      continue;
    if (local_tlb_contains(candidate.vpn) || local_tlb_inflight(candidate.vpn) || pending.find(candidate.vpn) != pending.end())
      continue;
    if (best == candidates.end() || candidate.target_cycle < best->second.target_cycle
        || (candidate.target_cycle == best->second.target_cycle && candidate.reuse_count > best->second.reuse_count)
        || (candidate.target_cycle == best->second.target_cycle && candidate.reuse_count == best->second.reuse_count
            && candidate.edge_score > best->second.edge_score))
      best = current;
  }
  if (best == candidates.end())
    return false;

  const auto candidate = best->second;
  const auto address = champsim::address{candidate.vpn << LOG2_PAGE_SIZE};
  const auto metadata = metadata_in | 0xa0000000U | (static_cast<uint32_t>(candidate.role) << 20)
      | (static_cast<uint32_t>(candidate.distance) << 16);
  if (!prefetch_line(address, true, metadata)) {
    ++stats[candidate.role].rejected;
    return false;
  }

  pending_entry prediction{};
  static_cast<candidate_entry&>(prediction) = candidate;
  prediction.prediction_id = prediction_seq++;
  prediction.issue_cycle = current_cycle;
  pending.emplace(candidate.vpn, prediction);
  candidates.erase(best);
  ++stats[candidate.role].issued;
  return true;
}

void star_tlb::record_demand(uint64_t vpn, bool cache_hit, bool useful_prefetch)
{
  const auto found = pending.find(vpn);
  if (found == pending.end() || found->second.demand_seen)
    return;
  auto& prediction = found->second;
  auto& value = stats[prediction.role];
  prediction.demand_seen = true;
  prediction.demand_cycle = current_cycle;
  ++value.demanded_after_issue;
  value.issue_to_demand_sum += current_cycle >= prediction.issue_cycle ? current_cycle - prediction.issue_cycle : 0;

  if (useful_prefetch) {
    ++value.timely;
    if (prediction.fill_seen)
      value.ready_lead_sum += current_cycle >= prediction.fill_cycle ? current_cycle - prediction.fill_cycle : 0;
    feedback_edge(prediction, 1);
    write_event(vpn, prediction, "timely");
    pending.erase(found);
  } else if (prediction.evicted_before_demand) {
    ++value.too_early;
    feedback_edge(prediction, -1);
    write_event(vpn, prediction, "too_early");
    pending.erase(found);
  } else if (cache_hit) {
    ++value.redundant;
    feedback_edge(prediction, -1);
    write_event(vpn, prediction, "redundant");
    pending.erase(found);
  } else {
    ++value.late;
  }
}

void star_tlb::expire_entries()
{
  for (auto current = candidates.begin(); current != candidates.end();) {
    const auto expiry = current->second.target_cycle + 4 * pim_interval_ema;
    if (current_cycle <= expiry) {
      ++current;
      continue;
    }
    current = candidates.erase(current);
  }

  for (auto current = pending.begin(); current != pending.end();) {
    const auto expiry = current->second.target_cycle + 8 * pim_interval_ema;
    if (current_cycle <= expiry) {
      ++current;
      continue;
    }
    auto& value = stats[current->second.role];
    if (current->second.demand_seen)
      ++value.unresolved_late;
    else if (current->second.fill_seen)
      ++value.too_early;
    else
      ++value.never_demanded;
    feedback_edge(current->second, -1);
    write_event(current->first, current->second, current->second.demand_seen ? "expired_late" : "expired_unused");
    current = pending.erase(current);
  }
}

void star_tlb::prefetcher_initialize()
{
  descriptors.clear();
  descriptor_cursor = 0;
  edge_table = {};
  candidates.clear();
  pending.clear();
  stats = {};
  graph = {};
  current_cycle = 0;
  demand_seq = 0;
  prediction_seq = 0;
  pim_interval_ema = 128;
  walk_latency_ema = 256;
  edge_generation = 1;
  ignored_non_pim = 0;
  missing_runtime_context = 0;
  have_last_descriptor = false;
  finalized = false;

  const auto* descriptor_path = std::getenv("STAR_TLB_DESCRIPTOR_CSV");
  descriptor_sideband_ready = descriptor_path != nullptr && *descriptor_path != '\0' && load_descriptors(descriptor_path);
  gemm_runtime_loop_context::state.reset();
  active_instance = this;
  gemm_runtime_loop_context::predicted_backedge_observer = &star_tlb::boundary_callback;

  if (event_log.is_open())
    event_log.close();
  if (const auto* path = std::getenv("STAR_TLB_EVENT_LOG"); path != nullptr && *path != '\0') {
    event_log.open(path, std::ios::out | std::ios::trunc);
    if (event_log)
      event_log << "prediction_id,role,source_context,target_context,distance,reuse_count,edge_confidence,edge_score,vpn,"
                   "created_cycle,ready_cycle,target_cycle,issue_cycle,demand_cycle,fill_cycle,outcome,issue_to_demand,"
                   "ready_lead,late_by\n";
  }
  fmt::print("star_tlb_v1 initialize descriptors:{} sideband_ready:{} interval_seed:{} walk_seed:{} max_lookahead:{}\n",
             descriptors.size(), descriptor_sideband_ready, pim_interval_ema, walk_latency_ema, MAX_LOOKAHEAD);
}

uint32_t star_tlb::prefetcher_cache_operate(champsim::address addr, champsim::address full_addr, champsim::address ip, uint8_t cache_hit,
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
  if (role >= ROLE_COUNT)
    return metadata_in;
  ++demand_seq;
  ++stats[role].demand_access;
  if (!cache_hit)
    ++stats[role].demand_miss;
  const auto demand_vpn = vpn_from_address(addr);
  record_demand(demand_vpn, cache_hit != 0, useful_prefetch);
  candidates.erase(demand_vpn);

  uint8_t context = 0;
  const auto found_context = gemm_runtime_loop_context::state.context_for(instr_id);
  if (!found_context.has_value())
    ++missing_runtime_context;
  else
    context = *found_context;

  if (descriptor_sideband_ready && role == ROLE_A && descriptor_cursor < descriptors.size()) {
    const auto raw_address = as_u64(full_addr);
    const auto& expected = descriptors[descriptor_cursor];
    if (raw_address == expected.a_base) {
      observe_descriptor(site_key_from_ip(ip), context, expected);
      ++descriptor_cursor;
    } else {
      // Only the first A access of a record is the architectural base. Other
      // A accesses are exact footprint probes and must not consume descriptors.
      const auto page_matches = (raw_address >> LOG2_PAGE_SIZE) == (expected.a_base >> LOG2_PAGE_SIZE);
      if (page_matches && raw_address != expected.a_base)
        ++graph.descriptor_mismatch;
    }
  }
  return metadata_in;
}

uint32_t star_tlb::prefetcher_cache_fill(champsim::address addr, long, long, uint8_t, champsim::address evicted_addr,
                                         uint32_t metadata_in)
{
  if (as_u64(evicted_addr) != 0) {
    const auto evicted = pending.find(vpn_from_address(evicted_addr));
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
  if (current_cycle > prediction.issue_cycle)
    walk_latency_ema = ema(walk_latency_ema, current_cycle - prediction.issue_cycle);
  if (!prediction.demand_seen)
    return metadata_in;

  auto& value = stats[prediction.role];
  ++value.late_completed;
  value.late_by_sum += current_cycle >= prediction.demand_cycle ? current_cycle - prediction.demand_cycle : 0;
  feedback_edge(prediction, 1);
  write_event(vpn, prediction, "late");
  pending.erase(found);
  return metadata_in;
}

void star_tlb::prefetcher_cycle_operate()
{
  ++current_cycle;
  expire_entries();
  issue_one_candidate();
}

void star_tlb::write_event(uint64_t vpn, const pending_entry& prediction, std::string_view outcome)
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
            << context_name(prediction.target_context) << ',' << static_cast<unsigned>(prediction.distance) << ','
            << static_cast<unsigned>(prediction.reuse_count) << ',' << static_cast<unsigned>(prediction.edge_confidence) << ','
            << prediction.edge_score << ',' << vpn << ',' << prediction.created_cycle << ',' << prediction.ready_cycle << ','
            << prediction.target_cycle << ',' << prediction.issue_cycle << ',' << prediction.demand_cycle << ',' << prediction.fill_cycle << ','
            << outcome << ',' << issue_to_demand << ',' << ready_lead << ',' << late_by << '\n';
}

void star_tlb::finalize_unresolved()
{
  if (finalized)
    return;
  for (const auto& [vpn, prediction] : pending) {
    auto& value = stats[prediction.role];
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

void star_tlb::prefetcher_final_stats()
{
  finalize_unresolved();
  uint64_t valid_edges = 0;
  for (const auto& set : edge_table)
    valid_edges += std::count_if(set.begin(), set.end(), [](const auto& edge) { return edge.valid; });

  fmt::print(
      "star_tlb_v1 graph descriptors_loaded:{} descriptors_seen:{} descriptor_cursor:{} descriptor_mismatch:{} boundary_triggers:{} "
      "transitions:{} edge_allocations:{} edge_reinforcements:{} edge_evictions:{} valid_edges:{} edge_capacity:{} selected:{} "
      "no_edge:{} low_confidence:{} ambiguous:{} prediction_chains:{} predicted_descriptors:{} positive_feedback:{} "
      "negative_feedback:{} stale_feedback:{} candidate_outstanding:{} pending_outstanding:{} interval_ema:{} walk_latency_ema:{} "
      "lookahead:{} ignored_non_pim:{} missing_context:{}\n",
      descriptors.size(), graph.descriptors_seen, descriptor_cursor, graph.descriptor_mismatch, graph.boundary_triggers, graph.transitions,
      graph.edge_allocations, graph.edge_reinforcements, graph.edge_evictions, valid_edges, EDGE_SETS * EDGE_WAYS, graph.edge_selected,
      graph.no_edge, graph.low_confidence, graph.ambiguous, graph.prediction_chains, graph.predicted_descriptors, graph.positive_feedback,
      graph.negative_feedback, graph.stale_feedback, candidates.size(), pending.size(), pim_interval_ema, walk_latency_ema,
      static_cast<unsigned>(lookahead_distance()), ignored_non_pim, missing_runtime_context);

  for (uint8_t role = 0; role < ROLE_COUNT; ++role) {
    const auto& value = stats[role];
    fmt::print(
        "star_tlb_v1 role {} access:{} miss:{} footprint_pages:{} footprint_reused:{} candidate:{} candidate_merged:{} "
        "resident_filter:{} inflight_filter:{} pending_filter:{} capacity_filter:{} issued:{} rejected:{} demanded:{} timely:{} "
        "late:{} late_completed:{} redundant:{} too_early:{} never:{} unresolved_late:{} issue_to_demand_avg:{:.2f} "
        "ready_lead_avg:{:.2f} late_by_avg:{:.2f}\n",
        role_name(role), value.demand_access, value.demand_miss, value.footprint_pages, value.footprint_reused, value.candidates,
        value.candidate_merged, value.filtered_resident, value.filtered_inflight, value.filtered_pending, value.filtered_capacity,
        value.issued, value.rejected, value.demanded_after_issue, value.timely, value.late, value.late_completed, value.redundant,
        value.too_early, value.never_demanded, value.unresolved_late, average(value.issue_to_demand_sum, value.demanded_after_issue),
        average(value.ready_lead_sum, value.timely), average(value.late_by_sum, value.late_completed));
  }

  if (active_instance == this) {
    gemm_runtime_loop_context::predicted_backedge_observer = nullptr;
    active_instance = nullptr;
  }
}
