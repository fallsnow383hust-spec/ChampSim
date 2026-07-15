#include "loop_boundary_tlb_realfill.h"

#include <algorithm>
#include <cstdlib>
#include <limits>

#include <fmt/core.h>

#include "cache.h"

namespace
{
template <typename AddressSlice>
uint64_t as_u64(AddressSlice addr)
{
  return addr.template to<uint64_t>();
}

template <typename Function>
void update_stats(loop_boundary_tlb_realfill& self, uint8_t phase, uint8_t role, loop_boundary_tlb_realfill::pattern_kind pattern, Function function)
{
  function(self.stats[phase][role]);
  function(self.pattern_stats[static_cast<uint8_t>(pattern)]);
}

double average(uint64_t sum, uint64_t count) { return count == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(count); }
} // namespace

uint8_t loop_boundary_tlb_realfill::role_from_ip(champsim::address ip) { return static_cast<uint8_t>(as_u64(ip) & 0x3ULL); }

uint8_t loop_boundary_tlb_realfill::phase_from_ip(champsim::address ip) { return static_cast<uint8_t>((as_u64(ip) >> 2) & 0x7ULL); }

uint64_t loop_boundary_tlb_realfill::base_key_from_ip(champsim::address ip)
{
  // Keep the operand role but remove the three trace-provided loop-context bits.
  return as_u64(ip) & ~0x1cULL;
}

uint64_t loop_boundary_tlb_realfill::vpn_from_address(champsim::address addr) { return as_u64(champsim::page_number{addr}); }

std::string_view loop_boundary_tlb_realfill::role_name(uint8_t role)
{
  constexpr std::array<std::string_view, ROLE_COUNT> names{"A", "B", "C"};
  return role < names.size() ? names[role] : "unknown";
}

std::string_view loop_boundary_tlb_realfill::phase_name(uint8_t phase)
{
  constexpr std::array<std::string_view, PHASE_COUNT> names{
      "START", "K_PROGRESS", "K_TO_IR", "IR_TO_JR", "JR_TO_IC", "IC_TO_PC", "PC_TO_JC"};
  return phase < names.size() ? names[phase] : "unknown";
}

std::string_view loop_boundary_tlb_realfill::pattern_name(pattern_kind pattern)
{
  constexpr std::array<std::string_view, PATTERN_COUNT> names{"STEADY", "BOUNDARY_POST", "BOUNDARY_RECURRENCE"};
  return names[static_cast<uint8_t>(pattern)];
}

void loop_boundary_tlb_realfill::update_stride(stride_state& state, int64_t stride)
{
  if (state.valid && state.stride == stride) {
    state.confidence = std::min<uint8_t>(7, static_cast<uint8_t>(state.confidence + 1));
  } else {
    state.stride = stride;
    state.confidence = 1;
    state.valid = true;
  }
}

bool loop_boundary_tlb_realfill::local_tlb_contains(uint64_t vpn) const
{
  return std::any_of(std::cbegin(intern_->block), std::cend(intern_->block), [vpn](const auto& block) {
    return block.valid && loop_boundary_tlb_realfill::vpn_from_address(block.address) == vpn;
  });
}

bool loop_boundary_tlb_realfill::local_tlb_inflight(uint64_t vpn) const
{
  const auto matches = [vpn](const auto& entry) { return loop_boundary_tlb_realfill::vpn_from_address(entry.address) == vpn; };
  return std::any_of(std::cbegin(intern_->MSHR), std::cend(intern_->MSHR), matches)
      || std::any_of(std::cbegin(intern_->inflight_fills), std::cend(intern_->inflight_fills), matches);
}

bool loop_boundary_tlb_realfill::issue_candidate(uint64_t current_address, int64_t byte_stride, uint8_t role, uint8_t phase,
                                                  pattern_kind pattern, uint32_t metadata_in)
{
  auto account = [&](auto function) { update_stats(*this, phase, role, pattern, function); };
  account([](auto& value) { ++value.candidates; });

  if (byte_stride == 0 || current_address > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  const auto signed_current = static_cast<int64_t>(current_address);
  if (byte_stride > 0 && signed_current > std::numeric_limits<int64_t>::max() - byte_stride)
    return false;
  const auto signed_prediction = signed_current + byte_stride;
  if (signed_prediction < 0)
    return false;
  const auto predicted_address_u64 = static_cast<uint64_t>(signed_prediction);
  const auto current_vpn = current_address >> LOG2_PAGE_SIZE;
  const auto predicted_vpn = predicted_address_u64 >> LOG2_PAGE_SIZE;
  if (predicted_vpn == current_vpn)
    return false;
  account([](auto& value) { ++value.cross_page; });

  if (local_tlb_contains(predicted_vpn)) {
    account([](auto& value) { ++value.filtered_resident; });
    return false;
  }
  if (local_tlb_inflight(predicted_vpn)) {
    account([](auto& value) { ++value.filtered_inflight; });
    return false;
  }
  if (pending.find(predicted_vpn) != pending.end()) {
    account([](auto& value) { ++value.filtered_pending; });
    return false;
  }

  // Translation is page-granular. Preserve byte-stride training precision,
  // then inject one page-aligned STLB request only when it crosses a page.
  const auto predicted_address = champsim::address{predicted_vpn << LOG2_PAGE_SIZE};
  if (!prefetch_line(predicted_address, true, metadata_in)) {
    account([](auto& value) { ++value.rejected; });
    return false;
  }

  pending.emplace(predicted_vpn,
                  pending_entry{prediction_seq++, current_address, current_cycle, demand_seq, 0, 0, role, phase, pattern, false, false, false});
  account([](auto& value) { ++value.issued; });
  return true;
}

void loop_boundary_tlb_realfill::record_demand_for_prediction(uint64_t vpn, bool cache_hit, bool useful_prefetch)
{
  const auto found = pending.find(vpn);
  if (found == pending.end() || found->second.demand_seen)
    return;

  auto& prediction = found->second;
  auto account = [&](auto function) { update_stats(*this, prediction.phase, prediction.role, prediction.pattern, function); };
  const auto issue_to_demand = current_cycle >= prediction.issue_cycle ? current_cycle - prediction.issue_cycle : 0;
  account([issue_to_demand](auto& value) {
    ++value.demanded_after_issue;
    value.issue_to_demand_sum += issue_to_demand;
    value.issue_to_demand_max = std::max(value.issue_to_demand_max, issue_to_demand);
  });

  if (useful_prefetch) {
    account([&](auto& value) {
      ++value.timely;
      if (prediction.fill_seen) {
        const auto ready_lead = current_cycle >= prediction.fill_cycle ? current_cycle - prediction.fill_cycle : 0;
        value.ready_lead_sum += ready_lead;
        value.ready_lead_max = std::max(value.ready_lead_max, ready_lead);
      }
    });
    write_event(vpn, prediction, "timely");
    pending.erase(found);
    return;
  }

  if (prediction.evicted_before_demand) {
    account([](auto& value) { ++value.too_early; });
    prediction.demand_seen = true;
    prediction.demand_cycle = current_cycle;
    write_event(vpn, prediction, "too_early");
    pending.erase(found);
    return;
  }

  if (cache_hit) {
    account([](auto& value) { ++value.nonuseful_hit; });
    prediction.demand_seen = true;
    prediction.demand_cycle = current_cycle;
    write_event(vpn, prediction, "nonuseful_hit");
    pending.erase(found);
    return;
  }

  prediction.demand_seen = true;
  prediction.demand_cycle = current_cycle;
  account([](auto& value) { ++value.late_at_demand; });
}

void loop_boundary_tlb_realfill::prefetcher_initialize()
{
  table = {};
  stats = {};
  pattern_stats = {};
  pending.clear();
  observed_requests.clear();
  current_cycle = 0;
  demand_seq = 0;
  prediction_seq = 0;
  ignored_non_pim = 0;
  duplicate_callback = 0;
  finalized = false;
  if (event_log.is_open())
    event_log.close();
  if (const auto* path = std::getenv("GEMM_TLB_EVENT_LOG"); path != nullptr && *path != '\0') {
    event_log.open(path, std::ios::out | std::ios::trunc);
    if (event_log)
      event_log << "prediction_id,role,phase,pattern,trigger_address,predicted_vpn,issue_cycle,demand_cycle,fill_cycle,outcome,"
                   "issue_to_demand,ready_lead,late_by\n";
  }
}

uint32_t loop_boundary_tlb_realfill::prefetcher_cache_operate(champsim::address addr, champsim::address full_addr, champsim::address ip,
                                                              uint8_t cache_hit, bool useful_prefetch, access_type type, uint64_t instr_id,
                                                              uint32_t metadata_in)
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
  const auto phase = phase_from_ip(ip);
  if (role >= ROLE_COUNT || phase >= PHASE_COUNT)
    return metadata_in;

  // A stalled tag lookup may call the prefetcher more than once. The extended
  // callback provides an instruction id so training and timeliness use the
  // first arrival only. The role suffix also supports a future fused trace.
  const auto request_key = (instr_id << 2) | role;
  if (!observed_requests.insert(request_key).second) {
    ++duplicate_callback;
    return metadata_in;
  }

  ++demand_seq;
  auto& stream = stats[phase][role];
  ++stream.demand_access;
  if (!cache_hit)
    ++stream.demand_miss;
  if (useful_prefetch)
    ++stream.useful_callback;

  // addr is page-aligned by the STLB module interface; full_addr retains the
  // original operand byte offset needed to learn +64B/+128B tile streams.
  const auto raw_address = as_u64(full_addr);
  const auto vpn = vpn_from_address(addr);
  record_demand_for_prediction(vpn, cache_hit != 0, useful_prefetch);

  const auto key = base_key_from_ip(ip);
  auto& entry = table[key % TRACKER_ENTRIES];
  if (!entry.valid || entry.tag != key) {
    entry = tracker_entry{};
    entry.tag = key;
    entry.valid = true;
  }

  if (!entry.have_last) {
    entry.have_last = true;
    entry.last_address = raw_address;
    entry.last_phase = phase;
    entry.last_phase_address[phase] = raw_address;
    entry.phase_seen[phase] = true;
    return metadata_in;
  }

  const auto delta = static_cast<int64_t>(raw_address) - static_cast<int64_t>(entry.last_address);
  update_stride(entry.steady, delta);
  update_stride(entry.post_boundary[entry.last_phase], delta);

  if (entry.phase_seen[phase]) {
    const auto recurrence = static_cast<int64_t>(raw_address) - static_cast<int64_t>(entry.last_phase_address[phase]);
    update_stride(entry.boundary_recurrence[phase], recurrence);
  }
  entry.phase_seen[phase] = true;
  entry.last_phase_address[phase] = raw_address;
  entry.last_address = raw_address;
  entry.last_phase = phase;

  if (phase >= 2) {
    bool has_boundary_candidate = false;
    const auto& post = entry.post_boundary[phase];
    if (post.valid && post.confidence >= BOUNDARY_CONFIDENCE) {
      issue_candidate(raw_address, post.stride, role, phase, pattern_kind::boundary_post, metadata_in);
      has_boundary_candidate = true;
    }
    const auto& recurrence = entry.boundary_recurrence[phase];
    if (recurrence.valid && recurrence.confidence >= BOUNDARY_CONFIDENCE) {
      issue_candidate(raw_address, recurrence.stride, role, phase, pattern_kind::boundary_recurrence, metadata_in);
      has_boundary_candidate = true;
    }
    if (!has_boundary_candidate && entry.steady.valid && entry.steady.confidence >= STEADY_CONFIDENCE)
      issue_candidate(raw_address, entry.steady.stride, role, phase, pattern_kind::steady, metadata_in);
  } else if (entry.steady.valid && entry.steady.confidence >= STEADY_CONFIDENCE) {
    issue_candidate(raw_address, entry.steady.stride, role, phase, pattern_kind::steady, metadata_in);
  }

  return metadata_in;
}

uint32_t loop_boundary_tlb_realfill::prefetcher_cache_fill(champsim::address addr, long, long, uint8_t, champsim::address evicted_addr,
                                                           uint32_t metadata_in)
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
  update_stats(*this, prediction.phase, prediction.role, prediction.pattern, [late_by](auto& value) {
    ++value.late_completed;
    value.late_by_sum += late_by;
    value.late_by_max = std::max(value.late_by_max, late_by);
  });
  write_event(vpn, prediction, "late");
  pending.erase(found);
  return metadata_in;
}

void loop_boundary_tlb_realfill::prefetcher_cycle_operate() { ++current_cycle; }

void loop_boundary_tlb_realfill::write_event(uint64_t vpn, const pending_entry& prediction, std::string_view outcome)
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
  event_log << prediction.prediction_id << ',' << role_name(prediction.role) << ',' << phase_name(prediction.phase) << ','
            << pattern_name(prediction.pattern) << ',' << prediction.trigger_address << ',' << vpn << ',' << prediction.issue_cycle << ','
            << prediction.demand_cycle << ',' << prediction.fill_cycle << ',' << outcome << ',' << issue_to_demand << ',' << ready_lead << ','
            << late_by << '\n';
}

void loop_boundary_tlb_realfill::finalize_unresolved()
{
  if (finalized)
    return;
  for (const auto& [vpn, prediction] : pending) {
    update_stats(*this, prediction.phase, prediction.role, prediction.pattern, [&prediction](auto& value) {
      if (prediction.demand_seen)
        ++value.unresolved_late;
      else
        ++value.never_demanded;
    });
    write_event(vpn, prediction, prediction.demand_seen ? "unresolved_late" : "never_demanded");
  }
  if (event_log)
    event_log.flush();
  finalized = true;
}

void loop_boundary_tlb_realfill::prefetcher_final_stats()
{
  finalize_unresolved();
  fmt::print("loop_boundary_tlb_realfill_v2 ignored_non_pim:{} duplicate_callback:{} outstanding:{} cycles:{}\n", ignored_non_pim,
             duplicate_callback, pending.size(), current_cycle);

  for (uint8_t index = 0; index < PATTERN_COUNT; ++index) {
    const auto pattern = static_cast<pattern_kind>(index);
    const auto& value = pattern_stats[index];
    fmt::print(
        "loop_boundary_tlb_realfill_v2 pattern {} candidate:{} cross_page:{} same_page:{} resident_filter:{} inflight_filter:{} pending_filter:{} "
        "issued:{} rejected:{} demanded:{} timely:{} late:{} late_completed:{} nonuseful_hit:{} too_early:{} never:{} unresolved_late:{} "
        "issue_to_demand_avg:{:.2f} issue_to_demand_max:{} ready_lead_avg:{:.2f} ready_lead_max:{} late_by_avg:{:.2f} late_by_max:{}\n",
        pattern_name(pattern), value.candidates, value.cross_page, value.candidates - value.cross_page, value.filtered_resident,
        value.filtered_inflight, value.filtered_pending, value.issued, value.rejected, value.demanded_after_issue, value.timely,
        value.late_at_demand, value.late_completed, value.nonuseful_hit, value.too_early, value.never_demanded, value.unresolved_late,
        average(value.issue_to_demand_sum, value.demanded_after_issue), value.issue_to_demand_max, average(value.ready_lead_sum, value.timely),
        value.ready_lead_max, average(value.late_by_sum, value.late_completed), value.late_by_max);
  }

  for (uint8_t phase = 0; phase < PHASE_COUNT; ++phase) {
    for (uint8_t role = 0; role < ROLE_COUNT; ++role) {
      const auto& value = stats[phase][role];
      if (value.demand_access == 0 && value.candidates == 0 && value.issued == 0)
        continue;
      fmt::print(
          "loop_boundary_tlb_realfill_v2 phase {} role {} access:{} miss:{} useful_callback:{} candidate:{} cross_page:{} same_page:{} "
          "resident_filter:{} inflight_filter:{} pending_filter:{} issued:{} rejected:{} demanded:{} timely:{} late:{} late_completed:{} "
          "nonuseful_hit:{} too_early:{} never:{} unresolved_late:{} issue_to_demand_avg:{:.2f} issue_to_demand_max:{} "
          "ready_lead_avg:{:.2f} ready_lead_max:{} late_by_avg:{:.2f} late_by_max:{}\n",
          phase_name(phase), role_name(role), value.demand_access, value.demand_miss, value.useful_callback, value.candidates, value.cross_page,
          value.candidates - value.cross_page, value.filtered_resident, value.filtered_inflight, value.filtered_pending, value.issued, value.rejected,
          value.demanded_after_issue, value.timely, value.late_at_demand, value.late_completed, value.nonuseful_hit, value.too_early,
          value.never_demanded, value.unresolved_late, average(value.issue_to_demand_sum, value.demanded_after_issue), value.issue_to_demand_max,
          average(value.ready_lead_sum, value.timely), value.ready_lead_max, average(value.late_by_sum, value.late_completed), value.late_by_max);
    }
  }
}
