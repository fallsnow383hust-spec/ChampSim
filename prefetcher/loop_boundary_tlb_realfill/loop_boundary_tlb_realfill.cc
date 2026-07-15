#include "loop_boundary_tlb_realfill.h"

#include <algorithm>

#include <fmt/core.h>

#include "cache.h"

namespace
{
template <typename AddressSlice>
uint64_t as_u64(AddressSlice addr)
{
  return addr.template to<uint64_t>();
}
} // namespace

uint8_t loop_boundary_tlb_realfill::role_from_ip(champsim::address ip) { return static_cast<uint8_t>(as_u64(ip) & 0x3ULL); }

uint8_t loop_boundary_tlb_realfill::phase_from_ip(champsim::address ip) { return static_cast<uint8_t>((as_u64(ip) >> 2) & 0x7ULL); }

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

void loop_boundary_tlb_realfill::prefetcher_initialize()
{
  table = {};
  stats = {};
  pending.clear();
  pending_fifo.clear();
  demand_seq = 0;
  duplicate_vpn = 0;
  duplicate_prediction = 0;
  pending_evictions = 0;
  champ_useful = 0;
}

uint32_t loop_boundary_tlb_realfill::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit,
                                                              bool useful_prefetch, access_type type, uint32_t metadata_in)
{
  const bool demand = type == access_type::LOAD || type == access_type::WRITE || type == access_type::RFO;
  if (!demand)
    return metadata_in;

  const auto role = role_from_ip(ip);
  const auto phase = phase_from_ip(ip);
  if (role >= ROLE_COUNT || phase >= PHASE_COUNT)
    return metadata_in;

  ++demand_seq;
  auto& stream = stats[phase][role];
  ++stream.accesses;
  if (!cache_hit)
    ++stream.misses;
  if (useful_prefetch)
    ++champ_useful;

  const auto vpn = as_u64(champsim::page_number{addr});
  if (const auto pending_it = pending.find(vpn); pending_it != pending.end()) {
    const auto information = pending_it->second;
    auto& issuing_stream = stats[information.phase][information.role];
    const auto distance = demand_seq - information.issue_seq;
    ++issuing_stream.pending_hits;
    if (useful_prefetch)
      ++issuing_stream.real_useful;
    issuing_stream.lead_sum += distance;
    issuing_stream.lead_max = std::max(issuing_stream.lead_max, distance);
    pending.erase(pending_it);
  }

  // The complete synthetic IP is the hardware key: PIM site + loop phase +
  // operand role. The converter reserves five low bits for phase and role.
  const auto key = as_u64(ip);
  auto& entry = table[key % TRACKER_ENTRIES];
  if (!entry.valid || entry.tag != key) {
    entry = tracker_entry{key, vpn, 0, 0, true};
    ++stream.trained;
    return metadata_in;
  }

  if (vpn == entry.last_vpn) {
    ++duplicate_vpn;
    return metadata_in;
  }

  const auto stride = static_cast<int64_t>(vpn) - static_cast<int64_t>(entry.last_vpn);
  if (stride == entry.stride)
    entry.confidence = std::min<uint8_t>(7, static_cast<uint8_t>(entry.confidence + 1));
  else {
    entry.stride = stride;
    entry.confidence = 1;
  }
  entry.last_vpn = vpn;
  ++stream.trained;

  if (stride == 0 || entry.confidence < CONFIDENCE_THRESHOLD)
    return metadata_in;

  const auto prediction = static_cast<int64_t>(vpn) + stride;
  if (prediction < 0)
    return metadata_in;

  ++stream.predictions;
  const auto predicted_vpn = static_cast<uint64_t>(prediction);
  if (pending.find(predicted_vpn) != pending.end()) {
    ++duplicate_prediction;
    return metadata_in;
  }

  const auto predicted_address = champsim::address{predicted_vpn << LOG2_PAGE_SIZE};
  if (prefetch_line(predicted_address, true, metadata_in)) {
    ++stream.issued;
    pending.emplace(predicted_vpn, pending_entry{demand_seq, role, phase});
    pending_fifo.push_back(predicted_vpn);
    while (pending_fifo.size() > PENDING_ENTRIES) {
      const auto victim = pending_fifo.front();
      pending_fifo.pop_front();
      if (pending.erase(victim) != 0)
        ++pending_evictions;
    }
  } else {
    ++stream.rejected;
  }
  return metadata_in;
}

uint32_t loop_boundary_tlb_realfill::prefetcher_cache_fill(champsim::address, long, long, uint8_t, champsim::address,
                                                           uint32_t metadata_in)
{
  return metadata_in;
}

void loop_boundary_tlb_realfill::prefetcher_final_stats()
{
  fmt::print("loop_boundary_tlb_realfill duplicate_vpn: {}\n", duplicate_vpn);
  fmt::print("loop_boundary_tlb_realfill duplicate_prediction: {}\n", duplicate_prediction);
  fmt::print("loop_boundary_tlb_realfill pending_evictions: {}\n", pending_evictions);
  fmt::print("loop_boundary_tlb_realfill champ_prefetch_useful_callback: {}\n", champ_useful);
  for (uint8_t phase = 0; phase < PHASE_COUNT; ++phase) {
    for (uint8_t role = 0; role < ROLE_COUNT; ++role) {
      const auto& value = stats[phase][role];
      if (value.accesses == 0)
        continue;
      const auto average_lead = value.pending_hits ? static_cast<double>(value.lead_sum) / static_cast<double>(value.pending_hits) : 0.0;
      fmt::print(
          "loop_boundary_tlb_realfill phase {} role {} access:{} miss:{} trained:{} prediction:{} issued:{} rejected:{} pending_hit:{} "
          "real_useful:{} lead_avg:{:.2f} lead_max:{}\n",
          phase_name(phase), role_name(role), value.accesses, value.misses, value.trained, value.predictions, value.issued, value.rejected,
          value.pending_hits, value.real_useful, average_lead, value.lead_max);
    }
  }
}
