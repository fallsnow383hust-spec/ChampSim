#include "pc_role_tlb_stride_realfill.h"

#include <algorithm>
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
} // namespace

uint8_t pc_role_tlb_stride_realfill::role_from_ip(champsim::address ip)
{
  const auto role = as_u64(ip) & 0x3ULL;
  return role < ROLE_COUNT ? static_cast<uint8_t>(role) : 0;
}

std::string_view pc_role_tlb_stride_realfill::role_name(uint8_t role)
{
  switch (role) {
  case 0:
    return "A";
  case 1:
    return "B";
  case 2:
    return "C";
  default:
    return "unknown";
  }
}

std::size_t pc_role_tlb_stride_realfill::timeliness_bucket(uint64_t distance)
{
  if (distance <= 1)
    return 0;
  if (distance <= 4)
    return 1;
  if (distance <= 16)
    return 2;
  if (distance <= 64)
    return 3;
  if (distance <= 256)
    return 4;
  return 5;
}

void pc_role_tlb_stride_realfill::record_timeliness(uint8_t role, uint64_t distance)
{
  auto& stats = per_role[role];
  stats.timeliness_sum += distance;
  stats.timeliness_max = std::max(stats.timeliness_max, distance);
  ++stats.timeliness_buckets[timeliness_bucket(distance)];
}

void pc_role_tlb_stride_realfill::record_real_timeliness(uint8_t role, uint64_t distance)
{
  auto& stats = per_role[role];
  ++stats.real_useful_with_pending;
  stats.real_timeliness_sum += distance;
  stats.real_timeliness_max = std::max(stats.real_timeliness_max, distance);
  ++stats.real_timeliness_buckets[timeliness_bucket(distance)];
}

void pc_role_tlb_stride_realfill::prefetcher_initialize()
{
  demand_seq = 0;
  per_role = {};
  trained = 0;
  predictions = 0;
  issued_shadow = 0;
  issued_real = 0;
  real_rejected = 0;
  same_page_filtered = 0;
  duplicate_callback = 0;
  shadow_useful_on_miss = 0;
  shadow_redundant_on_hit = 0;
  shadow_evictions = 0;
  champ_prefetch_useful_callback = 0;
  shadow_fifo.clear();
  shadow_buffer.clear();
  observed_requests.clear();
}

uint32_t pc_role_tlb_stride_realfill::prefetcher_cache_operate(champsim::address addr, champsim::address full_addr, champsim::address ip,
                                                               uint8_t cache_hit, bool useful_prefetch, access_type type, uint64_t instr_id,
                                                               uint32_t metadata_in)
{
  if (useful_prefetch)
    ++champ_prefetch_useful_callback;

  const bool is_demand_translation = (type == access_type::LOAD || type == access_type::WRITE || type == access_type::RFO);
  if (!is_demand_translation)
    return metadata_in;

  const auto role = role_from_ip(ip);
  auto& role_stat = per_role[role];
  const auto request_key = (instr_id << 2) | role;
  if (!observed_requests.insert(request_key).second) {
    ++duplicate_callback;
    ++role_stat.duplicate_callback;
    return metadata_in;
  }

  ++demand_seq;
  ++role_stat.demand_access;

  if (!cache_hit)
    ++role_stat.demand_miss;

  if (useful_prefetch)
    ++role_stat.champ_prefetch_useful_callback;

  const auto raw_address = as_u64(full_addr);
  const auto vpn = as_u64(champsim::page_number{addr});
  // Bits [3:2] carry dynamic fused-group START/END markers in the page-stream
  // trace. They describe the same static PIM PC and must not split training.
  const auto key = as_u64(ip) & ~0xcULL;

  const auto pending_it = shadow_buffer.find(vpn);
  if (pending_it != shadow_buffer.end()) {
    const auto issue_role = pending_it->second.role;
    const auto distance = demand_seq - pending_it->second.issue_seq;

    if (useful_prefetch)
      record_real_timeliness(issue_role, distance);

    if (cache_hit) {
      ++shadow_redundant_on_hit;
      ++per_role[issue_role].shadow_timely_on_hit;
      record_timeliness(issue_role, distance);
    } else {
      ++shadow_useful_on_miss;
      ++per_role[issue_role].shadow_useful_on_miss;
      record_timeliness(issue_role, distance);
    }

    shadow_buffer.erase(pending_it);
  }

  // Train on every unique instruction-carried base address. Whether the
  // current translation hit in STLB must not remove it from the byte stream.
  const auto idx = key % TRACKER_ENTRIES;
  auto& entry = table[idx];

  if (!entry.valid || entry.tag != key) {
    entry = tracker_entry{key, raw_address, 0, 0, true};
    ++trained;
    ++role_stat.trained;
    return metadata_in;
  }

  const auto stride = static_cast<int64_t>(raw_address) - static_cast<int64_t>(entry.last_address);
  if (stride != 0 && stride == entry.last_stride)
    ++entry.confidence;
  else {
    entry.last_stride = stride;
    entry.confidence = (stride != 0) ? 1 : 0;
  }

  entry.last_address = raw_address;
  ++trained;
  ++role_stat.trained;

  if (stride == 0 || entry.confidence < CONFIDENCE_THRESHOLD)
    return metadata_in;

  for (int degree = 1; degree <= PREFETCH_DEGREE; ++degree) {
    if (raw_address > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      continue;
    const auto signed_current = static_cast<int64_t>(raw_address);
    const auto byte_delta = stride * degree;
    if ((byte_delta > 0 && signed_current > std::numeric_limits<int64_t>::max() - byte_delta)
        || (byte_delta < 0 && signed_current < -byte_delta))
      continue;
    const auto predicted_address = signed_current + byte_delta;
    if (predicted_address < 0)
      continue;

    const auto pred_vpn_u64 = static_cast<uint64_t>(predicted_address) >> LOG2_PAGE_SIZE;
    if (pred_vpn_u64 == vpn) {
      ++same_page_filtered;
      ++role_stat.same_page_filtered;
      continue;
    }
    const auto pred_addr = champsim::address{pred_vpn_u64 << LOG2_PAGE_SIZE};
    ++predictions;
    ++role_stat.predictions;

    if (shadow_buffer.emplace(pred_vpn_u64, pending_prefetch{demand_seq, role}).second) {
      shadow_fifo.push_back(pred_vpn_u64);
      ++issued_shadow;
      ++role_stat.issued_shadow;
    }

    while (shadow_fifo.size() > SHADOW_BUFFER_ENTRIES) {
      const auto evicted = shadow_fifo.front();
      shadow_fifo.pop_front();
      const auto evicted_it = shadow_buffer.find(evicted);
      if (evicted_it != shadow_buffer.end()) {
        ++per_role[evicted_it->second.role].shadow_evictions;
        shadow_buffer.erase(evicted_it);
        ++shadow_evictions;
      }
    }

    if (prefetch_line(pred_addr, true, metadata_in)) {
      ++issued_real;
      ++role_stat.issued_real;
    } else {
      ++real_rejected;
      ++role_stat.real_rejected;
    }
  }

  return metadata_in;
}

uint32_t pc_role_tlb_stride_realfill::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch,
                                                            champsim::address evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void pc_role_tlb_stride_realfill::prefetcher_final_stats()
{
  fmt::print("pc_role_tlb_stride_realfill trained: {}\n", trained);
  fmt::print("pc_role_tlb_stride_realfill predictions: {}\n", predictions);
  fmt::print("pc_role_tlb_stride_realfill issued_shadow: {}\n", issued_shadow);
  fmt::print("pc_role_tlb_stride_realfill issued_real: {}\n", issued_real);
  fmt::print("pc_role_tlb_stride_realfill real_rejected: {}\n", real_rejected);
  fmt::print("pc_role_tlb_stride_realfill same_page_filtered: {}\n", same_page_filtered);
  fmt::print("pc_role_tlb_stride_realfill duplicate_callback: {}\n", duplicate_callback);
  fmt::print("pc_role_tlb_stride_realfill useful_shadow_on_tlb_miss: {}\n", shadow_useful_on_miss);
  fmt::print("pc_role_tlb_stride_realfill redundant_shadow_on_tlb_hit: {}\n", shadow_redundant_on_hit);
  fmt::print("pc_role_tlb_stride_realfill shadow_evictions: {}\n", shadow_evictions);
  fmt::print("pc_role_tlb_stride_realfill champ_prefetch_useful_callback: {}\n", champ_prefetch_useful_callback);

  for (uint8_t role = 0; role < ROLE_COUNT; ++role) {
    const auto& stats = per_role[role];
    const auto shadow_hits = stats.shadow_timely_on_hit + stats.shadow_useful_on_miss;
    const auto avg_distance = shadow_hits ? static_cast<double>(stats.timeliness_sum) / static_cast<double>(shadow_hits) : 0.0;

    fmt::print("pc_role_tlb_stride_realfill role {} demand_access: {}\n", role_name(role), stats.demand_access);
    fmt::print("pc_role_tlb_stride_realfill role {} demand_miss: {}\n", role_name(role), stats.demand_miss);
    fmt::print("pc_role_tlb_stride_realfill role {} trained: {}\n", role_name(role), stats.trained);
    fmt::print("pc_role_tlb_stride_realfill role {} predictions: {}\n", role_name(role), stats.predictions);
    fmt::print("pc_role_tlb_stride_realfill role {} issued_shadow: {}\n", role_name(role), stats.issued_shadow);
    fmt::print("pc_role_tlb_stride_realfill role {} issued_real: {}\n", role_name(role), stats.issued_real);
    fmt::print("pc_role_tlb_stride_realfill role {} real_rejected: {}\n", role_name(role), stats.real_rejected);
    fmt::print("pc_role_tlb_stride_realfill role {} same_page_filtered: {}\n", role_name(role), stats.same_page_filtered);
    fmt::print("pc_role_tlb_stride_realfill role {} duplicate_callback: {}\n", role_name(role), stats.duplicate_callback);
    fmt::print("pc_role_tlb_stride_realfill role {} shadow_timely_on_hit: {}\n", role_name(role), stats.shadow_timely_on_hit);
    fmt::print("pc_role_tlb_stride_realfill role {} shadow_useful_on_miss: {}\n", role_name(role), stats.shadow_useful_on_miss);
    fmt::print("pc_role_tlb_stride_realfill role {} shadow_evictions: {}\n", role_name(role), stats.shadow_evictions);
    fmt::print("pc_role_tlb_stride_realfill role {} champ_prefetch_useful_callback: {}\n", role_name(role), stats.champ_prefetch_useful_callback);
    fmt::print("pc_role_tlb_stride_realfill role {} real_useful_with_pending: {}\n", role_name(role), stats.real_useful_with_pending);
    fmt::print("pc_role_tlb_stride_realfill role {} timeliness_avg_demand_events: {:.2f}\n", role_name(role), avg_distance);
    fmt::print("pc_role_tlb_stride_realfill role {} timeliness_max_demand_events: {}\n", role_name(role), stats.timeliness_max);
    fmt::print("pc_role_tlb_stride_realfill role {} timeliness_buckets <=1:{} <=4:{} <=16:{} <=64:{} <=256:{} >256:{}\n", role_name(role),
               stats.timeliness_buckets[0], stats.timeliness_buckets[1], stats.timeliness_buckets[2], stats.timeliness_buckets[3],
               stats.timeliness_buckets[4], stats.timeliness_buckets[5]);

    const auto real_avg_distance =
        stats.real_useful_with_pending ? static_cast<double>(stats.real_timeliness_sum) / static_cast<double>(stats.real_useful_with_pending) : 0.0;
    fmt::print("pc_role_tlb_stride_realfill role {} real_timeliness_avg_demand_events: {:.2f}\n", role_name(role), real_avg_distance);
    fmt::print("pc_role_tlb_stride_realfill role {} real_timeliness_max_demand_events: {}\n", role_name(role), stats.real_timeliness_max);
    fmt::print("pc_role_tlb_stride_realfill role {} real_timeliness_buckets <=1:{} <=4:{} <=16:{} <=64:{} <=256:{} >256:{}\n", role_name(role),
               stats.real_timeliness_buckets[0], stats.real_timeliness_buckets[1], stats.real_timeliness_buckets[2],
               stats.real_timeliness_buckets[3], stats.real_timeliness_buckets[4], stats.real_timeliness_buckets[5]);
  }
}
