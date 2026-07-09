#include "b_next_vpn_realfill.h"

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

uint8_t b_next_vpn_realfill::role_from_ip(champsim::address ip)
{
  return static_cast<uint8_t>(as_u64(ip) & 0x3ULL);
}

std::size_t b_next_vpn_realfill::timeliness_bucket(uint64_t distance)
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

void b_next_vpn_realfill::record_timeliness(uint64_t distance)
{
  ++b_real_useful_with_pending;
  b_timeliness_sum += distance;
  b_timeliness_max = std::max(b_timeliness_max, distance);
  ++b_timeliness_buckets[timeliness_bucket(distance)];
}

void b_next_vpn_realfill::prefetcher_initialize()
{
  pending_fifo.clear();
  pending.clear();
  demand_seq = 0;
  demand_access = 0;
  b_demand_access = 0;
  b_demand_miss = 0;
  b_candidates = 0;
  b_pending_inserted = 0;
  b_pending_duplicate = 0;
  b_issued_real = 0;
  b_real_rejected = 0;
  b_prefetch_useful_callback = 0;
  b_real_useful_with_pending = 0;
  b_pending_evictions = 0;
  b_timeliness_sum = 0;
  b_timeliness_max = 0;
  b_timeliness_buckets = {};
}

uint32_t b_next_vpn_realfill::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch,
                                                       access_type type, uint32_t metadata_in)
{
  const bool is_demand_translation = (type == access_type::LOAD || type == access_type::WRITE || type == access_type::RFO);
  if (!is_demand_translation)
    return metadata_in;

  ++demand_seq;
  ++demand_access;

  const auto role = role_from_ip(ip);
  const auto vpn = as_u64(champsim::page_number{addr});

  const auto pending_it = pending.find(vpn);
  if (pending_it != pending.end()) {
    if (useful_prefetch)
      record_timeliness(demand_seq - pending_it->second.issue_seq);
    pending.erase(pending_it);
  }

  if (role != B_ROLE)
    return metadata_in;

  ++b_demand_access;
  if (!cache_hit)
    ++b_demand_miss;
  if (useful_prefetch)
    ++b_prefetch_useful_callback;

  const auto pred_vpn = vpn + 1;
  const auto pred_addr = champsim::address{pred_vpn << LOG2_PAGE_SIZE};
  ++b_candidates;

  if (pending.emplace(pred_vpn, pending_prefetch{demand_seq}).second) {
    pending_fifo.push_back(pred_vpn);
    ++b_pending_inserted;
  } else {
    ++b_pending_duplicate;
  }

  while (pending_fifo.size() > PENDING_ENTRIES) {
    const auto evicted = pending_fifo.front();
    pending_fifo.pop_front();
    if (pending.erase(evicted) != 0)
      ++b_pending_evictions;
  }

  if (prefetch_line(pred_addr, true, metadata_in))
    ++b_issued_real;
  else
    ++b_real_rejected;

  return metadata_in;
}

uint32_t b_next_vpn_realfill::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr,
                                                    uint32_t metadata_in)
{
  return metadata_in;
}

void b_next_vpn_realfill::prefetcher_final_stats()
{
  const auto avg_distance =
      b_real_useful_with_pending ? static_cast<double>(b_timeliness_sum) / static_cast<double>(b_real_useful_with_pending) : 0.0;

  fmt::print("b_next_vpn_realfill demand_access: {}\n", demand_access);
  fmt::print("b_next_vpn_realfill role B demand_access: {}\n", b_demand_access);
  fmt::print("b_next_vpn_realfill role B demand_miss: {}\n", b_demand_miss);
  fmt::print("b_next_vpn_realfill role B candidates_vpn_plus_1: {}\n", b_candidates);
  fmt::print("b_next_vpn_realfill role B pending_inserted: {}\n", b_pending_inserted);
  fmt::print("b_next_vpn_realfill role B pending_duplicate: {}\n", b_pending_duplicate);
  fmt::print("b_next_vpn_realfill role B issued_real: {}\n", b_issued_real);
  fmt::print("b_next_vpn_realfill role B real_rejected: {}\n", b_real_rejected);
  fmt::print("b_next_vpn_realfill role B champ_prefetch_useful_callback: {}\n", b_prefetch_useful_callback);
  fmt::print("b_next_vpn_realfill role B real_useful_with_pending: {}\n", b_real_useful_with_pending);
  fmt::print("b_next_vpn_realfill role B pending_evictions: {}\n", b_pending_evictions);
  fmt::print("b_next_vpn_realfill role B real_timeliness_avg_demand_events: {:.2f}\n", avg_distance);
  fmt::print("b_next_vpn_realfill role B real_timeliness_max_demand_events: {}\n", b_timeliness_max);
  fmt::print("b_next_vpn_realfill role B real_timeliness_buckets <=1:{} <=4:{} <=16:{} <=64:{} <=256:{} >256:{}\n",
             b_timeliness_buckets[0], b_timeliness_buckets[1], b_timeliness_buckets[2], b_timeliness_buckets[3], b_timeliness_buckets[4],
             b_timeliness_buckets[5]);
}
