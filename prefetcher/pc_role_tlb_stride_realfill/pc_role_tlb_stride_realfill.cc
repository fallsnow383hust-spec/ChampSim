#include "pc_role_tlb_stride_realfill.h"

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

void pc_role_tlb_stride_realfill::prefetcher_initialize()
{
  trained = 0;
  predictions = 0;
  issued_shadow = 0;
  issued_real = 0;
  real_rejected = 0;
  shadow_useful_on_miss = 0;
  shadow_redundant_on_hit = 0;
  shadow_evictions = 0;
  champ_prefetch_useful_callback = 0;
  shadow_fifo.clear();
  shadow_buffer.clear();
}

uint32_t pc_role_tlb_stride_realfill::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch,
                                                               access_type type, uint32_t metadata_in)
{
  if (useful_prefetch)
    ++champ_prefetch_useful_callback;

  const bool is_demand_translation = (type == access_type::LOAD || type == access_type::WRITE || type == access_type::RFO);
  if (!is_demand_translation)
    return metadata_in;

  const auto vpn = as_u64(champsim::page_number{addr});
  const auto key = as_u64(ip);

  if (shadow_buffer.erase(vpn) != 0) {
    if (cache_hit)
      ++shadow_redundant_on_hit;
    else
      ++shadow_useful_on_miss;
  }

  // Train only on misses in the attached TLB level. With the real-fill config,
  // this module is attached to STLB, so this means STLB misses.
  if (cache_hit)
    return metadata_in;

  const auto idx = key % TRACKER_ENTRIES;
  auto& entry = table[idx];

  if (!entry.valid || entry.tag != key) {
    entry = tracker_entry{key, vpn, 0, 0, true};
    ++trained;
    return metadata_in;
  }

  const auto stride = static_cast<int64_t>(vpn) - static_cast<int64_t>(entry.last_vpn);
  if (stride != 0 && stride == entry.last_stride)
    ++entry.confidence;
  else {
    entry.last_stride = stride;
    entry.confidence = (stride != 0) ? 1 : 0;
  }

  entry.last_vpn = vpn;
  ++trained;

  if (stride == 0 || entry.confidence < CONFIDENCE_THRESHOLD)
    return metadata_in;

  for (int degree = 1; degree <= PREFETCH_DEGREE; ++degree) {
    const auto pred_vpn = static_cast<int64_t>(vpn) + stride * degree;
    if (pred_vpn < 0)
      continue;

    const auto pred_vpn_u64 = static_cast<uint64_t>(pred_vpn);
    const auto pred_addr = champsim::address{pred_vpn_u64 << LOG2_PAGE_SIZE};
    ++predictions;

    if (shadow_buffer.insert(pred_vpn_u64).second) {
      shadow_fifo.push_back(pred_vpn_u64);
      ++issued_shadow;
    }

    while (shadow_fifo.size() > SHADOW_BUFFER_ENTRIES) {
      const auto evicted = shadow_fifo.front();
      shadow_fifo.pop_front();
      if (shadow_buffer.erase(evicted) != 0)
        ++shadow_evictions;
    }

    if (prefetch_line(pred_addr, true, metadata_in))
      ++issued_real;
    else
      ++real_rejected;
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
  fmt::print("pc_role_tlb_stride_realfill useful_shadow_on_tlb_miss: {}\n", shadow_useful_on_miss);
  fmt::print("pc_role_tlb_stride_realfill redundant_shadow_on_tlb_hit: {}\n", shadow_redundant_on_hit);
  fmt::print("pc_role_tlb_stride_realfill shadow_evictions: {}\n", shadow_evictions);
  fmt::print("pc_role_tlb_stride_realfill champ_prefetch_useful_callback: {}\n", champ_prefetch_useful_callback);
}
