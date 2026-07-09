#include "pc_role_tlb_stride.h"

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

void pc_role_tlb_stride::prefetcher_initialize()
{
  issued = 0;
  useful = 0;
  trained = 0;
  predictions = 0;
  shadow_useful_on_miss = 0;
  shadow_redundant_on_hit = 0;
  shadow_evictions = 0;
  shadow_fifo.clear();
  shadow_buffer.clear();
}

uint32_t pc_role_tlb_stride::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch,
                                                      access_type type, uint32_t metadata_in)
{
  if (useful_prefetch)
    ++useful;

  /*
   * Intended use: attach this module to a TLB level, either DTLB or STLB.
   *
   * The module is trained only on misses in the TLB level where it is attached.
   * If attached to DTLB, this means DTLB misses. If attached to STLB, this means
   * STLB misses. The index is the dynamic instruction PC. For PIM-GEMM
   * experiments, encode the operand role into the trace IP, e.g.
   * ip = base_pc + role_id, so the hardware model becomes PC+role without
   * changing ChampSim's trace format.
   */
  const bool is_demand_translation = (type == access_type::LOAD || type == access_type::WRITE || type == access_type::RFO);
  if (!is_demand_translation)
    return metadata_in;

  // ChampSim passes a page-aligned address to TLB prefetchers. Train in VPN
  // units, then convert the predicted VPN back to a page-base address before
  // calling prefetch_line().
  const auto vpn = as_u64(champsim::page_number{addr});
  const auto key = as_u64(ip);

  /*
   * Shadow/evaluation mode:
   *
   * ChampSim's current TLB prefetch path can deadlock for this synthetic trace
   * when a module injects real TLB PQ requests with prefetch_line(). To keep
   * the experiment stable, the prefetcher models a small TLB-prefetch buffer
   * internally and reports how many demand TLB misses would have been covered.
   * This does not change ChampSim's architectural DTLB/STLB/PTW stats.
   */
  if (shadow_buffer.erase(vpn) != 0) {
    if (cache_hit)
      ++shadow_redundant_on_hit;
    else
      ++shadow_useful_on_miss;
  }

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
    ++predictions;
    if (shadow_buffer.insert(pred_vpn_u64).second) {
      shadow_fifo.push_back(pred_vpn_u64);
      ++issued;
    }

    while (shadow_fifo.size() > SHADOW_BUFFER_ENTRIES) {
      const auto evicted = shadow_fifo.front();
      shadow_fifo.pop_front();
      if (shadow_buffer.erase(evicted) != 0)
        ++shadow_evictions;
    }
  }

  return metadata_in;
}

uint32_t pc_role_tlb_stride::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr,
                                                   uint32_t metadata_in)
{
  return metadata_in;
}

void pc_role_tlb_stride::prefetcher_final_stats()
{
  fmt::print("pc_role_tlb_stride trained: {}\n", trained);
  fmt::print("pc_role_tlb_stride predictions: {}\n", predictions);
  fmt::print("pc_role_tlb_stride issued_shadow: {}\n", issued);
  fmt::print("pc_role_tlb_stride useful_shadow_on_tlb_miss: {}\n", shadow_useful_on_miss);
  fmt::print("pc_role_tlb_stride redundant_shadow_on_tlb_hit: {}\n", shadow_redundant_on_hit);
  fmt::print("pc_role_tlb_stride shadow_evictions: {}\n", shadow_evictions);
  fmt::print("pc_role_tlb_stride champ_prefetch_useful_callback: {}\n", useful);
}
