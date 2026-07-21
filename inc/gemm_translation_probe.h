#ifndef GEMM_TRANSLATION_PROBE_H
#define GEMM_TRANSLATION_PROBE_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fmt/core.h>
#include <limits>
#include <string_view>
#include <unordered_map>

namespace gemm_translation_probe {

constexpr uint64_t role_mask = 0x3ULL;
constexpr uint64_t group_start_mask = 1ULL << 2;
constexpr uint64_t group_end_mask = 1ULL << 3;

struct role_stats {
  uint64_t translations = 0;
  uint64_t latency_sum = 0;
  uint64_t latency_max = 0;
  std::array<uint64_t, 6> latency_buckets{};
  uint64_t dtlb_access = 0;
  uint64_t dtlb_hit = 0;
  uint64_t dtlb_miss = 0;
  uint64_t stlb_access = 0;
  uint64_t stlb_hit = 0;
  uint64_t stlb_miss = 0;
  uint64_t ptw_requests = 0;
};

struct psc_level_stats {
  uint64_t lookup = 0;
  uint64_t hit = 0;
  uint64_t selected = 0;
  uint64_t fill = 0;
};

struct fused_role_record {
  uint64_t expected = 0;
  uint64_t completed = 0;
  uint64_t first_start = std::numeric_limits<uint64_t>::max();
  uint64_t last_done = 0;
  uint64_t last_nonwalk_done = 0;
  bool last_done_has_walk = false;
};

struct fused_record {
  std::array<fused_role_record, 3> role{};
  bool sealed = false;
  uint64_t first_start = std::numeric_limits<uint64_t>::max();
  uint64_t last_done = 0;
  uint64_t last_nonwalk_done = 0;
  uint8_t critical_role_mask = 0;
  bool last_done_has_walk = false;
  uint64_t stlb_miss_pages = 0;
};

struct page_record {
  uint64_t fused_id = 0;
  uint64_t start_cycle = 0;
  uint64_t vpn = 0;
  uint8_t role = 0;
  bool started = false;
  bool dtlb_miss = false;
  bool stlb_miss_leader = false;
};

struct walk_interval {
  uint64_t start_cycle = 0;
  uint64_t done_cycle = 0;
};

struct fused_stats {
  uint64_t groups_seen = 0;
  uint64_t complete = 0;
  uint64_t with_stlb_miss = 0;
  uint64_t critical_stlb_miss = 0;
  uint64_t page_walk_exposed = 0;
  uint64_t page_walk_hidden = 0;
  uint64_t span_sum = 0;
  uint64_t span_max = 0;
  uint64_t exposed_cycles_sum = 0;
  uint64_t exposed_cycles_max = 0;
  uint64_t stlb_miss_pages = 0;
  std::array<uint64_t, 3> critical_role{};
  std::array<uint64_t, 6> span_buckets{};
  std::array<uint64_t, 6> exposed_buckets{};
};

struct probe_state {
  std::unordered_map<uint64_t, page_record> pages;
  std::unordered_map<uint64_t, fused_record> fused;
  std::unordered_map<uint64_t, walk_interval> walks;
  std::array<role_stats, 3> roles{};
  std::array<psc_level_stats, 3> psc{};
  std::array<std::array<psc_level_stats, 3>, 3> role_psc{};
  fused_stats fused_summary{};
  uint64_t ptw_requests = 0;
  uint64_t next_fused_id = 0;
  uint64_t current_fused_id = 0;
  bool have_current_fused = false;
  uint64_t malformed_groups = 0;

  ~probe_state() { print(); }

  static bool enabled()
  {
    static const bool value = [] {
      const char* env = std::getenv("GEMM_TRANSLATION_PROBE");
      return env != nullptr && std::string_view(env) != "0";
    }();
    return value;
  }

  static uint8_t role_from_ip(uint64_t ip) { return static_cast<uint8_t>(ip & role_mask); }
  static bool group_start_from_ip(uint64_t ip) { return (ip & group_start_mask) != 0; }
  static bool group_end_from_ip(uint64_t ip) { return (ip & group_end_mask) != 0; }

  static std::string_view role_name(uint8_t role)
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

  static std::size_t latency_bucket(uint64_t cycles)
  {
    if (cycles <= 1)
      return 0;
    if (cycles <= 8)
      return 1;
    if (cycles <= 32)
      return 2;
    if (cycles <= 128)
      return 3;
    if (cycles <= 512)
      return 4;
    return 5;
  }

  static bool all_complete(const fused_record& record)
  {
    return record.sealed
        && std::all_of(record.role.begin(), record.role.end(), [](const auto& role) { return role.expected == role.completed; });
  }

  void on_instruction(uint64_t instr_id, uint64_t ip)
  {
    if (!enabled())
      return;

    const auto role = role_from_ip(ip);
    if (role >= roles.size())
      return;

    if (group_start_from_ip(ip)) {
      current_fused_id = next_fused_id++;
      have_current_fused = true;
      fused.try_emplace(current_fused_id);
      ++fused_summary.groups_seen;
    }

    if (!have_current_fused) {
      ++malformed_groups;
      return;
    }

    auto& record = fused[current_fused_id];
    ++record.role[role].expected;
    pages.insert_or_assign(instr_id, page_record{current_fused_id, 0, 0, role, false, false, false});

    if (group_end_from_ip(ip)) {
      record.sealed = true;
      have_current_fused = false;
      maybe_finalize(current_fused_id);
    }
  }

  void on_l1d_translation_start(uint64_t instr_id, uint64_t vaddr, uint64_t cycle)
  {
    if (!enabled())
      return;
    const auto found = pages.find(instr_id);
    if (found == pages.end())
      return;
    found->second.started = true;
    found->second.start_cycle = cycle;
    found->second.vpn = vaddr >> 12;
    auto& record = fused[found->second.fused_id];
    record.first_start = std::min(record.first_start, cycle);
    auto& role = record.role[found->second.role];
    role.first_start = std::min(role.first_start, cycle);
  }

  void on_l1d_translation_done(uint64_t instr_id, uint64_t ip, uint64_t vaddr, uint64_t cycle)
  {
    if (!enabled())
      return;
    const auto found = pages.find(instr_id);
    if (found == pages.end() || !found->second.started)
      return;

    auto page = found->second;
    page.vpn = vaddr >> 12;
    auto walk_it = walks.find(page.vpn);
    if (page.dtlb_miss && walk_it != walks.end() && walk_it->second.done_cycle == 0)
      walk_it->second.done_cycle = cycle;
    const auto walk_dependent = page.stlb_miss_leader
        || (page.dtlb_miss && walk_it != walks.end() && walk_it->second.done_cycle == cycle);
    const auto role = page.role < roles.size() ? page.role : role_from_ip(ip);
    const auto latency = cycle >= page.start_cycle ? cycle - page.start_cycle : 0;

    auto& role_stat = roles[role];
    ++role_stat.translations;
    role_stat.latency_sum += latency;
    role_stat.latency_max = std::max(role_stat.latency_max, latency);
    ++role_stat.latency_buckets[latency_bucket(latency)];

    auto& record = fused[page.fused_id];
    auto& fused_role = record.role[role];
    ++fused_role.completed;
    if (cycle > fused_role.last_done) {
      fused_role.last_done = cycle;
      fused_role.last_done_has_walk = walk_dependent;
    } else if (cycle == fused_role.last_done && walk_dependent) {
      fused_role.last_done_has_walk = true;
    }
    if (!walk_dependent)
      fused_role.last_nonwalk_done = std::max(fused_role.last_nonwalk_done, cycle);

    const auto role_bit = static_cast<uint8_t>(1U << role);
    if (cycle > record.last_done) {
      record.last_done = cycle;
      record.critical_role_mask = role_bit;
      record.last_done_has_walk = walk_dependent;
    } else if (cycle == record.last_done) {
      record.critical_role_mask = static_cast<uint8_t>(record.critical_role_mask | role_bit);
      record.last_done_has_walk = record.last_done_has_walk || walk_dependent;
    }
    if (!walk_dependent)
      record.last_nonwalk_done = std::max(record.last_nonwalk_done, cycle);
    if (walk_dependent)
      ++record.stlb_miss_pages;

    pages.erase(found);
    maybe_finalize(page.fused_id);
  }

  void on_tlb_access(std::string_view name, uint64_t instr_id, uint64_t ip, uint64_t vaddr, uint64_t cycle, bool hit)
  {
    if (!enabled())
      return;
    const auto page_it = pages.find(instr_id);
    const auto role = page_it == pages.end() ? role_from_ip(ip) : page_it->second.role;
    if (role >= roles.size())
      return;

    auto& role_stat = roles[role];
    if (name.find("DTLB") != std::string_view::npos) {
      ++role_stat.dtlb_access;
      if (hit) {
        ++role_stat.dtlb_hit;
      } else {
        ++role_stat.dtlb_miss;
        if (page_it != pages.end()) {
          page_it->second.dtlb_miss = true;
          page_it->second.vpn = vaddr >> 12;
        }
      }
    } else if (name.find("STLB") != std::string_view::npos) {
      ++role_stat.stlb_access;
      if (hit) {
        ++role_stat.stlb_hit;
      } else {
        ++role_stat.stlb_miss;
        if (page_it != pages.end()) {
          page_it->second.stlb_miss_leader = true;
          page_it->second.vpn = vaddr >> 12;
        }
        walks.insert_or_assign(vaddr >> 12, walk_interval{cycle, 0});
      }
    }
  }

  void on_ptw_request(uint64_t instr_id, uint64_t ip)
  {
    if (!enabled())
      return;
    ++ptw_requests;
    const auto page_it = pages.find(instr_id);
    const auto role = page_it == pages.end() ? role_from_ip(ip) : page_it->second.role;
    if (role < roles.size())
      ++roles[role].ptw_requests;
  }

  static std::string_view psc_name(std::size_t slot)
  {
    switch (slot) {
    case 0:
      return "PML4";
    case 1:
      return "PDP";
    case 2:
      return "PD";
    default:
      return "unknown";
    }
  }

  static bool psc_lookup_slot(std::size_t total_psc_levels, std::size_t lookup_index, std::size_t& slot)
  {
    if (total_psc_levels == 3) {
      if (lookup_index >= 3)
        return false;
      slot = lookup_index;
      return true;
    }
    if (total_psc_levels == 4) {
      if (lookup_index == 0 || lookup_index > 3)
        return false;
      slot = lookup_index - 1;
      return true;
    }
    if (lookup_index >= 3)
      return false;
    slot = lookup_index;
    return true;
  }

  static bool psc_internal_level_slot(std::size_t total_psc_levels, std::size_t internal_level, std::size_t& slot)
  {
    if (internal_level == 0 || internal_level > total_psc_levels)
      return false;
    return psc_lookup_slot(total_psc_levels, total_psc_levels - internal_level, slot);
  }

  void on_psc_lookup(std::size_t total_psc_levels, std::size_t lookup_index, bool hit, uint64_t instr_id, uint64_t ip)
  {
    if (!enabled())
      return;
    std::size_t slot = 0;
    if (!psc_lookup_slot(total_psc_levels, lookup_index, slot))
      return;

    ++psc[slot].lookup;
    if (hit)
      ++psc[slot].hit;

    const auto page_it = pages.find(instr_id);
    const auto role = page_it == pages.end() ? role_from_ip(ip) : page_it->second.role;
    if (role < roles.size()) {
      ++role_psc[role][slot].lookup;
      if (hit)
        ++role_psc[role][slot].hit;
    }
  }

  void on_psc_selected(
      std::size_t total_psc_levels, std::size_t internal_level, bool selected_from_psc, uint64_t instr_id, uint64_t ip)
  {
    if (!enabled() || !selected_from_psc)
      return;
    std::size_t slot = 0;
    if (!psc_internal_level_slot(total_psc_levels, internal_level, slot))
      return;
    ++psc[slot].selected;
    const auto page_it = pages.find(instr_id);
    const auto role = page_it == pages.end() ? role_from_ip(ip) : page_it->second.role;
    if (role < roles.size())
      ++role_psc[role][slot].selected;
  }

  void on_psc_fill(std::size_t total_psc_levels, std::size_t internal_level)
  {
    if (!enabled())
      return;
    std::size_t slot = 0;
    if (psc_internal_level_slot(total_psc_levels, internal_level, slot))
      ++psc[slot].fill;
  }

  void maybe_finalize(uint64_t fused_id)
  {
    const auto found = fused.find(fused_id);
    if (found == fused.end() || !all_complete(found->second))
      return;

    const auto& record = found->second;
    const auto first_start = record.first_start == std::numeric_limits<uint64_t>::max() ? record.last_done : record.first_start;
    const auto span = record.last_done >= first_start ? record.last_done - first_start : 0;
    const auto has_walk = record.stlb_miss_pages != 0;
    const auto critical_walk = record.last_done_has_walk;
    const auto exposed_cycles = critical_walk && record.last_done > record.last_nonwalk_done ? record.last_done - record.last_nonwalk_done : 0;

    ++fused_summary.complete;
    fused_summary.span_sum += span;
    fused_summary.span_max = std::max(fused_summary.span_max, span);
    ++fused_summary.span_buckets[latency_bucket(span)];
    fused_summary.stlb_miss_pages += record.stlb_miss_pages;
    if (has_walk)
      ++fused_summary.with_stlb_miss;
    if (critical_walk)
      ++fused_summary.critical_stlb_miss;
    if (exposed_cycles != 0) {
      ++fused_summary.page_walk_exposed;
      fused_summary.exposed_cycles_sum += exposed_cycles;
      fused_summary.exposed_cycles_max = std::max(fused_summary.exposed_cycles_max, exposed_cycles);
      ++fused_summary.exposed_buckets[latency_bucket(exposed_cycles)];
    } else if (has_walk) {
      ++fused_summary.page_walk_hidden;
    }
    for (uint8_t role = 0; role < 3; ++role) {
      if ((record.critical_role_mask & static_cast<uint8_t>(1U << role)) != 0)
        ++fused_summary.critical_role[role];
    }

    fused.erase(found);
  }

  void print() const
  {
    if (!enabled())
      return;

    const auto complete = fused_summary.complete;
    const auto avg_span = complete ? static_cast<double>(fused_summary.span_sum) / static_cast<double>(complete) : 0.0;
    const auto avg_exposed = fused_summary.page_walk_exposed
        ? static_cast<double>(fused_summary.exposed_cycles_sum) / static_cast<double>(fused_summary.page_walk_exposed)
        : 0.0;

    fmt::print("gemm_translation_probe group_encoding: ip[1:0]=role ip[2]=START ip[3]=END\n");
    fmt::print("gemm_translation_probe fused_groups_seen: {}\n", fused_summary.groups_seen);
    fmt::print("gemm_translation_probe fused_complete: {}\n", complete);
    fmt::print("gemm_translation_probe fused_incomplete: {}\n", fused.size());
    fmt::print("gemm_translation_probe malformed_groups: {}\n", malformed_groups);
    fmt::print("gemm_translation_probe fused_with_stlb_miss: {}\n", fused_summary.with_stlb_miss);
    fmt::print("gemm_translation_probe fused_critical_stlb_miss: {}\n", fused_summary.critical_stlb_miss);
    fmt::print("gemm_translation_probe fused_page_walk_exposed: {}\n", fused_summary.page_walk_exposed);
    fmt::print("gemm_translation_probe fused_page_walk_hidden: {}\n", fused_summary.page_walk_hidden);
    fmt::print("gemm_translation_probe fused_translation_span_avg_cycles: {:.2f}\n", avg_span);
    fmt::print("gemm_translation_probe fused_translation_span_max_cycles: {}\n", fused_summary.span_max);
    fmt::print("gemm_translation_probe fused_exposed_walk_cycles_avg: {:.2f}\n", avg_exposed);
    fmt::print("gemm_translation_probe fused_exposed_walk_cycles_max: {}\n", fused_summary.exposed_cycles_max);
    fmt::print("gemm_translation_probe fused_walk_dependent_pages: {}\n", fused_summary.stlb_miss_pages);
    fmt::print("gemm_translation_probe fused_critical_role A:{} B:{} C:{}\n", fused_summary.critical_role[0], fused_summary.critical_role[1],
               fused_summary.critical_role[2]);
    fmt::print("gemm_translation_probe fused_span_buckets <=1:{} <=8:{} <=32:{} <=128:{} <=512:{} >512:{}\n",
               fused_summary.span_buckets[0], fused_summary.span_buckets[1], fused_summary.span_buckets[2], fused_summary.span_buckets[3],
               fused_summary.span_buckets[4], fused_summary.span_buckets[5]);
    fmt::print("gemm_translation_probe exposed_walk_buckets <=1:{} <=8:{} <=32:{} <=128:{} <=512:{} >512:{}\n",
               fused_summary.exposed_buckets[0], fused_summary.exposed_buckets[1], fused_summary.exposed_buckets[2],
               fused_summary.exposed_buckets[3], fused_summary.exposed_buckets[4], fused_summary.exposed_buckets[5]);

    for (uint8_t role = 0; role < 3; ++role) {
      const auto& stat = roles[role];
      const auto avg_latency = stat.translations ? static_cast<double>(stat.latency_sum) / static_cast<double>(stat.translations) : 0.0;
      fmt::print("gemm_translation_probe role {} translations: {}\n", role_name(role), stat.translations);
      fmt::print("gemm_translation_probe role {} latency_avg_cycles: {:.2f}\n", role_name(role), avg_latency);
      fmt::print("gemm_translation_probe role {} latency_max_cycles: {}\n", role_name(role), stat.latency_max);
      fmt::print("gemm_translation_probe role {} latency_buckets <=1:{} <=8:{} <=32:{} <=128:{} <=512:{} >512:{}\n", role_name(role),
                 stat.latency_buckets[0], stat.latency_buckets[1], stat.latency_buckets[2], stat.latency_buckets[3], stat.latency_buckets[4],
                 stat.latency_buckets[5]);
      fmt::print("gemm_translation_probe role {} DTLB access:{} hit:{} miss:{}\n", role_name(role), stat.dtlb_access, stat.dtlb_hit, stat.dtlb_miss);
      fmt::print("gemm_translation_probe role {} STLB access:{} hit:{} miss:{}\n", role_name(role), stat.stlb_access, stat.stlb_hit, stat.stlb_miss);
      fmt::print("gemm_translation_probe role {} PTW requests:{}\n", role_name(role), stat.ptw_requests);
    }

    fmt::print("gemm_translation_probe ptw_requests: {}\n", ptw_requests);
    for (std::size_t slot = 0; slot < psc.size(); ++slot) {
      const auto miss = psc[slot].lookup - psc[slot].hit;
      const auto hit_rate = psc[slot].lookup ? static_cast<double>(psc[slot].hit) / static_cast<double>(psc[slot].lookup) : 0.0;
      fmt::print("gemm_translation_probe psc {} lookup:{} hit:{} miss:{} hit_rate:{:.6f} selected:{} fill:{}\n", psc_name(slot), psc[slot].lookup,
                 psc[slot].hit, miss, hit_rate, psc[slot].selected, psc[slot].fill);
      for (uint8_t role = 0; role < 3; ++role) {
        const auto& stat = role_psc[role][slot];
        const auto role_miss = stat.lookup - stat.hit;
        const auto role_hit_rate = stat.lookup ? static_cast<double>(stat.hit) / static_cast<double>(stat.lookup) : 0.0;
        fmt::print("gemm_translation_probe role {} psc {} lookup:{} hit:{} miss:{} hit_rate:{:.6f} selected:{}\n", role_name(role), psc_name(slot),
                   stat.lookup, stat.hit, role_miss, role_hit_rate, stat.selected);
      }
    }
  }
};

inline probe_state state;

inline bool should_probe_cache(std::string_view name) { return probe_state::enabled() && name.find("L1D") != std::string_view::npos; }

} // namespace gemm_translation_probe

#endif
