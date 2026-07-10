#ifndef GEMM_TRANSLATION_PROBE_H
#define GEMM_TRANSLATION_PROBE_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fmt/core.h>
#include <string>
#include <string_view>
#include <unordered_map>

namespace gemm_translation_probe {

struct role_latency {
  bool valid = false;
  uint64_t latency = 0;
};

struct fused_record {
  std::array<role_latency, 3> role{};
};

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
};

struct fused_stats {
  uint64_t complete = 0;
  uint64_t max_latency_sum = 0;
  uint64_t max_latency_max = 0;
  std::array<uint64_t, 3> critical_role{};
  std::array<uint64_t, 6> max_latency_buckets{};
};

struct psc_level_stats {
  uint64_t lookup = 0;
  uint64_t hit = 0;
  uint64_t selected = 0;
  uint64_t fill = 0;
};

struct probe_state {
  std::unordered_map<uint64_t, uint64_t> starts;
  std::unordered_map<uint64_t, fused_record> fused;
  std::array<role_stats, 3> roles{};
  fused_stats fused_summary{};
  std::array<psc_level_stats, 6> psc{};
  uint64_t ptw_requests = 0;

  ~probe_state() { print(); }

  static uint64_t cycle_from_time(uint64_t time_since_epoch, uint64_t period)
  {
    return period == 0 ? time_since_epoch : time_since_epoch / period;
  }

  static uint8_t role_from_ip(uint64_t ip) { return static_cast<uint8_t>(ip & 0x3ULL); }
  static uint64_t fused_id_from_instr(uint64_t instr_id) { return instr_id / 3; }

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

  void on_l1d_translation_start(uint64_t instr_id, uint64_t cycle)
  {
    starts.emplace(instr_id, cycle);
  }

  void on_l1d_translation_done(uint64_t instr_id, uint64_t ip, uint64_t cycle)
  {
    const auto start_it = starts.find(instr_id);
    if (start_it == starts.end())
      return;

    const auto role = role_from_ip(ip);
    if (role >= roles.size()) {
      starts.erase(start_it);
      return;
    }

    const auto latency = cycle >= start_it->second ? cycle - start_it->second : 0;
    starts.erase(start_it);

    auto& role_stat = roles[role];
    ++role_stat.translations;
    role_stat.latency_sum += latency;
    role_stat.latency_max = std::max(role_stat.latency_max, latency);
    ++role_stat.latency_buckets[latency_bucket(latency)];

    const auto fused_id = fused_id_from_instr(instr_id);
    auto& record = fused[fused_id];
    record.role[role] = role_latency{true, latency};

    if (record.role[0].valid && record.role[1].valid && record.role[2].valid) {
      uint8_t critical = 0;
      uint64_t max_latency = record.role[0].latency;
      for (uint8_t r = 1; r < 3; ++r) {
        if (record.role[r].latency > max_latency) {
          critical = r;
          max_latency = record.role[r].latency;
        }
      }

      ++fused_summary.complete;
      fused_summary.max_latency_sum += max_latency;
      fused_summary.max_latency_max = std::max(fused_summary.max_latency_max, max_latency);
      ++fused_summary.critical_role[critical];
      ++fused_summary.max_latency_buckets[latency_bucket(max_latency)];
      fused.erase(fused_id);
    }
  }

  void on_tlb_access(std::string_view name, uint64_t ip, bool hit)
  {
    const auto role = role_from_ip(ip);
    if (role >= roles.size())
      return;

    auto& role_stat = roles[role];
    if (name.find("DTLB") != std::string_view::npos) {
      ++role_stat.dtlb_access;
      if (hit)
        ++role_stat.dtlb_hit;
      else
        ++role_stat.dtlb_miss;
    } else if (name.find("STLB") != std::string_view::npos) {
      ++role_stat.stlb_access;
      if (hit)
        ++role_stat.stlb_hit;
      else
        ++role_stat.stlb_miss;
    }
  }

  void on_ptw_request() { ++ptw_requests; }

  void on_psc_lookup(std::size_t level, bool hit)
  {
    if (level >= psc.size())
      return;
    ++psc[level].lookup;
    if (hit)
      ++psc[level].hit;
  }

  void on_psc_selected(std::size_t level)
  {
    if (level < psc.size())
      ++psc[level].selected;
  }

  void on_psc_fill(std::size_t level)
  {
    if (level < psc.size())
      ++psc[level].fill;
  }

  void print() const
  {
    const bool enabled = [] {
      const char* env = std::getenv("GEMM_TRANSLATION_PROBE");
      return env != nullptr && std::string_view(env) != "0";
    }();
    if (!enabled)
      return;

    fmt::print("gemm_translation_probe fused_complete: {}\n", fused_summary.complete);
    const auto avg_max = fused_summary.complete ? static_cast<double>(fused_summary.max_latency_sum) / static_cast<double>(fused_summary.complete) : 0.0;
    fmt::print("gemm_translation_probe fused_max_latency_avg_cycles: {:.2f}\n", avg_max);
    fmt::print("gemm_translation_probe fused_max_latency_max_cycles: {}\n", fused_summary.max_latency_max);
    fmt::print("gemm_translation_probe fused_critical_role A:{} B:{} C:{}\n", fused_summary.critical_role[0], fused_summary.critical_role[1],
               fused_summary.critical_role[2]);
    fmt::print("gemm_translation_probe fused_max_latency_buckets <=1:{} <=8:{} <=32:{} <=128:{} <=512:{} >512:{}\n",
               fused_summary.max_latency_buckets[0], fused_summary.max_latency_buckets[1], fused_summary.max_latency_buckets[2],
               fused_summary.max_latency_buckets[3], fused_summary.max_latency_buckets[4], fused_summary.max_latency_buckets[5]);

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
    }

    fmt::print("gemm_translation_probe ptw_requests: {}\n", ptw_requests);
    for (std::size_t level = 0; level < psc.size(); ++level) {
      if (psc[level].lookup == 0 && psc[level].fill == 0 && psc[level].selected == 0)
        continue;
      const auto hit_rate = psc[level].lookup ? static_cast<double>(psc[level].hit) / static_cast<double>(psc[level].lookup) : 0.0;
      fmt::print("gemm_translation_probe psc level {} lookup:{} hit:{} hit_rate:{:.4f} selected:{} fill:{}\n", level, psc[level].lookup, psc[level].hit,
                 hit_rate, psc[level].selected, psc[level].fill);
    }
  }
};

inline probe_state state;

inline bool should_probe_cache(std::string_view name) { return name.find("L1D") != std::string_view::npos; }

} // namespace gemm_translation_probe

#endif
