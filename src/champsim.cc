/*
 *    Copyright 2023 The ChampSim Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "champsim.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>
#include <fmt/chrono.h>
#include <fmt/core.h>

#include "modules.h"
#include "event_listeners.h"
#include "ooo_cpu.h"
#include "operable.h"
#include "phase_info.h"
#include "tracereader.h"

constexpr int DEADLOCK_CYCLE{500};

const auto start_time = std::chrono::steady_clock::now();

std::chrono::seconds elapsed_time() { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time); }

namespace champsim
{
long do_cycle(modules::environment_module& env, std::vector<tracereader>& traces, std::vector<std::size_t> trace_index, champsim::chrono::clock& global_clock)
{
  auto operables = env.operable_view();
  std::sort(std::begin(operables), std::end(operables),
            [](const champsim::operable& lhs, const champsim::operable& rhs) { return lhs.current_time < rhs.current_time; });

  // Operate
  long progress{0};
  for (champsim::operable& op : operables) {
    progress += op.operate_on(global_clock);
  }

  // Read from trace
  for (champsim::modules::core_module& cpu : env.cpu_view()) {
    auto& trace = traces.at(trace_index.at(cpu.get_cpu_num()));
    for (auto pkt_count = cpu.instructions_requested(); !trace.eof() && pkt_count > 0; --pkt_count) {
      cpu.push_instruction(trace());
    }
  }

  return progress;
}

phase_stats do_phase(const phase_info& phase, modules::environment_module& env, std::vector<tracereader>& traces, champsim::chrono::clock& global_clock)
{
  auto operables = env.operable_view();
  auto [phase_name, is_warmup, length, trace_index, trace_names] = phase;

  // Initialize phase
  for (champsim::operable& op : operables) {
    op.warmup = is_warmup;
    op.begin_phase();
  }

  const auto time_quantum = std::accumulate(std::cbegin(operables), std::cend(operables), champsim::chrono::clock::duration::max(),
                                            [](const auto acc, const operable& y) { return std::min(acc, y.clock_period); });

  bool livelock_trigger{false};
  uint64_t livelock_period{10000000};
  uint64_t livelock_timer{0};
  //                                   die | critical | warning
  std::vector<double> livelock_threshold{0.01, 0.02, 0.05};
  std::vector<uint64_t> livelock_instr(std::size(env.cpu_view()), 0);

  // Perform phase
  int stalled_cycle{0};
  std::vector<bool> phase_complete(std::size(env.cpu_view()), false);
  while (!std::accumulate(std::begin(phase_complete), std::end(phase_complete), true, std::logical_and{})) {
    auto next_phase_complete = phase_complete;
    global_clock.tick(time_quantum);

    auto progress = do_cycle(env, traces, trace_index, global_clock);

    if (progress == 0) {
      ++stalled_cycle;
    } else {
      stalled_cycle = 0;
    }

    // Livelock detect, every livelock_period cycles, check progress and alert the user
    livelock_timer++;
    if (livelock_timer >= livelock_period) {
      // for each cpu
      for (champsim::modules::core_module& cpu : env.cpu_view()) {
        // for each threshold
        for (auto thres = std::begin(livelock_threshold); thres != std::end(livelock_threshold); thres++) {
          double livelock_ipc = std::ceil(cpu.sim_instr() - livelock_instr[cpu.get_cpu_num()]) / std::ceil(livelock_period);
          if (livelock_ipc <= *thres) {
            if (std::distance(std::begin(livelock_threshold), thres) == 0) {
              livelock_trigger = true;
              fmt::print("{} CPU {} panic: IPC {:.5g} < {:.5g}\n", phase_name, cpu.get_cpu_num(), livelock_ipc, *thres);
            } else if (std::distance(std::begin(livelock_threshold), thres) == 1)
              fmt::print("{} CPU {} critical: IPC {:.5g} < {:.5g}\n", phase_name, cpu.get_cpu_num(), livelock_ipc, *thres);
            else
              fmt::print("{} CPU {} warning: IPC {:.5g} < {:.5g}\n", phase_name, cpu.get_cpu_num(), livelock_ipc, *thres);

            break;
          }
        }
        livelock_instr[cpu.get_cpu_num()] = cpu.sim_instr();
      }
      livelock_timer = 0;
    }

    if (stalled_cycle >= DEADLOCK_CYCLE || livelock_trigger) {
      std::for_each(std::begin(operables), std::end(operables), [](champsim::operable& c) { c.print_deadlock(); });
      abort();
    }

    // If any trace reaches EOF, terminate all phases
    if (std::any_of(std::begin(traces), std::end(traces), [](const auto& tr) { return tr.eof(); })) {
      std::fill(std::begin(next_phase_complete), std::end(next_phase_complete), true);
    }

    // Check for phase finish
    for (champsim::modules::core_module& cpu : env.cpu_view()) {
      // Phase complete
      next_phase_complete[cpu.get_cpu_num()] = next_phase_complete[cpu.get_cpu_num()] || (cpu.sim_instr() >= length);
    }

    for (champsim::modules::core_module& cpu : env.cpu_view()) {
      if (next_phase_complete[cpu.get_cpu_num()] != phase_complete[cpu.get_cpu_num()]) {
        for (champsim::operable& op : operables) {
          op.end_phase(cpu.get_cpu_num());
        }

        fmt::print("{} finished CPU {} instructions: {} cycles: {} cumulative IPC: {:.4g} (Simulation time: {:%H hr %M min %S sec})\n", phase_name, cpu.get_cpu_num(),
                   cpu.sim_instr(), cpu.sim_cycle(), std::ceil(cpu.sim_instr()) / std::ceil(cpu.sim_cycle()), elapsed_time());
      }
    }

    phase_complete = next_phase_complete;
  }

  for (champsim::modules::core_module& cpu : env.cpu_view()) {
    fmt::print("{} complete CPU {} instructions: {} cycles: {} cumulative IPC: {:.4g} (Simulation time: {:%H hr %M min %S sec})\n", phase_name, cpu.get_cpu_num(),
               cpu.sim_instr(), cpu.sim_cycle(), std::ceil(cpu.sim_instr()) / std::ceil(cpu.sim_cycle()), elapsed_time());
  }

  phase_stats stats;
  stats.name = phase.name;

  for (std::size_t i = 0; i < std::size(trace_index); ++i) {
    stats.trace_names.push_back(trace_names.at(trace_index.at(i)));
  }

  auto cpus = env.cpu_view();
  std::transform(std::begin(cpus), std::end(cpus), std::back_inserter(stats.sim_cpu_stats), [](const champsim::modules::core_module& cpu) { return cpu.get_sim_stats(); });
  std::transform(std::begin(cpus), std::end(cpus), std::back_inserter(stats.roi_cpu_stats), [](const champsim::modules::core_module& cpu) { return cpu.get_roi_stats(); });

  auto caches = env.cache_view();
  std::transform(std::begin(caches), std::end(caches), std::back_inserter(stats.sim_cache_stats), [](const champsim::modules::cache_module& cache) { return cache.get_sim_stats(); });
  std::transform(std::begin(caches), std::end(caches), std::back_inserter(stats.roi_cache_stats), [](const champsim::modules::cache_module& cache) { return cache.get_roi_stats(); });

  champsim::modules::memory_controller_module& dram = env.dram_view();
  for(std::size_t chan_no = 0; chan_no < dram.get_num_channels(); ++chan_no) {
    stats.sim_dram_stats.push_back(dram.get_sim_stats(chan_no));
    stats.roi_dram_stats.push_back(dram.get_roi_stats(chan_no));
  }

  return stats;
}

// simulation entry point
std::vector<phase_stats> main(modules::environment_module& env, std::vector<phase_info>& phases, std::vector<tracereader>& traces)
{
  for (champsim::operable& op : env.operable_view()) {
    op.initialize();
  }

  champsim::chrono::clock global_clock;
  std::vector<phase_stats> results;
  for (auto phase : phases) {
    // call event listeners
    handle_event<Event::BEGIN_PHASE>(phase.is_warmup);
    // handle_begin_phase(0, phase.is_warmup);

    auto stats = do_phase(phase, env, traces, global_clock);
    if (!phase.is_warmup) {
      results.push_back(stats);
    }
  }

  return results;
}
} // namespace champsim
