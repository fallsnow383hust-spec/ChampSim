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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <CLI/CLI.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "cache.h" // for CACHE
#include "champsim.h"
#include "defaults.hpp"
#include "environment.h"
#include "legacy_environment.h"
#include "event_listeners.h"
#include "modules.h"
#include "ooo_cpu.h" // for O3_CPU
#include "phase_info.h"
#include "stats_printer.h"
#include "tracereader.h"
#include "vmem.h"

namespace champsim
{
std::vector<phase_stats> main(modules::environment_module& env, std::vector<phase_info>& phases, std::vector<tracereader>& traces);
}

std::size_t NUM_CPUS = 1;
unsigned BLOCK_SIZE = 64;
unsigned PAGE_SIZE = 4096;
unsigned LOG2_BLOCK_SIZE = 6;
unsigned LOG2_PAGE_SIZE = 12;

int main(int argc, char** argv) // NOLINT(bugprone-exception-escape)
{
  CLI::App app{"A microarchitecture simulator for research and education"};

  std::string config_file_path;
  bool knob_cloudsuite{false};
  bool knob_dump{false};
  long long warmup_instructions = 0;
  long long simulation_instructions = std::numeric_limits<long long>::max();
  std::string json_file_name;
  std::vector<std::string> requested_listeners;
  std::vector<std::string> trace_names;

  app.add_option("--config", config_file_path, "Path to the JSON configuration file (use \"-\" for stdin)");
  app.add_flag("-c,--cloudsuite", knob_cloudsuite, "Read all traces using the cloudsuite format");
  app.add_flag("--dump", knob_dump, "Print each module builder's parameters as modules are constructed");
  auto* warmup_instr_option = app.add_option("-w,--warmup-instructions", warmup_instructions, "The number of instructions in the warmup phase");
  auto* deprec_warmup_instr_option =
      app.add_option("--warmup_instructions", warmup_instructions, "[deprecated] use --warmup-instructions instead")->excludes(warmup_instr_option);
  auto* sim_instr_option = app.add_option("-i,--simulation-instructions", simulation_instructions,
                                          "The number of instructions in the detailed phase. If not specified, run to the end of the trace.");
  auto* deprec_sim_instr_option =
      app.add_option("--simulation_instructions", simulation_instructions, "[deprecated] use --simulation-instructions instead")->excludes(sim_instr_option);

  auto* json_option =
      app.add_option("--json", json_file_name, "The name of the file to receive JSON output. If no name is specified, stdout will be used")->expected(0, 1);

  app.add_option("--listeners", requested_listeners, "A list of the listeners to be attached to the run");

  // Parse CLI first pass to get config file, then we'll know NUM_CPUS for trace validation
  app.allow_extras(true);
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  // Enable dump mode if requested
  if (knob_dump) fmt::print("=== Module Builder Dump ===\n");

  // Read JSON config from file or stdin
  nlohmann::json config_json;
  if (config_file_path == "-") {
    // Explicit stdin request
    try {
      config_json = nlohmann::json::parse(std::cin);
    } catch (const nlohmann::json::parse_error& e) {
      fmt::print("ERROR: Failed to parse JSON from stdin: {}\n", e.what());
      return 1;
    }
  } else {
    if (config_file_path.empty()) config_file_path = "champsim_config.json";
    std::ifstream config_stream(config_file_path);
    if (config_stream.is_open()) {
      try {
        config_json = nlohmann::json::parse(config_stream);
      } catch (const nlohmann::json::parse_error& e) {
        fmt::print("ERROR: Failed to parse JSON config file {}: {}\n", config_file_path, e.what());
        return 1;
      }
    }
  }

  // Print config description if present
  if (config_json.contains("_description") && config_json["_description"].is_string()) {
    fmt::print(stderr, "\nConfig: {}\n\n", config_json["_description"].get<std::string>());
  }

  // Construct the environment via the module system
  std::string env_model = config_json.value("environment", std::string("LEGACY_ENVIRONMENT"));
  // Set globals from the environment
  NUM_CPUS = config_json.value("num_cores", 1u); // default to 1 CPU if not specified, needed for trace validation
  BLOCK_SIZE = config_json.value("block_size", 64u);
  PAGE_SIZE = config_json.value("page_size", 4096u);
  LOG2_BLOCK_SIZE = champsim::lg2(BLOCK_SIZE);
  LOG2_PAGE_SIZE = champsim::lg2(PAGE_SIZE);

  auto env_builder = champsim::modules::ModuleBuilder("environment", env_model)
    .add_parameter("config_json", config_json);
  champsim::modules::ModuleBuilder::set_dump_enabled(knob_dump);
  auto* gen_environment = champsim::modules::environment_module::create_instance(env_builder, static_cast<champsim::modules::environment_module*>(nullptr));

  if (knob_dump) fmt::print("=== End Module Builder Dump ===\n");

  auto set_heartbeat_callback = [&](auto) {
    for (champsim::modules::core_module& cpu : gen_environment->typed_view<champsim::modules::core_module>("core")) {
      cpu.quiet(true);
    }
  };

  // Re-parse with full validation now that NUM_CPUS is known
  CLI::App app2{"A microarchitecture simulator for research and education"};
  app2.add_option("--config", config_file_path, "Path to the JSON configuration file");
  app2.add_flag("-c,--cloudsuite", knob_cloudsuite, "Read all traces using the cloudsuite format");
  app2.add_flag("--dump", knob_dump, "Print each module builder's parameters as modules are constructed");
  app2.add_flag("--hide-heartbeat", set_heartbeat_callback, "Hide the heartbeat output");
  warmup_instr_option = app2.add_option("-w,--warmup-instructions", warmup_instructions, "The number of instructions in the warmup phase");
  deprec_warmup_instr_option =
      app2.add_option("--warmup_instructions", warmup_instructions, "[deprecated] use --warmup-instructions instead")->excludes(warmup_instr_option);
  sim_instr_option = app2.add_option("-i,--simulation-instructions", simulation_instructions,
                                          "The number of instructions in the detailed phase. If not specified, run to the end of the trace.");
  deprec_sim_instr_option =
      app2.add_option("--simulation_instructions", simulation_instructions, "[deprecated] use --simulation-instructions instead")->excludes(sim_instr_option);
  json_option =
      app2.add_option("--json", json_file_name, "The name of the file to receive JSON output. If no name is specified, stdout will be used")->expected(0, 1);
  app2.add_option("--listeners", requested_listeners, "A list of the listeners to be attached to the run");
  app2.add_option("traces", trace_names, "The paths to the traces")->required()->expected((int)NUM_CPUS)->check(CLI::ExistingFile);

  CLI11_PARSE(app2, argc, argv);

  init_event_listeners(requested_listeners);

  const bool warmup_given = (warmup_instr_option->count() > 0) || (deprec_warmup_instr_option->count() > 0);
  const bool simulation_given = (sim_instr_option->count() > 0) || (deprec_sim_instr_option->count() > 0);

  if (deprec_warmup_instr_option->count() > 0) {
    fmt::print("WARNING: option --warmup_instructions is deprecated. Use --warmup-instructions instead.\n");
  }

  if (deprec_sim_instr_option->count() > 0) {
    fmt::print("WARNING: option --simulation_instructions is deprecated. Use --simulation-instructions instead.\n");
  }

  if (simulation_given && !warmup_given) {
    // Warmup is 20% by default
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    warmup_instructions = simulation_instructions / 5;
  }

  std::vector<champsim::tracereader> traces;
  std::transform(
      std::begin(trace_names), std::end(trace_names), std::back_inserter(traces),
      [knob_cloudsuite, repeat = simulation_given, i = uint8_t(0)](auto name) mutable { return get_tracereader(name, i++, knob_cloudsuite, repeat); });

  std::vector<champsim::phase_info> phases{
      {champsim::phase_info{"Warmup", true, static_cast<uint64_t>(warmup_instructions), std::vector<std::size_t>(std::size(trace_names), 0), trace_names},
       champsim::phase_info{"Simulation", false, static_cast<uint64_t>(simulation_instructions), std::vector<std::size_t>(std::size(trace_names), 0), trace_names}}};

  for (auto& p : phases) {
    std::iota(std::begin(p.trace_index), std::end(p.trace_index), 0);
  }

  fmt::print("\n*** ChampSim Multicore Out-of-Order Simulator ***\nWarmup Instructions: {}\nSimulation Instructions: {}\nNumber of CPUs: {}\nPage size: {}\n\n",
             phases.at(0).length, phases.at(1).length, std::size(gen_environment->typed_view<champsim::modules::core_module>("core")), PAGE_SIZE);

  auto phase_stats = champsim::main(*gen_environment, phases, traces);

  fmt::print("\nChampSim completed all CPUs\n\n");

  champsim::plain_printer{std::cout}.print(phase_stats);

  for (champsim::operable& op : gen_environment->typed_view<champsim::operable>("operable")) {
    op.end_simulation();
  }

  if (json_option->count() > 0) {
    if (json_file_name.empty()) {
      champsim::json_printer{std::cout}.print(phase_stats);
    } else {
      std::ofstream json_file{json_file_name};
      champsim::json_printer{json_file}.print(phase_stats);
    }
  }

  return 0;
}
