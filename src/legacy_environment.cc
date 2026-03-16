/*
 * Default environment implementation for ChampSim.
 * Reads a JSON configuration file and constructs the entire simulation hierarchy
 * using the module system, mirroring what the Python config scripts generate at compile time.
 */

#include "legacy_environment.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "champsim.h"
#include "chrono.h"
#include "defaults.hpp"
#include "util/bits.h"

using json = nlohmann::json;
using namespace champsim::modules;

namespace {

// Helper: convert frequency in MHz to clock period in picoseconds
int64_t freq_to_period(double freq_mhz) {
  return static_cast<int64_t>(1000000.0 / freq_mhz);
}

// Helper: parse a size value that may be an integer or a string with suffix (e.g. "2MB", "4kB", "64B")
int64_t parse_size_value(const json& val) {
  if (val.is_number()) return val.get<int64_t>();
  if (!val.is_string()) return 0;
  auto s = val.get<std::string>();
  // Ordered longest-suffix-first to avoid prefix ambiguity (e.g. "kB" before "k")
  static const std::pair<std::string, int64_t> suffixes[] = {
    {"TiB", int64_t{1} << 40}, {"TB", int64_t{1} << 40},
    {"GiB", int64_t{1} << 30}, {"GB", int64_t{1} << 30},
    {"MiB", int64_t{1} << 20}, {"MB", int64_t{1} << 20},
    {"kiB", int64_t{1} << 10}, {"kB", int64_t{1} << 10},
    {"T", int64_t{1} << 40}, {"G", int64_t{1} << 30},
    {"M", int64_t{1} << 20}, {"k", int64_t{1} << 10},
    {"B", 1}
  };
  for (auto& [suffix, mult] : suffixes) {
    if (s.size() > suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0) {
      return std::stoll(s.substr(0, s.size() - suffix.size())) * mult;
    }
  }
  return std::stoll(s);
}

// Helper: parse prefetch_activate string like "LOAD,PREFETCH" into access_type vector
std::vector<access_type> parse_pref_activate(const json& j) {
  std::vector<access_type> result;
  std::string s;
  if (j.is_string()) {
    s = j.get<std::string>();
  } else if (j.is_array()) {
    for (auto& elem : j) {
      s += elem.get<std::string>() + ",";
    }
  }
  // Parse comma-separated values
  std::string token;
  for (char c : s) {
    if (c == ',') {
      if (!token.empty()) {
        if (token == "LOAD") result.push_back(access_type::LOAD);
        else if (token == "RFO") result.push_back(access_type::RFO);
        else if (token == "PREFETCH") result.push_back(access_type::PREFETCH);
        else if (token == "WRITE") result.push_back(access_type::WRITE);
        else if (token == "TRANSLATION") result.push_back(access_type::TRANSLATION);
        token.clear();
      }
    } else if (c != ' ') {
      token += c;
    }
  }
  if (!token.empty()) {
    if (token == "LOAD") result.push_back(access_type::LOAD);
    else if (token == "RFO") result.push_back(access_type::RFO);
    else if (token == "PREFETCH") result.push_back(access_type::PREFETCH);
    else if (token == "WRITE") result.push_back(access_type::WRITE);
    else if (token == "TRANSLATION") result.push_back(access_type::TRANSLATION);
  }
  return result;
}

// Helper: parse replacement/prefetcher/branch/btb module name(s) from JSON into string vector
std::vector<std::string> parse_module_list(const json& j, const std::string& key, const std::string& default_val) {
  if (!j.contains(key)) return {default_val};
  auto& v = j[key];
  if (v.is_string()) return {v.get<std::string>()};
  if (v.is_array()) {
    std::vector<std::string> result;
    for (auto& elem : v) {
      if (elem.is_string()) result.push_back(elem.get<std::string>());
      else if (elem.is_object() && elem.contains("model")) result.push_back(elem["model"].get<std::string>());
    }
    return result;
  }
  if (v.is_object() && v.contains("model")) return {v["model"].get<std::string>()};
  return {default_val};
}

// Helper: extract per-module parameter maps from a nested module spec
// Supports: "prefetcher": "no", "prefetcher": {"model": "ip_stride", "degree": 4},
//           "prefetcher": ["no", {"model": "ip_stride", "degree": 4}]
// Returns map from model name -> ModuleBuilder containing per-model params
using param_map_type = ModuleBuilder::module_builder_map_type;
param_map_type parse_module_params(const json& j, const std::string& key) {
  param_map_type result;
  if (!j.contains(key)) return result;
  auto& v = j[key];

  auto extract_params = [](const json& obj) -> ModuleBuilder {
    auto model = obj.contains("model") ? obj["model"].get<std::string>() : std::string{};
    ModuleBuilder params{"", model, nullptr};
    for (auto& [k, val] : obj.items()) {
      if (k == "model") continue;
      if (val.is_number_integer()) params.add_parameter(k, val.get<int64_t>());
      else if (val.is_number_unsigned()) params.add_parameter(k, val.get<uint64_t>());
      else if (val.is_number_float()) params.add_parameter(k, val.get<double>());
      else if (val.is_boolean()) params.add_parameter(k, val.get<bool>());
      else if (val.is_string()) params.add_parameter(k, val.get<std::string>());
    }
    return params;
  };

  if (v.is_object() && v.contains("model")) {
    result[v["model"].get<std::string>()] = extract_params(v);
  } else if (v.is_array()) {
    for (auto& elem : v) {
      if (elem.is_object() && elem.contains("model")) {
        result[elem["model"].get<std::string>()] = extract_params(elem);
      }
    }
  }
  return result;
}

// Helper: forward all JSON scalar values to a ModuleBuilder.
// For ints: stored as int64_t (numeric_any_cast handles conversion to uint32_t, size_t, etc.)
// For objects/arrays: stored as json (useful for nested module specs)
// renames maps JSON key names to builder parameter names.
void add_json_params(ModuleBuilder& builder, const json& j,
                     const std::map<std::string, std::string>& renames = {}) {
  for (auto& [key, val] : j.items()) {
    std::string param_name = key;
    if (auto it = renames.find(key); it != renames.end())
      param_name = it->second;
    if (val.is_boolean())            builder.add_parameter(param_name, val.get<bool>());
    else if (val.is_number_integer()) builder.add_parameter(param_name, val.get<int64_t>());
    else if (val.is_number_float())  builder.add_parameter(param_name, val.get<double>());
    else if (val.is_string())        builder.add_parameter(param_name, val.get<std::string>());
    else                             builder.add_parameter(param_name, val); // objects, arrays
  }
}

// Helper: set a bandwidth::maximum_type parameter from JSON if present
void json_bandwidth(ModuleBuilder& builder, const json& j,
                    const std::string& json_key, const std::string& param_name) {
  if (j.contains(json_key))
    builder.add_parameter(param_name, champsim::bandwidth::maximum_type{j[json_key].get<long long>()});
}

// Struct to hold info about a cache before construction
struct cache_config {
  std::string name;
  std::string model = "DEFAULT_CACHE";
  std::string lower_level;     // name of lower-level cache or "DRAM"
  std::string lower_translate;  // name of TLB cache for translation
  ModuleBuilder defaults_builder;
  bool is_tlb = false;
  bool first_level = false;
  bool is_instruction_cache = false;

  // JSON-specified overrides (empty if not provided - use defaults)
  json config;

  // Queue sizing
  std::size_t queue_factor = 32;

  // Computed
  int frequency = 4000;
};

// Struct to hold PTW info
struct ptw_config {
  std::string name;
  std::string model = "DEFAULT_PTW";
  std::string lower_level; // the L1D cache this PTW uses
  int cpu_index = 0;
  int frequency = 4000;
  std::size_t queue_factor = 32;
  json config;
};

// Struct to hold core info
struct core_config {
  std::string name;
  std::string model = "DEFAULT_CORE";
  int index = 0;
  int frequency = 4000;
  std::string l1i_name;
  std::string l1d_name;
  std::string itlb_name;
  std::string dtlb_name;
  std::string l2c_name;
  std::string stlb_name;
  std::string ptw_name;
  json config;
};

// An upper-lower pair for channel construction
struct ul_pair {
  std::string lower_name;
  std::string upper_name;
};

} // anonymous namespace


// Register environment as "DEFAULT_ENVIRONMENT" environment model
static champsim::modules::environment_module::register_module<champsim::legacy_environment> default_env_register("DEFAULT_ENVIRONMENT");

champsim::legacy_environment::legacy_environment(champsim::modules::ModuleBuilder builder)
{
  // Store the environment builder itself
  builder_params_[(builder.get_name().empty() ? "ENVIRONMENT" : builder.get_name())] = builder;

  // Local variables (formerly member variables)
  std::vector<channel_module*> channels;
  memory_controller_module* DRAM = nullptr;
  vmem_module* vmem = nullptr;
  std::vector<page_table_walker_module*> ptws;
  std::vector<cache_module*> caches;
  std::vector<core_module*> cores;

  // Accept pre-parsed JSON from the builder
  json config = builder.get_parameter<json>("config_json");

  // Extract root config (support suffixed sizes like "64B", "4kB")
  block_size_ = config.contains("block_size") ? static_cast<unsigned>(parse_size_value(config["block_size"])) : 64u;
  page_size_ = config.contains("page_size") ? static_cast<unsigned>(parse_size_value(config["page_size"])) : 4096u;
  unsigned log2_block_size = static_cast<unsigned>(champsim::lg2(block_size_));
  unsigned log2_page_size = static_cast<unsigned>(champsim::lg2(page_size_));
  std::size_t num_cores_cfg = config.value("num_cores", 1u);
  num_cpus_ = num_cores_cfg;

  // Parse cores from JSON
  auto cpu_json_array = config.value("ooo_cpu", json::array({json::object()}));
  // Duplicate to fill num_cores if needed
  while (cpu_json_array.size() < num_cores_cfg) {
    std::size_t src_idx = cpu_json_array.size() % cpu_json_array.size();
    cpu_json_array.push_back(cpu_json_array[src_idx]);
  }

  // Build core configs
  std::vector<core_config> core_cfgs;
  for (std::size_t i = 0; i < num_cores_cfg; i++) {
    core_config cc;
    cc.index = static_cast<int>(i);
    cc.name = fmt::format("cpu{}", i);
    cc.frequency = cpu_json_array[i].value("frequency", 4000);
    cc.config = cpu_json_array[i];

    cc.l1i_name = fmt::format("{}_L1I", cc.name);
    cc.l1d_name = fmt::format("{}_L1D", cc.name);
    cc.itlb_name = fmt::format("{}_ITLB", cc.name);
    cc.dtlb_name = fmt::format("{}_DTLB", cc.name);
    cc.l2c_name = fmt::format("{}_L2C", cc.name);
    cc.stlb_name = fmt::format("{}_STLB", cc.name);
    cc.ptw_name = fmt::format("{}_PTW", cc.name);

    core_cfgs.push_back(cc);
  }

  // Read DIB config from JSON
  json dib_json = config.value("DIB", json::object());

  // Build cache configs - for each core: L1I, L1D, ITLB, DTLB, L2C, STLB, and a shared LLC
  std::map<std::string, cache_config> cache_cfgs;

  // Read per-cache-type JSON overrides from config file
  json l1i_json = config.value("L1I", json::object());
  json l1d_json = config.value("L1D", json::object());
  json l2c_json = config.value("L2C", json::object());
  json itlb_json = config.value("ITLB", json::object());
  json dtlb_json = config.value("DTLB", json::object());
  json stlb_json = config.value("STLB", json::object());
  json llc_json = config.value("LLC", json::object());
  json ptw_json = config.value("PTW", json::object());

  // Build per-core caches
  for (auto& cc : core_cfgs) {
    int core_freq = cc.frequency;

    // L1I
    {
      cache_config c;
      c.name = cc.l1i_name;
      c.lower_level = cc.l2c_name;
      c.lower_translate = cc.itlb_name;
      c.defaults_builder = champsim::defaults::default_l1i();
      c.is_instruction_cache = true;
      c.first_level = true;
      c.queue_factor = 32;
      c.frequency = core_freq;
      c.config = l1i_json;
      cache_cfgs[c.name] = c;
    }
    // L1D
    {
      cache_config c;
      c.name = cc.l1d_name;
      c.lower_level = cc.l2c_name;
      c.lower_translate = cc.dtlb_name;
      c.defaults_builder = champsim::defaults::default_l1d();
      c.first_level = true;
      c.queue_factor = 32;
      c.frequency = core_freq;
      c.config = l1d_json;
      cache_cfgs[c.name] = c;
    }
    // ITLB
    {
      cache_config c;
      c.name = cc.itlb_name;
      c.lower_level = cc.stlb_name;
      c.defaults_builder = champsim::defaults::default_itlb();
      c.is_tlb = true;
      c.first_level = true;
      c.queue_factor = 16;
      c.frequency = core_freq;
      c.config = itlb_json;
      cache_cfgs[c.name] = c;
    }
    // DTLB
    {
      cache_config c;
      c.name = cc.dtlb_name;
      c.lower_level = cc.stlb_name;
      c.defaults_builder = champsim::defaults::default_dtlb();
      c.is_tlb = true;
      c.first_level = true;
      c.queue_factor = 16;
      c.frequency = core_freq;
      c.config = dtlb_json;
      cache_cfgs[c.name] = c;
    }
    // L2C
    {
      cache_config c;
      c.name = cc.l2c_name;
      c.lower_level = "LLC";
      c.lower_translate = cc.stlb_name;
      c.defaults_builder = champsim::defaults::default_l2c();
      c.queue_factor = 16;
      c.frequency = core_freq;
      c.config = l2c_json;
      cache_cfgs[c.name] = c;
    }
    // STLB
    {
      cache_config c;
      c.name = cc.stlb_name;
      c.lower_level = cc.ptw_name;
      c.defaults_builder = champsim::defaults::default_stlb();
      c.is_tlb = true;
      c.queue_factor = 16;
      c.frequency = core_freq;
      c.config = stlb_json;
      cache_cfgs[c.name] = c;
    }
  }

  // LLC (shared)
  {
    cache_config c;
    c.name = "LLC";
    c.lower_level = "DRAM";
    c.defaults_builder = champsim::defaults::default_llc();
    c.queue_factor = 32;
    c.frequency = llc_json.value("frequency", core_cfgs.empty() ? 4000 : core_cfgs[0].frequency);
    c.config = llc_json;
    cache_cfgs[c.name] = c;
  }

  // Build PTW configs
  std::vector<ptw_config> ptw_cfgs;
  for (auto& cc : core_cfgs) {
    ptw_config pc;
    pc.name = cc.ptw_name;
    pc.lower_level = cc.l1d_name;
    pc.cpu_index = cc.index;
    pc.frequency = cc.frequency;
    pc.queue_factor = 32;
    pc.config = ptw_json;
    ptw_cfgs.push_back(pc);
  }

  // Determine the order caches will be built - we use a fixed order matching the generated code:
  // LLC, then per-core: DTLB, ITLB, L1D, L1I, L2C, STLB
  std::vector<std::string> cache_build_order;
  cache_build_order.push_back("LLC");
  for (auto& cc : core_cfgs) {
    cache_build_order.push_back(cc.dtlb_name);
    cache_build_order.push_back(cc.itlb_name);
    cache_build_order.push_back(cc.l1d_name);
    cache_build_order.push_back(cc.l1i_name);
    cache_build_order.push_back(cc.l2c_name);
    cache_build_order.push_back(cc.stlb_name);
  }

  // Build the upper-level pairs (lower_name, upper_name) for channel construction
  // This follows the same order as the generated code:
  // For PTWs: (lower_level, ptw_name)
  // For caches: (lower_level, cache_name), (lower_translate, cache_name) if present
  // For cores: (L1I, core_name), (L1D, core_name)
  std::vector<ul_pair> ul_pairs;

  // PTW lower levels
  for (auto& ptw : ptw_cfgs) {
    ul_pairs.push_back({ptw.lower_level, ptw.name});
  }
  // Cache lower levels
  for (auto& cache_name : cache_build_order) {
    auto& cc = cache_cfgs[cache_name];
    ul_pairs.push_back({cc.lower_level, cc.name});
    if (!cc.lower_translate.empty()) {
      ul_pairs.push_back({cc.lower_translate, cc.name});
    }
  }
  // Core fetch/data channels
  for (auto& core : core_cfgs) {
    ul_pairs.push_back({core.l1i_name, core.name});
    ul_pairs.push_back({core.l1d_name, core.name});
  }

  // Helper to find index of a ul_pair
  auto find_ul_index = [&](const std::string& lower, const std::string& upper) -> std::size_t {
    for (std::size_t i = 0; i < ul_pairs.size(); i++) {
      if (ul_pairs[i].lower_name == lower && ul_pairs[i].upper_name == upper)
        return i;
    }
    fmt::print("ERROR: Could not find channel for pair ({}, {})\n", lower, upper);
    std::exit(-1);
  };

  // Helper to find all ul_pairs where lower == name
  auto find_upper_indices = [&](const std::string& lower_name) -> std::vector<std::size_t> {
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < ul_pairs.size(); i++) {
      if (ul_pairs[i].lower_name == lower_name)
        result.push_back(i);
    }
    return result;
  };

  // ====== Build channels ======
  // One channel per ul_pair
  channels.reserve(ul_pairs.size());
  for (std::size_t i = 0; i < ul_pairs.size(); i++) {
    auto& pair = ul_pairs[i];
    std::string ch_name = pair.upper_name + "_" + pair.lower_name + "_channel";

    // Determine queue sizes and offset bits for this channel
    std::size_t rq_size = 32, pq_size = 32, wq_size = 32;
    unsigned offset_bits = log2_block_size;
    bool match_offset = false;

    // Check if the lower is a cache
    auto cache_it = cache_cfgs.find(pair.lower_name);
    if (cache_it != cache_cfgs.end()) {
      auto& cc = cache_it->second;
      rq_size = cc.config.value("rq_size", static_cast<int>(cc.queue_factor));
      wq_size = cc.config.value("wq_size", static_cast<int>(cc.queue_factor));
      pq_size = cc.config.value("pq_size", static_cast<int>(cc.queue_factor));
      offset_bits = cc.is_tlb ? log2_page_size : log2_block_size;
      match_offset = cc.first_level || cc.config.value("wq_check_full_addr", false);
    }
    // Check if the lower is a PTW
    else {
      bool is_ptw = false;
      for (auto& ptw : ptw_cfgs) {
        if (ptw.name == pair.lower_name) {
          rq_size = ptw.config.value("rq_size", static_cast<int>(ptw.queue_factor));
          wq_size = 0;
          pq_size = 0;
          offset_bits = log2_page_size;
          match_offset = false;
          is_ptw = true;
          break;
        }
      }
      if (!is_ptw && pair.lower_name == "DRAM") {
        // DRAM channel - use DRAM's queue sizes from physical_memory config
        auto pm = config.value("physical_memory", json::object());
        rq_size = pm.value("rq_size", 64);
        wq_size = pm.value("wq_size", 64);
        pq_size = pm.value("pq_size", 64);
        offset_bits = log2_block_size;
        match_offset = false;
      }
      // If it's a cache feeding a core (L1I->core or L1D->core)
      else if (!is_ptw && pair.lower_name != "DRAM") {
        // This is a core->L1I or core->L1D channel
        auto cache_check = cache_cfgs.find(pair.lower_name);
        if (cache_check != cache_cfgs.end()) {
          auto& cc2 = cache_check->second;
          rq_size = cc2.config.value("rq_size", static_cast<int>(cc2.queue_factor));
          wq_size = cc2.config.value("wq_size", static_cast<int>(cc2.queue_factor));
          pq_size = cc2.config.value("pq_size", static_cast<int>(cc2.queue_factor));
          offset_bits = cc2.is_tlb ? log2_page_size : log2_block_size;
          match_offset = true;
        }
      }
    }

    auto ch_builder = ModuleBuilder{ch_name, "DEFAULT_CHANNEL", static_cast<environment_module*>(this)};
    ch_builder.add_parameter("rq_size", rq_size)
      .add_parameter("pq_size", pq_size)
      .add_parameter("wq_size", wq_size)
      .add_parameter("offset_bits", champsim::data::bits{offset_bits})
      .add_parameter("match_offset_bits", match_offset);

    builder_params_[ch_name] = ch_builder;
    channels.push_back(module_base<channel_module, environment_module>::create_instance(ch_builder));
  }

  // ====== Build DRAM ======
  json pmem_json = config.value("physical_memory", json::object());
  {
    // Support "frequency" as alias for "data_rate" (frequency == data_rate in DRAM context)
    int data_rate = pmem_json.value("data_rate", pmem_json.value("frequency", 3200));
    auto dram_builder = ModuleBuilder{"DRAM", "DEFAULT_MEMORY_CONTROLLER", static_cast<environment_module*>(this),
                                      champsim::defaults::default_memory_controller()};
    dram_builder
      .add_parameter("dbus_period", champsim::chrono::picoseconds{freq_to_period(data_rate)})
      .add_parameter("mc_period", champsim::chrono::picoseconds{freq_to_period(data_rate / 2)})
      .add_parameter("t_rp", pmem_json.value("tRP", 24))
      .add_parameter("t_rcd", pmem_json.value("tRCD", 24))
      .add_parameter("t_cas", pmem_json.value("tCAS", 24))
      .add_parameter("t_ras", pmem_json.value("tRAS", 52))
      .add_parameter("refresh_period", champsim::chrono::microseconds{1000 * pmem_json.value("refresh_period", 32)})
      .add_parameter("rq_size", pmem_json.value("rq_size", 64))
      .add_parameter("wq_size", pmem_json.value("wq_size", 64))
      .add_parameter("channels", pmem_json.value("channels", 1))
      .add_parameter("channel_width", champsim::data::bytes{pmem_json.value("channel_width", 8)})
      .add_parameter("rows", pmem_json.value("bank_rows", 65536))
      .add_parameter("columns", pmem_json.value("bank_columns", 1024))
      .add_parameter("ranks", pmem_json.value("ranks", 1))
      .add_parameter("bankgroups", pmem_json.value("bankgroups", 8))
      .add_parameter("banks", pmem_json.value("banks", 4))
      .add_parameter("refreshes_per_period", pmem_json.value("refreshes_per_period", 8192));

    std::vector<channel_module*> dram_ul_channels;
    for (auto idx : find_upper_indices("DRAM"))
      dram_ul_channels.push_back(channels.at(idx));
    dram_builder.add_parameter("ul_channels", dram_ul_channels);

    builder_params_["DRAM"] = dram_builder;
    DRAM = module_base<memory_controller_module, environment_module>::create_instance(dram_builder);
  }

  // ====== Build vmem ======
  json vmem_json = config.value("virtual_memory", json::object());
  {
    int max_freq = 0;
    for (auto& cc : core_cfgs) max_freq = std::max(max_freq, cc.frequency);
    for (auto& [name, cc] : cache_cfgs) max_freq = std::max(max_freq, cc.frequency);
    for (auto& pc : ptw_cfgs) max_freq = std::max(max_freq, pc.frequency);
    if (max_freq == 0) max_freq = 4000;
    int64_t global_clock_period = freq_to_period(max_freq);

    auto vmem_builder = ModuleBuilder{"VMEM", "DEFAULT_VMEM", static_cast<environment_module*>(this),
                                      champsim::defaults::default_vmem()};
    vmem_builder
      .add_parameter("page_table_page_size", champsim::data::bytes{
          vmem_json.contains("pte_page_size") ? static_cast<int>(parse_size_value(vmem_json["pte_page_size"])) : 4096})
      .add_parameter("page_table_levels", vmem_json.value("num_levels", 5))
      .add_parameter("minor_fault_penalty", champsim::chrono::picoseconds{global_clock_period * vmem_json.value("minor_fault_penalty", 200)})
      .add_parameter("dram", DRAM);

    auto randomization = vmem_json.value("randomization", 1);
    vmem_builder.add_parameter("randomization_seed",
      randomization == 0 ? std::optional<uint64_t>{} : std::optional<uint64_t>{static_cast<uint64_t>(randomization)});

    builder_params_["VMEM"] = vmem_builder;
    vmem = module_base<vmem_module, environment_module>::create_instance(vmem_builder);
  }

  // ====== Build PTWs ======
  ptws.reserve(ptw_cfgs.size());
  for (auto& pc : ptw_cfgs) {
    auto ptw_builder = ModuleBuilder{pc.name, pc.model, static_cast<environment_module*>(this),
                                     champsim::defaults::default_ptw()};

    std::vector<channel_module*> ptw_ul_channels;
    for (auto idx : find_upper_indices(pc.name))
      ptw_ul_channels.push_back(channels.at(idx));
    ptw_builder.add_parameter("upper_levels", ptw_ul_channels);
    ptw_builder.add_parameter("vmem", vmem);
    ptw_builder.add_parameter("cpu", pc.cpu_index);
    ptw_builder.add_parameter("lower_level", channels.at(find_ul_index(pc.lower_level, pc.name)));
    ptw_builder.add_parameter("clock_period", champsim::chrono::picoseconds{freq_to_period(pc.frequency)});

    // Forward JSON scalar overrides (mshr_size, etc.)
    add_json_params(ptw_builder, pc.config);
    json_bandwidth(ptw_builder, pc.config, "max_read", "max_tag_check");
    json_bandwidth(ptw_builder, pc.config, "max_write", "max_fill");

    // PSCL dimensions
    if (pc.config.contains("pscl5_set")) {
      std::array<std::array<uint32_t, 3>, 16> pscl_dims{};
      pscl_dims[0] = {5, static_cast<uint32_t>(pc.config.value("pscl5_set", 1)), static_cast<uint32_t>(pc.config.value("pscl5_way", 2))};
      pscl_dims[1] = {4, static_cast<uint32_t>(pc.config.value("pscl4_set", 1)), static_cast<uint32_t>(pc.config.value("pscl4_way", 4))};
      pscl_dims[2] = {3, static_cast<uint32_t>(pc.config.value("pscl3_set", 2)), static_cast<uint32_t>(pc.config.value("pscl3_way", 4))};
      pscl_dims[3] = {2, static_cast<uint32_t>(pc.config.value("pscl2_set", 4)), static_cast<uint32_t>(pc.config.value("pscl2_way", 8))};
      ptw_builder.add_parameter("pscl_dims", pscl_dims);
    }

    builder_params_[pc.name] = ptw_builder;
    ptws.push_back(module_base<page_table_walker_module, environment_module>::create_instance(ptw_builder));
  }

  // ====== Build caches ======
  std::map<std::string, std::size_t> cache_index_map;
  static const std::map<std::string, std::string> cache_renames = {
    {"sets", "num_sets"}, {"ways", "num_ways"}
  };

  caches.reserve(cache_build_order.size());
  for (auto& cache_name : cache_build_order) {
    auto& cc = cache_cfgs[cache_name];

    auto cache_builder = ModuleBuilder{cc.name, cc.model, static_cast<environment_module*>(this), cc.defaults_builder};

    std::vector<channel_module*> cache_ul_channels;
    for (auto idx : find_upper_indices(cc.name))
      cache_ul_channels.push_back(channels.at(idx));
    cache_builder.add_parameter("upper_levels", cache_ul_channels);
    cache_builder.add_parameter("offset_bits", champsim::data::bits{cc.is_tlb ? log2_page_size : log2_block_size});

    // Module lists and nested params from JSON
    cache_builder.add_parameter("replacement_modules", parse_module_list(cc.config, "replacement", "lru"));
    cache_builder.add_parameter("prefetcher_modules", parse_module_list(cc.config, "prefetcher", "no"));
    cache_builder.add_parameter("replacement_params", parse_module_params(cc.config, "replacement"));
    cache_builder.add_parameter("prefetcher_params", parse_module_params(cc.config, "prefetcher"));

    cache_builder.add_parameter("lower_level", channels.at(find_ul_index(cc.lower_level, cc.name)));
    if (!cc.lower_translate.empty())
      cache_builder.add_parameter("lower_translate", channels.at(find_ul_index(cc.lower_translate, cc.name)));
    cache_builder.add_parameter("clock_period", champsim::chrono::picoseconds{freq_to_period(cc.frequency)});

    // Latency: support "latency" (total), "hit_latency", "fill_latency" in any combination
    {
      bool has_total = cc.config.contains("latency");
      bool has_hit   = cc.config.contains("hit_latency");
      bool has_fill  = cc.config.contains("fill_latency");
      if (has_total) {
        auto total = cc.config["latency"].get<int64_t>();
        int64_t hit, fill;
        if (has_hit && has_fill) {
          hit  = cc.config["hit_latency"].get<int64_t>();
          fill = cc.config["fill_latency"].get<int64_t>();
        } else if (has_hit) {
          hit  = cc.config["hit_latency"].get<int64_t>();
          fill = total - hit;
        } else if (has_fill) {
          fill = cc.config["fill_latency"].get<int64_t>();
          hit  = total - fill;
        } else {
          hit  = total / 2;
          fill = total - hit;
        }
        cache_builder.add_parameter("hit_latency", std::max(hit, int64_t{1}));
        cache_builder.add_parameter("fill_latency", std::max(fill, int64_t{1}));
      }
    }
    // Derive sets/ways from "size" if specified (e.g. "size": "2MB" or "size": 32768)
    if (cc.config.contains("size")) {
      auto total_bytes = parse_size_value(cc.config["size"]);
      unsigned offset = cc.is_tlb ? log2_page_size : log2_block_size;
      if (!cc.config.contains("sets") && !cc.config.contains("ways")) {
        // Use default ways, derive sets
        auto default_ways = cc.defaults_builder.get_parameter<uint32_t>("num_ways");
        cache_builder.add_parameter("num_sets", champsim::next_pow2(static_cast<uint32_t>(total_bytes / (default_ways * (1u << offset)))));
      } else if (cc.config.contains("ways") && !cc.config.contains("sets")) {
        auto ways = cc.config["ways"].get<uint32_t>();
        cache_builder.add_parameter("num_sets", champsim::next_pow2(static_cast<uint32_t>(total_bytes / (ways * (1u << offset)))));
      } else if (cc.config.contains("sets") && !cc.config.contains("ways")) {
        auto sets = cc.config["sets"].get<uint32_t>();
        cache_builder.add_parameter("num_ways", static_cast<uint32_t>(total_bytes / (sets * (1u << offset))));
      }
      // If both sets and ways are explicit, size is ignored (explicit values win)
    }
    // Forward all JSON scalar overrides (sets, ways, pq_size, mshr_size, hit_latency, fill_latency, bools, etc.)
    add_json_params(cache_builder, cc.config, cache_renames);
    // Bandwidth types need explicit wrapping (enum class)
    json_bandwidth(cache_builder, cc.config, "max_tag_check", "max_tag_bandwidth");
    json_bandwidth(cache_builder, cc.config, "max_fill", "max_fill_bandwidth");
    // prefetch_activate needs special parsing
    if (cc.config.contains("prefetch_activate"))
      cache_builder.add_parameter("pref_activate_mask", parse_pref_activate(cc.config["prefetch_activate"]));

    cache_index_map[cc.name] = caches.size();
    builder_params_[cc.name] = cache_builder;
    caches.push_back(module_base<cache_module, environment_module>::create_instance(cache_builder));
  }

  // ====== Build cores ======
  cores.reserve(core_cfgs.size());
  for (auto& cc : core_cfgs) {
    auto core_builder = ModuleBuilder{cc.name, cc.model, static_cast<environment_module*>(this),
                                      champsim::defaults::default_core()};

    auto l1i_ptr = caches.at(cache_index_map[cc.l1i_name]);
    auto l1d_ptr = caches.at(cache_index_map[cc.l1d_name]);
    core_builder.add_parameter("l1i", l1i_ptr);
    core_builder.add_parameter("l1i_bandwidth", l1i_ptr->get_max_tag_bandwidth());
    core_builder.add_parameter("fetch_queues", channels.at(find_ul_index(cc.l1i_name, cc.name)));
    core_builder.add_parameter("l1d", l1d_ptr);
    core_builder.add_parameter("l1d_bandwidth", l1d_ptr->get_max_tag_bandwidth());
    core_builder.add_parameter("data_queues", channels.at(find_ul_index(cc.l1d_name, cc.name)));

    core_builder.add_parameter("branch_predictor_modules", parse_module_list(cc.config, "branch_predictor", "hashed_perceptron"));
    core_builder.add_parameter("btb_modules", parse_module_list(cc.config, "btb", "basic_btb"));
    core_builder.add_parameter("branch_predictor_params", parse_module_params(cc.config, "branch_predictor"));
    core_builder.add_parameter("btb_params", parse_module_params(cc.config, "btb"));
    core_builder.add_parameter("cpu", cc.index);
    core_builder.add_parameter("clock_period", champsim::chrono::picoseconds{freq_to_period(cc.frequency)});

    // Forward all JSON scalar overrides (buffer sizes, latencies, etc.)
    add_json_params(core_builder, cc.config);
    // Bandwidth types need explicit wrapping (enum class)
    for (auto& key : {"fetch_width", "decode_width", "dispatch_width", "execute_width",
                       "lq_width", "sq_width", "retire_width"})
      json_bandwidth(core_builder, cc.config, key, key);
    json_bandwidth(core_builder, cc.config, "scheduler_size", "schedule_width");
    // DIB parameters (from separate dib_json, not core config)
    add_json_params(core_builder, dib_json, {{"sets", "dib_set"}, {"ways", "dib_way"}, {"window_size", "dib_window"}});

    builder_params_[cc.name] = core_builder;
    cores.push_back(module_base<core_module, environment_module>::create_instance(core_builder));
  }

  // Populate generic storage from local variables
  for (auto* ch : channels) {
    modules_by_type_["channel"].push_back(ch);
    module_order_.emplace_back(ch->NAME, "channel");
  }
  modules_by_type_["memory_controller"].push_back(DRAM);
  module_order_.emplace_back("DRAM", "memory_controller");
  modules_by_type_["vmem"].push_back(vmem);
  module_order_.emplace_back("VMEM", "vmem");
  for (auto* p : ptws) {
    modules_by_type_["page_table_walker"].push_back(p);
    module_order_.emplace_back(p->NAME, "page_table_walker");
  }
  for (auto* c : caches) {
    modules_by_type_["cache"].push_back(c);
    module_order_.emplace_back(c->NAME, "cache");
  }
  for (auto* c : cores) {
    modules_by_type_["core"].push_back(c);
    module_order_.emplace_back(c->NAME, "core");
  }
}

// ====== View function ======

auto champsim::legacy_environment::view(const std::string& interface_type) const -> std::vector<std::any>
{
  if (interface_type == "operable") {
    std::vector<std::any> result;
    // Use a per-type counter to map module_order_ entries to modules_by_type_ indices
    std::map<std::string, std::size_t> type_idx;
    for (auto& [name, iface] : module_order_) {
      auto to_op = champsim::modules::interface_registry::get_to_operable(iface);
      if (!to_op) continue;
      auto& vec = modules_by_type_.at(iface);
      auto idx = type_idx[iface]++;
      result.push_back(static_cast<champsim::operable*>(to_op(vec.at(idx))));
    }
    return result;
  }

  auto it = modules_by_type_.find(interface_type);
  if (it == modules_by_type_.end()) return {};
  return it->second;
}
