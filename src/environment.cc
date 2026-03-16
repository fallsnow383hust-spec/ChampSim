/*
 * Explicit environment implementation for ChampSim.
 * Reads a hierarchical JSON configuration where each module explicitly specifies
 * its name, interface type ("module"), and model ("model"). References to other
 * modules use "@name" syntax and are resolved in declaration order.
 */

#include "environment.h"

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "champsim.h"
#include "chrono.h"
#include "bandwidth.h"
#include "util/bits.h"

using json = nlohmann::json;
using namespace champsim::modules;

namespace {

// Parse prefetch_activate array ["LOAD","PREFETCH"] into access_type vector
std::vector<access_type> parse_pref_activate(const json& j) {
  std::vector<access_type> result;
  if (!j.is_array()) return result;
  for (auto& elem : j) {
    std::string s = elem.get<std::string>();
    if (s == "LOAD") result.push_back(access_type::LOAD);
    else if (s == "RFO") result.push_back(access_type::RFO);
    else if (s == "PREFETCH") result.push_back(access_type::PREFETCH);
    else if (s == "WRITE") result.push_back(access_type::WRITE);
    else if (s == "TRANSLATION") result.push_back(access_type::TRANSLATION);
  }
  return result;
}

// Try to parse a JSON object as a type-wrapped value, e.g. {"picoseconds": 250}
// Returns true and sets the std::any result if recognized.
bool try_parse_typed_value(const json& obj, std::any& out) {
  if (!obj.is_object() || obj.size() != 1) return false;
  auto it = obj.begin();
  const std::string& type_key = it.key();
  const json& val = it.value();

  if (type_key == "picoseconds") {
    out = champsim::chrono::picoseconds{val.get<int64_t>()};
    return true;
  } else if (type_key == "microseconds") {
    out = champsim::chrono::microseconds{val.get<int64_t>()};
    return true;
  } else if (type_key == "bits") {
    out = champsim::data::bits{static_cast<unsigned>(val.get<int64_t>())};
    return true;
  } else if (type_key == "bytes") {
    out = champsim::data::bytes{static_cast<int>(val.get<int64_t>())};
    return true;
  } else if (type_key == "bandwidth") {
    out = champsim::bandwidth::maximum_type{val.get<long long>()};
    return true;
  } else if (type_key == "optional_uint64") {
    if (val.is_null()) out = std::optional<uint64_t>{};
    else out = std::optional<uint64_t>{val.get<uint64_t>()};
    return true;
  }
  return false;
}

// Check if a string is an @-reference
bool is_ref(const std::string& s) { return !s.empty() && s[0] == '@'; }
std::string ref_name(const std::string& s) { return s.substr(1); }

// Check if a JSON array is entirely @-references
bool is_ref_array(const json& arr) {
  if (!arr.is_array() || arr.empty()) return false;
  for (auto& elem : arr) {
    if (!elem.is_string() || !is_ref(elem.get<std::string>())) return false;
  }
  return true;
}

} // anonymous namespace

// Register as "EXPLICIT_ENVIRONMENT"
static environment_module::register_module<champsim::environment> explicit_env_register("EXPLICIT_ENVIRONMENT");

champsim::environment::environment(ModuleBuilder builder)
{
  builder_params_[(builder.get_name().empty() ? "ENVIRONMENT" : builder.get_name())] = builder;
  json config = builder.get_parameter<json>("config_json");
  bool do_dump = builder.get_dump();

  block_size_ = config.value("block_size", 64u);
  page_size_ = config.value("page_size", 4096u);

  if (!config.contains("children")) {
    fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: config must contain a 'children' array\n");
    std::exit(-1);
  }

  auto& children = config["children"];

  for (auto& child : children) {
    if (!child.contains("name") || !child.contains("module") || !child.contains("model")) {
      fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: each child must have 'name', 'module', and 'model'\n");
      std::exit(-1);
    }

    std::string name = child["name"].get<std::string>();
    std::string iface = child["module"].get<std::string>();
    std::string model = child["model"].get<std::string>();

    auto mod_builder = ModuleBuilder{name, model, static_cast<environment_module*>(this)};
    if (do_dump) mod_builder.enable_dump();

    // Process all JSON parameters (skip reserved keys)
    for (auto& [key, val] : child.items()) {
      if (key == "name" || key == "module" || key == "model" || key == "children") continue;

      if (val.is_null()) {
        // Null module pointer: determine type from known parameter names
        if (key == "lower_level" || key == "lower_translate" ||
            key == "fetch_queues" || key == "data_queues") {
          mod_builder.add_parameter(key, static_cast<channel_module*>(nullptr));
        } else if (key == "l1i" || key == "l1d") {
          mod_builder.add_parameter(key, static_cast<cache_module*>(nullptr));
        } else if (key == "vmem") {
          mod_builder.add_parameter(key, static_cast<vmem_module*>(nullptr));
        } else if (key == "dram") {
          mod_builder.add_parameter(key, static_cast<memory_controller_module*>(nullptr));
        }
      } else if (val.is_string() && is_ref(val.get<std::string>())) {
        // Single @-reference: resolve to pointer
        std::string rn = ref_name(val.get<std::string>());
        auto mit = modules_by_name_.find(rn);
        if (mit == modules_by_name_.end()) {
          fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: @-reference '{}' not found (used in '{}' param '{}')\n", rn, name, key);
          std::exit(-1);
        }
        mod_builder.add_raw_parameter(key, mit->second);
      } else if (val.is_array() && is_ref_array(val)) {
        // Array of @-references: resolve to vector of typed pointers
        std::vector<std::any> refs;
        std::string ref_iface;
        for (auto& elem : val) {
          std::string rn = ref_name(elem.get<std::string>());
          auto mit = modules_by_name_.find(rn);
          if (mit == modules_by_name_.end()) {
            fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: @-reference '{}' not found (in array param '{}' of '{}')\n", rn, key, name);
            std::exit(-1);
          }
          std::string curr_iface = module_interfaces_.at(rn);
          if (ref_iface.empty()) {
            ref_iface = curr_iface;
          } else if (curr_iface != ref_iface) {
            fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: mixed interface types in array '{}' of '{}': expected '{}', got '{}' for '{}'\n",
                       key, name, ref_iface, curr_iface, rn);
            std::exit(-1);
          }
          refs.push_back(mit->second);
        }
        mod_builder.add_raw_parameter(key, interface_registry::make_vector(ref_iface, refs));
      } else if (val.is_object()) {
        // Check for type-wrapped value, e.g. {"picoseconds": 250}
        std::any typed_val;
        if (try_parse_typed_value(val, typed_val)) {
          mod_builder.add_raw_parameter(key, std::move(typed_val));
        } else {
          // Store as json object (for complex nested params)
          mod_builder.add_parameter(key, val);
        }
      } else if (val.is_boolean()) {
        mod_builder.add_parameter(key, val.get<bool>());
      } else if (val.is_number_integer()) {
        mod_builder.add_parameter(key, val.get<int64_t>());
      } else if (val.is_number_float()) {
        mod_builder.add_parameter(key, val.get<double>());
      } else if (val.is_string()) {
        // Special handling for pref_activate_mask as comma-separated string
        mod_builder.add_parameter(key, val.get<std::string>());
      } else if (val.is_array()) {
        // Non-ref array: check for string arrays, numeric arrays-of-arrays, etc.
        if (!val.empty() && val[0].is_string()) {
          if (key == "pref_activate_mask") {
            mod_builder.add_parameter(key, parse_pref_activate(val));
          } else {
            std::vector<std::string> sv;
            for (auto& e : val) sv.push_back(e.get<std::string>());
            mod_builder.add_parameter(key, sv);
          }
        } else if (key == "pscl_dims" && !val.empty() && val[0].is_array()) {
          // Array of [level, set, way] triples → std::array<std::array<uint32_t, 3>, 16>
          std::array<std::array<uint32_t, 3>, 16> dims{};
          for (std::size_t i = 0; i < val.size() && i < 16; i++) {
            for (std::size_t j = 0; j < val[i].size() && j < 3; j++) {
              dims[i][j] = static_cast<uint32_t>(val[i][j].get<int64_t>());
            }
          }
          mod_builder.add_parameter(key, dims);
        } else {
          mod_builder.add_parameter(key, val);
        }
      }
    }

    // Handle nested children: extract sub-module declarations for the parent module
    if (child.contains("children")) {
      // Group children by interface type
      std::vector<std::string> prefetcher_models, replacement_models;
      std::vector<std::string> bp_models, btb_models;
      ModuleBuilder::nested_params_type prefetcher_params, replacement_params;
      ModuleBuilder::nested_params_type bp_params, btb_params;

      for (auto& sub : child["children"]) {
        std::string sub_iface = sub["module"].get<std::string>();
        std::string sub_model = sub["model"].get<std::string>();

        // Extract extra parameters (beyond name/module/model)
        std::map<std::string, std::any> extra;
        for (auto& [sk, sv] : sub.items()) {
          if (sk == "name" || sk == "module" || sk == "model") continue;
          if (sv.is_boolean()) extra[sk] = sv.get<bool>();
          else if (sv.is_number_integer()) extra[sk] = sv.get<int64_t>();
          else if (sv.is_number_float()) extra[sk] = sv.get<double>();
          else if (sv.is_string()) extra[sk] = sv.get<std::string>();
        }

        if (sub_iface == "prefetcher") {
          prefetcher_models.push_back(sub_model);
          if (!extra.empty()) prefetcher_params[sub_model] = extra;
        } else if (sub_iface == "replacement") {
          replacement_models.push_back(sub_model);
          if (!extra.empty()) replacement_params[sub_model] = extra;
        } else if (sub_iface == "branch_predictor") {
          bp_models.push_back(sub_model);
          if (!extra.empty()) bp_params[sub_model] = extra;
        } else if (sub_iface == "btb") {
          btb_models.push_back(sub_model);
          if (!extra.empty()) btb_params[sub_model] = extra;
        }
      }

      if (!prefetcher_models.empty()) {
        mod_builder.add_parameter("prefetcher_modules", prefetcher_models);
        mod_builder.add_parameter("prefetcher_params", prefetcher_params);
      }
      if (!replacement_models.empty()) {
        mod_builder.add_parameter("replacement_modules", replacement_models);
        mod_builder.add_parameter("replacement_params", replacement_params);
      }
      if (!bp_models.empty()) {
        mod_builder.add_parameter("bp_impls", bp_models);
        mod_builder.add_parameter("bp_params", bp_params);
      }
      if (!btb_models.empty()) {
        mod_builder.add_parameter("btb_impls", btb_models);
        mod_builder.add_parameter("btb_params", btb_params);
      }
    }

    // Create the module via the interface registry
    std::any typed_ptr = interface_registry::create(iface, mod_builder);
    modules_by_name_[name] = typed_ptr;
    module_interfaces_[name] = iface;
    builder_params_[name] = mod_builder;

    // Store in the appropriate collection
    if (iface == "channel") {
      channels_.push_back(std::any_cast<channel_module*>(typed_ptr));
    } else if (iface == "cache") {
      caches_.push_back(std::any_cast<cache_module*>(typed_ptr));
    } else if (iface == "memory_controller") {
      if (DRAM_ != nullptr) {
        fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: multiple memory_controller modules declared (duplicate: '{}')\n", name);
        std::exit(-1);
      }
      DRAM_ = std::any_cast<memory_controller_module*>(typed_ptr);
    } else if (iface == "vmem") {
      vmem_ = std::any_cast<vmem_module*>(typed_ptr);
    } else if (iface == "page_table_walker") {
      ptws_.push_back(std::any_cast<page_table_walker_module*>(typed_ptr));
    } else if (iface == "core") {
      cores_.push_back(std::any_cast<core_module*>(typed_ptr));
    }
  }

  num_cpus_ = cores_.size();
}

// ====== View functions ======

auto champsim::environment::cpu_view() -> std::vector<std::reference_wrapper<champsim::modules::core_module>>
{
  std::vector<std::reference_wrapper<champsim::modules::core_module>> retval;
  auto make_ref = [](auto* x) { return std::ref(*x); };
  std::transform(std::begin(cores_), std::end(cores_), std::back_inserter(retval), make_ref);
  return retval;
}

auto champsim::environment::cache_view() -> std::vector<std::reference_wrapper<champsim::modules::cache_module>>
{
  std::vector<std::reference_wrapper<champsim::modules::cache_module>> retval;
  auto make_ref = [](auto* x) { return std::ref(*x); };
  std::transform(std::begin(caches_), std::end(caches_), std::back_inserter(retval), make_ref);
  return retval;
}

auto champsim::environment::ptw_view() -> std::vector<std::reference_wrapper<champsim::modules::page_table_walker_module>>
{
  std::vector<std::reference_wrapper<champsim::modules::page_table_walker_module>> retval;
  auto make_ref = [](auto* x) { return std::ref(*x); };
  std::transform(std::begin(ptws_), std::end(ptws_), std::back_inserter(retval), make_ref);
  return retval;
}

auto champsim::environment::dram_view() -> champsim::modules::memory_controller_module&
{
  return *DRAM_;
}

auto champsim::environment::operable_view() -> std::vector<std::reference_wrapper<champsim::operable>>
{
  std::vector<std::reference_wrapper<champsim::operable>> retval;
  auto make_ref = [](auto* x) { return std::ref<champsim::operable>(*x); };
  std::transform(std::begin(cores_), std::end(cores_), std::back_inserter(retval), make_ref);
  std::transform(std::begin(caches_), std::end(caches_), std::back_inserter(retval), make_ref);
  std::transform(std::begin(ptws_), std::end(ptws_), std::back_inserter(retval), make_ref);
  retval.push_back(std::ref<champsim::operable>(*DRAM_));
  return retval;
}
