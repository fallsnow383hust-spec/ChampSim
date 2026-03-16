/*
 * Explicit environment implementation for ChampSim.
 * Reads a hierarchical JSON configuration where each module explicitly specifies
 * its name, interface type ("module"), and model ("model"). References to other
 * modules use "@name" syntax and are resolved in declaration order.
 *
 * This implementation is fully generic: no interface types or module names are
 * hardcoded. Any registered interface and model will work without alteration.
 */

#include "environment.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "champsim.h"
#include "chrono.h"
#include "access_type.h"
#include "bandwidth.h"
#include "util/bits.h"
#include "util/units.h"

using json = nlohmann::json;
using namespace champsim::modules;

namespace {

// Split a string like "4G" into {4.0, "G"} or "32000" into {32000.0, ""}.
std::pair<double, std::string> parse_number_and_suffix(const std::string& s) {
  std::size_t pos = 0;
  double value = std::stod(s, &pos);
  return {value, s.substr(pos)};
}

// Parse a frequency string with SI suffix → picoseconds period.
// Formula: ps = round(1e12 / (value * multiplier))
champsim::chrono::picoseconds parse_frequency_string(const std::string& s) {
  auto [value, suffix] = parse_number_and_suffix(s);
  double multiplier = 1.0;
  if (suffix.empty())      multiplier = 1.0;
  else if (suffix == "K")  multiplier = 1e3;
  else if (suffix == "M")  multiplier = 1e6;
  else if (suffix == "G")  multiplier = 1e9;
  else if (suffix == "T")  multiplier = 1e12;
  else {
    fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: unknown frequency suffix '{}' in '{}'\n", suffix, s);
    std::exit(-1);
  }
  auto ps = static_cast<int64_t>(std::round(1e12 / (value * multiplier)));
  return champsim::chrono::picoseconds{ps};
}

// Parse a time string with suffix → std::any of the appropriate chrono type.
// The suffix determines the exact stored type (must match consumer expectations).
std::any parse_time_string(const std::string& s) {
  auto [value, suffix] = parse_number_and_suffix(s);
  auto int_val = static_cast<int64_t>(std::round(value));
  if (suffix.empty() || suffix == "p")
    return champsim::chrono::picoseconds{int_val};
  else if (suffix == "n")
    return champsim::chrono::nanoseconds{int_val};
  else if (suffix == "u")
    return champsim::chrono::microseconds{int_val};
  else if (suffix == "m")
    return champsim::chrono::milliseconds{int_val};
  else if (suffix == "s")
    return champsim::chrono::seconds{int_val};
  fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: unknown time suffix '{}' in '{}'\n", suffix, s);
  std::exit(-1);
}

// Parse a bytes string with SI or IEC binary suffix → champsim::data::bytes.
champsim::data::bytes parse_bytes_string(const std::string& s) {
  auto [value, suffix] = parse_number_and_suffix(s);
  auto int_val = static_cast<long long>(std::round(value));
  if (suffix.empty())
    return champsim::data::bytes{int_val};
  // IEC binary prefixes
  if (suffix == "Ki") return champsim::data::kibibytes{int_val};
  if (suffix == "Mi") return champsim::data::mebibytes{int_val};
  if (suffix == "Gi") return champsim::data::gibibytes{int_val};
  if (suffix == "Ti") return champsim::data::tebibytes{int_val};
  // SI decimal prefixes
  if (suffix == "K") return champsim::data::bytes{int_val * 1000LL};
  if (suffix == "M") return champsim::data::bytes{int_val * 1000000LL};
  if (suffix == "G") return champsim::data::bytes{int_val * 1000000000LL};
  if (suffix == "T") return champsim::data::bytes{int_val * 1000000000000LL};
  fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: unknown bytes suffix '{}' in '{}'\n", suffix, s);
  std::exit(-1);
}

// Parse a bits string with SI suffix → champsim::data::bits.
champsim::data::bits parse_bits_string(const std::string& s) {
  auto [value, suffix] = parse_number_and_suffix(s);
  auto int_val = static_cast<unsigned long long>(std::round(value));
  if (suffix.empty()) return champsim::data::bits{int_val};
  if (suffix == "K")  return champsim::data::bits{int_val * 1000ULL};
  if (suffix == "M")  return champsim::data::bits{int_val * 1000000ULL};
  if (suffix == "G")  return champsim::data::bits{int_val * 1000000000ULL};
  fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: unknown bits suffix '{}' in '{}'\n", suffix, s);
  std::exit(-1);
}

// Try to parse a JSON object as a type-wrapped value, e.g. {"frequency": "4G"}
// Returns true and sets the std::any result if recognized.
bool try_parse_typed_value(const json& obj, std::any& out) {
  if (!obj.is_object() || obj.size() != 1) return false;
  auto it = obj.begin();
  const std::string& type_key = it.key();
  const json& val = it.value();

  if (type_key == "frequency" && val.is_string()) {
    out = parse_frequency_string(val.get<std::string>());
    return true;
  } else if (type_key == "time" && val.is_string()) {
    out = parse_time_string(val.get<std::string>());
    return true;
  } else if (type_key == "bytes" && val.is_string()) {
    out = parse_bytes_string(val.get<std::string>());
    return true;
  } else if (type_key == "bits" && val.is_string()) {
    out = parse_bits_string(val.get<std::string>());
    return true;
  } else if (type_key == "bandwidth") {
    out = champsim::bandwidth::maximum_type{val.get<long long>()};
    return true;
  } else if (type_key == "optional_uint64") {
    if (val.is_null()) out = std::optional<uint64_t>{};
    else out = std::optional<uint64_t>{val.get<uint64_t>()};
    return true;
  } else if (type_key == "null") {
    std::string iface_name = val.get<std::string>();
    out = interface_registry::make_null_pointer(iface_name);
    return true;
  } else if (type_key == "access_types" && val.is_array()) {
    std::vector<access_type> result;
    for (auto& elem : val) {
      auto at = access_type_from_string(elem.get<std::string>());
      if (at != access_type::NUM_TYPES)
        result.push_back(at);
    }
    out = result;
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

    // Process all JSON parameters (skip reserved keys)
    for (auto& [key, val] : child.items()) {
      if (key == "name" || key == "module" || key == "model" || key == "children" || key == "_comment") continue;

      if (val.is_null()) {
        // Null value: skip — modules should use optional defaults for nullable params
        continue;
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
        // Check for type-wrapped value, e.g. {"frequency": "4G"}, {"time": "250ps"}, {"null": "channel"}
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
        mod_builder.add_parameter(key, val.get<std::string>());
      } else if (val.is_array()) {
        // Non-ref array: check for string arrays, numeric arrays-of-arrays, etc.
        if (!val.empty() && val[0].is_string()) {
            std::vector<std::string> sv;
            for (auto& e : val) sv.push_back(e.get<std::string>());
            mod_builder.add_parameter(key, sv);
        } else if (!val.empty() && val[0].is_array()) {
          // Array of arrays → std::array<std::array<uint32_t, 3>, 16>
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

    // Handle nested children generically: group by interface type
    if (child.contains("children")) {
      std::map<std::string, std::vector<std::string>> child_models_by_iface;
      std::map<std::string, ModuleBuilder::module_builder_map_type> child_params_by_iface;

      for (auto& sub : child["children"]) {
        std::string sub_iface = sub["module"].get<std::string>();
        std::string sub_name = sub["name"].get<std::string>();
        std::string sub_model = sub["model"].get<std::string>();

        // Extract extra parameters (beyond name/module/model)
        ModuleBuilder child_builder{sub_name, sub_model, nullptr};
        for (auto& [sk, sv] : sub.items()) {
          if (sk == "name" || sk == "module" || sk == "model") continue;
          if (sv.is_boolean()) child_builder.add_parameter(sk, sv.get<bool>());
          else if (sv.is_number_integer()) child_builder.add_parameter(sk, sv.get<int64_t>());
          else if (sv.is_number_float()) child_builder.add_parameter(sk, sv.get<double>());
          else if (sv.is_string()) child_builder.add_parameter(sk, sv.get<std::string>());
        }

        child_models_by_iface[sub_iface].push_back(sub_model);
        child_params_by_iface[sub_iface][sub_model] = std::move(child_builder);
      }

      for (auto& [child_iface, models] : child_models_by_iface) {
        mod_builder.add_parameter(child_iface + "_modules", models);
        if (child_params_by_iface.count(child_iface)) {
          mod_builder.add_parameter(child_iface + "_params", child_params_by_iface[child_iface]);
        }
      }
    }

    // Create the module via the interface registry
    std::any typed_ptr = interface_registry::create(iface, mod_builder);
    modules_by_name_[name] = typed_ptr;
    module_interfaces_[name] = iface;
    builder_params_[name] = mod_builder;

    // Store in the type-indexed collection
    modules_by_type_[iface].push_back(typed_ptr);
    module_order_.emplace_back(name, iface);
  }

  // Count cores for num_cpus
  auto it = modules_by_type_.find("core");
  if (it != modules_by_type_.end()) {
    num_cpus_ = it->second.size();
  }
}

// ====== Generic view function ======

auto champsim::environment::view(const std::string& interface_type) const -> std::vector<std::any>
{
  if (interface_type == "operable") {
    // Aggregate all operable modules in declaration order
    std::vector<std::any> result;
    for (auto& [name, iface] : module_order_) {
      auto to_op = interface_registry::get_to_operable(iface);
      if (to_op) {
        auto& typed_ptr = modules_by_name_.at(name);
        result.push_back(static_cast<champsim::operable*>(to_op(typed_ptr)));
      }
    }
    return result;
  }

  auto it = modules_by_type_.find(interface_type);
  if (it == modules_by_type_.end()) return {};
  return it->second;
}
