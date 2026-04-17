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
#include <limits>
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

// Try to parse an @-reference string, returning the referenced name if valid.
std::optional<std::string> try_parse_ref(const std::string& s) {
  if (!s.empty() && s[0] == '@') return s.substr(1);
  return std::nullopt;
}

// Check if a JSON array is entirely @-references
bool is_ref_array(const json& arr) {
  if (!arr.is_array() || arr.empty()) return false;
  for (auto& elem : arr) {
    if (!elem.is_string() || !try_parse_ref(elem.get<std::string>())) return false;
  }
  return true;
}

// Populate a ModuleBuilder with parameters from a JSON node, with full type
// support (typed objects, @-references, arrays, scalars) and recursive children.
// This handles arbitrary nesting depth for submodules.
void populate_builder(const json& node, ModuleBuilder& builder,
                      const std::map<std::string, std::any>& modules_by_name,
                      const std::map<std::string, std::string>& module_interfaces)
{
  const std::string& name = builder.get_name();

  // Process all JSON parameters (skip reserved keys)
  for (auto& [key, val] : node.items()) {
    if (key == "name" || key == "module" || key == "model" || key == "children" || key == "_comment") continue;

    if (val.is_null()) {
      continue;
    } else if (auto ref = val.is_string() ? try_parse_ref(val.get<std::string>()) : std::nullopt) {
      const auto& rn = *ref;
      auto mit = modules_by_name.find(rn);
      if (mit == modules_by_name.end()) {
        fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: @-reference '{}' not found (used in '{}' param '{}')\n", rn, name, key);
        std::exit(-1);
      }
      builder.add_raw_parameter(key, mit->second);
    } else if (val.is_array() && is_ref_array(val)) {
      std::vector<std::any> refs;
      std::string ref_iface;
      for (auto& elem : val) {
        auto rn = *try_parse_ref(elem.get<std::string>());
        auto mit = modules_by_name.find(rn);
        if (mit == modules_by_name.end()) {
          fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: @-reference '{}' not found (in array param '{}' of '{}')\n", rn, key, name);
          std::exit(-1);
        }
        std::string curr_iface = module_interfaces.at(rn);
        if (ref_iface.empty()) {
          ref_iface = curr_iface;
        } else if (curr_iface != ref_iface) {
          fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: mixed interface types in array '{}' of '{}': expected '{}', got '{}' for '{}'\n",
                     key, name, ref_iface, curr_iface, rn);
          std::exit(-1);
        }
        refs.push_back(mit->second);
      }
      builder.add_raw_parameter(key, interface_registry::make_vector(ref_iface, refs));
    } else if (val.is_object()) {
      std::any typed_val;
      if (try_parse_typed_value(val, typed_val)) {
        builder.add_raw_parameter(key, std::move(typed_val));
      } else {
        builder.add_parameter(key, val);
      }
    } else if (val.is_boolean()) {
      builder.add_parameter(key, val.get<bool>());
    } else if (val.is_number_integer()) {
      builder.add_parameter(key, val.get<int64_t>());
    } else if (val.is_number_float()) {
      builder.add_parameter(key, val.get<double>());
    } else if (val.is_string()) {
      builder.add_parameter(key, val.get<std::string>());
    } else if (val.is_array()) {
      if (!val.empty() && val[0].is_string()) {
        std::vector<std::string> sv;
        for (auto& e : val) sv.push_back(e.get<std::string>());
        builder.add_parameter(key, sv);
      } else if (!val.empty() && val[0].is_array()) {
        std::vector<std::array<uint32_t, 3>> dims;
        for (std::size_t i = 0; i < val.size(); i++) {
          std::array<uint32_t, 3> entry{};
          for (std::size_t j = 0; j < val[i].size() && j < 3; j++) {
            entry[j] = static_cast<uint32_t>(val[i][j].get<int64_t>());
          }
          dims.push_back(entry);
        }
        builder.add_parameter(key, dims);
      } else {
        builder.add_parameter(key, val);
      }
    }
  }

  // Recursively handle nested children as submodules
  if (node.contains("children")) {
    for (auto& sub : node["children"]) {
      if (!sub.contains("name") || !sub.contains("module") || !sub.contains("model")) {
        fmt::print("[EXPLICIT_ENVIRONMENT] ERROR: submodule of '{}' missing 'name', 'module', or 'model'\n", name);
        std::exit(-1);
      }
      std::string sub_iface = sub["module"].get<std::string>();
      std::string sub_name = sub["name"].get<std::string>();
      std::string sub_model = sub["model"].get<std::string>();

      ModuleBuilder child_builder{sub_name, sub_model};
      populate_builder(sub, child_builder, modules_by_name, module_interfaces);
      builder.add_submodule(sub_iface, std::move(child_builder));
    }
  }
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

    auto mod_builder = ModuleBuilder{name, model};

    // Populate parameters (with full type support) and submodules (recursive)
    populate_builder(child, mod_builder, modules_by_name_, module_interfaces_);

    // Create the module via the interface registry
    std::any typed_ptr = interface_registry::create(iface, mod_builder, static_cast<environment_module*>(this));
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

  // Compute deadlock threshold purely from parameter types.
  // Every champsim::chrono::picoseconds parameter is a time value; the minimum
  // is the time quantum and the sum of all latencies is the worst-case total.
  // sum/min gives worst-case cycles, floored at 500.
  {
    using ps_rep = champsim::chrono::picoseconds::rep;
    ps_rep min_ps = std::numeric_limits<ps_rep>::max();
    ps_rep sum_ps = 0;
    for (auto& [name, bp] : builder_params_) {
      for (auto& [key, val] : bp.get_parameters()) {
        if (auto* p = std::any_cast<champsim::chrono::picoseconds>(&val)) {
          if (p->count() > 0) {
            min_ps = std::min(min_ps, p->count());
            sum_ps += p->count();
          }
        }
      }
    }
    if (min_ps < std::numeric_limits<ps_rep>::max() && min_ps > 0)
      deadlock_cycles_ = static_cast<int>(std::max((sum_ps / min_ps)*3, ps_rep{500}));
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
