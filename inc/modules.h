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

#ifndef MODULES_H
#define MODULES_H

#include <map>
#include <deque>
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <cassert>
#include <any>

#include "access_type.h"
#include "address.h"
#include "block.h"
#include "champsim.h"
#include "instruction.h"
#include "operable.h"
#include "packet.h"
#include "cache_stats.h"
#include "core_stats.h"
#include "dram_stats.h"
#include "bandwidth.h"
#include <any>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include "util/type_traits.h"

//class CACHE;
//class O3_CPU;
namespace champsim::modules {

struct environment_module;

struct ModuleBuilder {
  private:
  std::map<std::string,std::any> parameters;
  std::string module_name = "";
  std::string model = "";
  std::any parent = nullptr;
  static bool global_dump_enabled_;
  static std::string dump_log_;

  static std::string parent_dump_string(const std::any& parent) {
    if (!parent.has_value()) return "<unset>";
    if (parent.type() == typeid(std::nullptr_t)) return "nullptr";
    return parent.type().name();
  }

  static std::string builder_identity_string(const ModuleBuilder& builder) {
    return fmt::format("ModuleBuilder{{name={}, model={}, parent={}}}", builder.module_name, builder.model, parent_dump_string(builder.parent));
  }

  static std::string builder_identity_string(const ModuleBuilder& builder, const std::string& parent_builder_name) {
    auto parent_str = parent_dump_string(builder.parent);
    if (parent_str == "nullptr" && !parent_builder_name.empty()) {
      parent_str = parent_builder_name;
    }
    return fmt::format("ModuleBuilder{{name={}, model={}, parent={}}}", builder.module_name, builder.model, parent_str);
  }

  public:
  static void set_dump_enabled(bool enabled) { global_dump_enabled_ = enabled; }
  static bool is_dump_enabled() { return global_dump_enabled_; }

  bool get_dump() const { return is_dump_enabled(); }
  static const std::string& get_dump_log() { return dump_log_; }
  static void clear_dump_log() { dump_log_.clear(); }

  template<typename T>
  std::string dump_line(const std::string& mod, const std::string& name, const T& val, const char* tag) {
    using value_type = std::decay_t<T>;
    if constexpr (std::is_same_v<value_type, std::map<std::string, ModuleBuilder>>) {
      std::string summaries;
      bool first = true;
      for (auto& [model_name, nested_builder] : val) {
        if (!first) summaries += ", ";
        first = false;
        summaries += fmt::format("{}: {}", model_name, builder_identity_string(nested_builder, mod));
      }
      if (summaries.empty()) summaries = "<empty>";
      return fmt::format("  [{}] {} = {} ({})\n", mod, name, summaries, tag);
    }
    if constexpr (std::is_same_v<value_type, ModuleBuilder>) {
      return fmt::format("  [{}] {} = {} ({})\n", mod, name, builder_identity_string(val, mod), tag);
    }
    if constexpr (fmt::is_formattable<T>::value) {
      try { return fmt::format("  [{}] {} = {} ({})\n", mod, name, val, tag); }
      catch (...) {}
    } else if constexpr (std::is_pointer_v<T>) {
      try { return fmt::format("  [{}] {} = {} ({})\n", mod, name, std::vector<T>{val}, tag); }
      catch (...) {}
    }
    return fmt::format("  [{}] {} = <unprintable> ({})\n", mod, name, tag);
  }

  template<typename T>
  T get_parameter(std::string name, bool optional = false, T default_value = T{}) {
    if(auto it = parameters.find(name); it != parameters.end()) {
      try {
        auto val = std::any_cast<T>(it->second);
        if (is_dump_enabled()) {
          auto line = dump_line(module_name, name, val, "set");
          dump_log_ += line;
          fmt::print("{}", line);
        }
        return val;
      }
      catch(const std::bad_any_cast&) {
        // For arithmetic types, try converting from whatever numeric type was stored
        T result{};
        if (champsim::numeric_any_cast(it->second, result)) {
          if (is_dump_enabled()) {
            auto line = dump_line(module_name, name, result, "set");
            dump_log_ += line;
            fmt::print("{}", line);
          }
          return result;
        }
        fmt::print("[MODULE] [{}] ERROR: Casting failed while retrieving parameter {}, is your parameter type correct?\n",module_name,name);
        exit(-1);
      }
    } else {
      if(optional) {
        if (is_dump_enabled()) {
          auto line = dump_line(module_name, name, default_value, "default");
          dump_log_ += line;
          fmt::print("{}", line);
        }
        return default_value;
      }
      fmt::print("[MODULE] [{}] ERROR: parameter {} not found\n",module_name,name);
      exit(-1);
    }
  }

  template<typename T>
  ModuleBuilder& add_parameter(std::string name, T value) {
    //if(parameters.find(name) != parameters.end()) {
    //  fmt::print("[MODULE] ERROR: duplicate parameter name used: {}\n",name);
    //  exit(-1);
    //} // should we allow this? it would allow for parameters to be set by defaults and then overridden, but it would also allow for accidental overwriting of parameters which could be bad
    parameters[name] = value;
    return *this;
  }

  std::string get_model() const { return model; }
  std::string get_name() const { return module_name; }

  template<typename T>
  T* get_parent() const { return std::any_cast<T*>(parent); }

  // Type for storing per-model builders (model_name -> builder)
  using module_builder_map_type = std::map<std::string, ModuleBuilder>;

  const std::map<std::string, std::any>& get_parameters() const { return parameters; }

  bool is_valid() const {return model != "" && module_name != "" && parent.has_value() && parent.type() != typeid(std::nullptr_t);}

  // Populate this builder with parameters from a module_builder_map_type map, if the model has an entry
  ModuleBuilder& apply_nested_params(const module_builder_map_type& params_map) {
    if (auto it = params_map.find(model); it != params_map.end()) {
      if (!it->second.module_name.empty()) {
        module_name = it->second.module_name;
      }
      for (auto& [k, v] : it->second.parameters) {
        parameters[k] = v;
      }
    }
    return *this;
  }

  // Store a pre-built std::any directly (avoids double-wrapping when passing resolved references)
  ModuleBuilder& add_raw_parameter(std::string name, std::any value) {
    parameters[name] = std::move(value);
    return *this;
  }

  ModuleBuilder() {}
  ModuleBuilder(std::string name_, std::string model_, std::any parent_, ModuleBuilder defaults = ModuleBuilder{}) : module_name(name_), model(model_), parent(parent_) {
    if(!defaults.parameters.empty()) {
      for(auto& [param_name, param_value] : defaults.parameters) {
        parameters[param_name] = param_value;
      }
    }
  }
};

// Registry for module interfaces: maps interface name strings to factory functions.
// This allows runtime lookup of which module_base specialization to use for creation.
class interface_registry {
public:
  struct interface_info {
    std::function<std::any(ModuleBuilder)> create;
    std::function<std::any(const std::vector<std::any>&)> make_vector;
    // Returns operable* from a typed any, or nullptr if the interface is not operable
    std::function<champsim::operable*(const std::any&)> to_operable;
    // Creates a typed null pointer wrapped in std::any
    std::function<std::any()> make_null_pointer;
  };

private:
  static std::map<std::string, interface_info>& registry() {
    static std::map<std::string, interface_info> r;
    return r;
  }

public:
  static void register_interface(const std::string& name, interface_info info) {
    if (registry().count(name)) {
      fmt::print("[MODULE] ERROR: duplicate interface name: {}\n", name);
      exit(-1);
    }
    registry()[name] = std::move(info);
  }

  static std::any create(const std::string& interface_name, ModuleBuilder builder) {
    auto it = registry().find(interface_name);
    if (it == registry().end()) {
      fmt::print("[MODULE] ERROR: unknown interface: {}\n", interface_name);
      exit(-1);
    }
    return it->second.create(std::move(builder));
  }

  static std::any make_vector(const std::string& interface_name, const std::vector<std::any>& elements) {
    auto it = registry().find(interface_name);
    if (it == registry().end()) {
      fmt::print("[MODULE] ERROR: unknown interface for vector: {}\n", interface_name);
      exit(-1);
    }
    return it->second.make_vector(elements);
  }

  static bool has_interface(const std::string& name) {
    return registry().count(name) > 0;
  }

  // Get the to_operable converter for an interface, or nullptr if not operable
  static std::function<champsim::operable*(const std::any&)> get_to_operable(const std::string& interface_name) {
    auto it = registry().find(interface_name);
    if (it == registry().end()) return nullptr;
    return it->second.to_operable;
  }

  // Create a typed null pointer for the given interface
  static std::any make_null_pointer(const std::string& interface_name) {
    auto it = registry().find(interface_name);
    if (it == registry().end()) {
      fmt::print("[MODULE] ERROR: unknown interface for null pointer: {}\n", interface_name);
      exit(-1);
    }
    return it->second.make_null_pointer();
  }
};

//Module base, defining the base type B for the module and component type C that it is used by
template<typename B, typename C>
struct module_base {
    std::string NAME;
    C* intern_;
    using function_type = typename std::function<std::unique_ptr<B>(ModuleBuilder builder)>;

    private:
    static std::map<std::string,std::any>& module_map() {
        static std::map<std::string,std::any> map;
        return map;
    }
    static std::map<std::string,std::vector<std::unique_ptr<B>>>& instance_map() {
        static std::map<std::string,std::vector<std::unique_ptr<B>>> map;
        return map;
    }

    static void add_module(std::string name, std::function<std::unique_ptr<B>(ModuleBuilder builder)> module_constructor) {
        if(module_map().find(name) != module_map().end()) {
            fmt::print("[MODULE] ERROR: duplicate module name used: {}\n", name);
            exit(-1);
        }
        module_map()[name] = module_constructor;
    }

    public:
    //bind the internal pointer to its managing component
    //should probably remove this call
    void bind(C* bind_arg) {intern_ = bind_arg;};

    //create an instance of the module, which will be stored in this base-module-type's static list
    static B* create_instance(ModuleBuilder builder) {
        if(!builder.is_valid()) {
            fmt::print("[MODULE] ERROR: invalid module builder used for module {}\n",builder.get_name());
            exit(-1);
        }
        if(module_map().find(builder.get_model()) == module_map().end()) {
            fmt::print("[MODULE] [{}] ERROR: specified model {} is not registered\n",builder.get_name(),builder.get_model());
            exit(-1);
        }
        try {
          B* instance_ptr = instance_map()[builder.get_name()].emplace_back(std::any_cast<std::function<std::unique_ptr<B>(ModuleBuilder builder)>>(module_map()[builder.get_model()])(builder)).get();
          //It seems sketchy for the module wrapper to be tracking these separately from the module itself, can we fix this?
          instance_ptr->NAME =  builder.get_name();
          instance_ptr->bind(builder.get_parent<C>());
          return(instance_ptr);
        }
        catch(const std::bad_any_cast& caught) {
          fmt::print("[MODULE] ERROR: Casting failed while constructing {}, are your registration and instance calls consistent?\n",builder.get_name());
          exit(-1);
        }
    }

    template<typename T>
    static T* get_instance(std::string name) {
        if(instance_map().find(name) == instance_map().end() || instance_map()[name].empty()) {
            fmt::print("[MODULE] ERROR: no instances found for module {}\n",name);
            exit(-1);
        }
        try {
          return std::any_cast<T*>(instance_map()[name].front().get());
        }
        catch(const std::bad_any_cast& caught) {
          fmt::print("[MODULE] ERROR: Casting failed while retrieving {}, is your instance type correct?\n",name);
          exit(-1);
        }
    }

    //register a derived type D of base type B and constructor with arguments Params with the module system
    //this is necessary to be able to create instances
    template<typename D> 
    struct register_module {
      register_module(std::string model_name) {
          
          std::function<std::unique_ptr<B>(ModuleBuilder builder)> create_module([](ModuleBuilder builder){return std::unique_ptr<B>(new D(builder));});
          add_module(model_name,create_module);
      }
    };

    // Register this module_base specialization as a named interface in the interface_registry.
    // This allows the explicit environment to create modules by interface name string.
    struct register_interface {
      register_interface(std::string interface_name) {
        interface_registry::interface_info info;
        info.create = [](ModuleBuilder builder) -> std::any {
          return create_instance(std::move(builder));
        };
        info.make_vector = [](const std::vector<std::any>& elements) -> std::any {
          std::vector<B*> vec;
          for (auto& e : elements) {
            vec.push_back(std::any_cast<B*>(e));
          }
          return vec;
        };
        if constexpr (std::is_base_of_v<champsim::operable, B>) {
          info.to_operable = [](const std::any& a) -> champsim::operable* {
            return static_cast<champsim::operable*>(std::any_cast<B*>(a));
          };
        }
        info.make_null_pointer = []() -> std::any {
          return static_cast<B*>(nullptr);
        };
        interface_registry::register_interface(interface_name, std::move(info));
      }
    };

};

  struct core_module: public module_base<core_module,environment_module>, public operable {
    //interface for core module
    virtual void push_instruction(ooo_model_instr instr) = 0;
    virtual std::size_t instructions_requested() = 0;
    virtual uint64_t sim_instr() const = 0;
    virtual uint8_t get_cpu_num() const = 0;
    virtual uint64_t sim_cycle() const = 0;

    core_module(champsim::chrono::picoseconds clock_period_) : operable(clock_period_) {}
    virtual ~core_module() = default;

    using stats_type = cpu_stats;
    virtual stats_type get_sim_stats() const = 0;
    virtual stats_type get_roi_stats() const = 0;

    virtual void quiet(bool enable) = 0;
  };

  struct cache_module: public module_base<cache_module,environment_module>, public operable {
    //interface for cache module
    cache_module(champsim::chrono::picoseconds clock_period_) : operable(clock_period_) {}
    virtual ~cache_module() = default;

    using stats_type = cache_stats;
    virtual champsim::bandwidth::maximum_type get_max_tag_bandwidth() const = 0;
    virtual stats_type get_sim_stats() const = 0;
    virtual stats_type get_roi_stats() const = 0;

    virtual bool is_virtual_prefetch() const = 0;
    virtual bool prefetch_line(champsim::address pf_addr, bool fill_this_level, uint32_t prefetch_metadata) = 0;
    virtual void impl_update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                       champsim::address victim_addr, access_type type, bool hit) const = 0;
    virtual void impl_prefetcher_branch_operate(champsim::address ip, uint8_t branch_type, champsim::address target) const = 0;

    virtual long invalidate_entry(champsim::address inval_addr) = 0;
    virtual std::size_t get_mshr_occupancy() const = 0;
    virtual std::size_t get_mshr_size() const = 0;
    virtual double get_mshr_occupancy_ratio() const = 0;

    virtual std::vector<std::size_t> get_rq_occupancy() const = 0;
    virtual std::vector<std::size_t> get_rq_size() const = 0;
    virtual std::vector<double> get_rq_occupancy_ratio() const = 0;

    virtual std::vector<std::size_t> get_wq_occupancy() const = 0;
    virtual std::vector<std::size_t> get_wq_size() const = 0;
    virtual std::vector<double> get_wq_occupancy_ratio() const = 0;

    virtual std::vector<std::size_t> get_pq_occupancy() const = 0;
    virtual std::vector<std::size_t> get_pq_size() const = 0;
    virtual std::vector<double> get_pq_occupancy_ratio() const = 0;

    virtual std::size_t num_sets() const = 0;
    virtual std::size_t num_ways() const = 0;
  };

  struct memory_controller_module: public module_base<memory_controller_module,environment_module>, public operable {
    //interface for memory controller module
    memory_controller_module(champsim::chrono::picoseconds clock_period_) : operable(clock_period_) {}
    virtual ~memory_controller_module() = default;

    using stats_type = dram_stats;
    virtual std::size_t get_num_channels() const = 0;
    virtual stats_type get_sim_stats(std::size_t channel_no) const = 0;
    virtual stats_type get_roi_stats(std::size_t channel_no) const = 0;

    virtual champsim::data::bytes size() const = 0;
  }; 

  struct page_table_walker_module: public module_base<page_table_walker_module,environment_module>, public operable {
    //interface for page table walker module
    page_table_walker_module(champsim::chrono::picoseconds clock_period_) : operable(clock_period_) {}
    virtual ~page_table_walker_module() = default;
  }; 

  struct channel_module: public module_base<channel_module,environment_module> {
    //interface for channel module
    using request_type = champsim::request;
    using response_type = champsim::response;
    using stats_type = champsim::cache_queue_stats;

    virtual bool add_rq(const request_type& packet) = 0;
    virtual bool add_wq(const request_type& packet) = 0;
    virtual bool add_pq(const request_type& packet) = 0;

    virtual std::size_t rq_occupancy() const = 0;
    virtual std::size_t wq_occupancy() const = 0;
    virtual std::size_t pq_occupancy() const = 0;

    virtual std::size_t rq_size() const = 0;
    virtual std::size_t wq_size() const = 0;
    virtual std::size_t pq_size() const = 0;

    // Queue accessors for upper-level iteration
    virtual std::deque<request_type>& get_rq() = 0;
    virtual std::deque<request_type>& get_wq() = 0;
    virtual std::deque<request_type>& get_pq() = 0;
    virtual std::deque<response_type>& get_returned() = 0;

    // Stats accessors
    virtual stats_type& get_sim_stats() = 0;
    virtual stats_type& get_roi_stats() = 0;

    virtual ~channel_module() = default;
  }; 

  struct vmem_module: public module_base<vmem_module,environment_module> {
    virtual ~vmem_module() = default;
    virtual std::size_t available_ppages() const = 0;
    virtual std::pair<champsim::page_number, champsim::chrono::clock::duration> va_to_pa(uint32_t cpu_num, champsim::page_number vaddr) = 0;
    virtual std::pair<champsim::address, champsim::chrono::clock::duration> get_pte_pa(uint32_t cpu_num, champsim::page_number vaddr, std::size_t level) = 0;
    virtual champsim::data::bits shamt(std::size_t level) const = 0;
    virtual uint64_t get_offset(champsim::address vaddr, std::size_t level) const = 0;
    virtual std::size_t get_pt_levels() const = 0;
  };

  struct prefetcher: public module_base<prefetcher,cache_module> {

      virtual ~prefetcher() = default;

      //prefetcher initialize
      virtual void prefetcher_initialize() {}

      //prefetcher cache operate
      virtual uint32_t prefetcher_cache_operate([[maybe_unused]] champsim::address addr, [[maybe_unused]] champsim::address ip, [[maybe_unused]] bool cache_hit, [[maybe_unused]] bool useful_prefetch,
                                                [[maybe_unused]] access_type type, [[maybe_unused]] uint32_t metadata_in) {
        return prefetcher_cache_operate(addr,ip,(uint8_t)cache_hit,useful_prefetch,type,metadata_in);
      }
      virtual uint32_t prefetcher_cache_operate([[maybe_unused]] champsim::address addr, [[maybe_unused]] champsim::address ip, [[maybe_unused]] uint8_t cache_hit, [[maybe_unused]] bool useful_prefetch,
                                                [[maybe_unused]] access_type type, [[maybe_unused]] uint32_t metadata_in) {
        return prefetcher_cache_operate(addr,ip,cache_hit,useful_prefetch,champsim::to_underlying<access_type>(type),metadata_in);
      }
      virtual uint32_t prefetcher_cache_operate([[maybe_unused]] champsim::address addr, [[maybe_unused]] champsim::address ip, [[maybe_unused]] bool cache_hit, [[maybe_unused]] bool useful_prefetch,
                                                [[maybe_unused]] std::underlying_type_t<access_type> type, [[maybe_unused]] uint32_t metadata_in) {
        return prefetcher_cache_operate(addr.to<uint64_t>(),ip.to<uint64_t>(),cache_hit,type,metadata_in);
      }
      virtual uint32_t prefetcher_cache_operate([[maybe_unused]] uint64_t addr, [[maybe_unused]] uint64_t ip, [[maybe_unused]] bool cache_hit,[[maybe_unused]] std::underlying_type_t<access_type> type, 
                                                [[maybe_unused]] uint32_t metadata_in) {return metadata_in;}

      //prefetcher cache fill
      virtual uint32_t prefetcher_cache_fill([[maybe_unused]] champsim::address addr, [[maybe_unused]] long set, [[maybe_unused]] long way, [[maybe_unused]] bool prefetch, 
                                                [[maybe_unused]] champsim::address evicted_addr, [[maybe_unused]] uint32_t metadata_in) {
        return prefetcher_cache_fill(addr,set,way,(uint8_t)prefetch,evicted_addr,metadata_in);
      }
      virtual uint32_t prefetcher_cache_fill([[maybe_unused]] champsim::address addr, [[maybe_unused]] long set, [[maybe_unused]] long way, [[maybe_unused]] uint8_t prefetch,
                                             [[maybe_unused]] champsim::address evicted_addr, [[maybe_unused]] uint32_t metadata_in) {
        return prefetcher_cache_fill(addr.to<uint64_t>(), set, way, prefetch, evicted_addr.to<uint64_t>(), metadata_in);
      }
      virtual uint32_t prefetcher_cache_fill([[maybe_unused]] uint64_t addr, [[maybe_unused]] long set, [[maybe_unused]] long way, [[maybe_unused]] bool prefetch, 
                                                [[maybe_unused]] uint64_t evicted_addr, [[maybe_unused]] uint32_t metadata_in) {return metadata_in;}

      //prefetcher cycle operate
      virtual void prefetcher_cycle_operate() {}

      //prefetcher final stats
      virtual void prefetcher_final_stats() {}

      //prefetcher branch operate
      virtual void prefetcher_branch_operate([[maybe_unused]] champsim::address ip, [[maybe_unused]] uint8_t branch_type, [[maybe_unused]] champsim::address branch_target) {
        prefetcher_branch_operate(ip.to<uint64_t>(), branch_type, branch_target.to<uint64_t>());
      }
      virtual void prefetcher_branch_operate([[maybe_unused]] uint64_t ip, [[maybe_unused]] uint8_t branch_type, [[maybe_unused]] uint64_t branch_target) {}

      bool prefetch_line(champsim::address pf_addr, bool fill_this_level, uint32_t prefetch_metadata) const;
      bool prefetch_line(uint64_t pf_addr, bool fill_this_level, uint32_t prefetch_metadata) const;
  };


  struct replacement: public module_base<replacement,cache_module> {

      virtual ~replacement() = default;

      //initialize replacement
      virtual void initialize_replacement() {}

      //find victim
      virtual long find_victim([[maybe_unused]] uint32_t triggering_cpu, [[maybe_unused]] uint64_t instr_id, [[maybe_unused]] long set, [[maybe_unused]] const champsim::cache_block* current_set, [[maybe_unused]] champsim::address ip,
                                      [[maybe_unused]] champsim::address full_addr, [[maybe_unused]] access_type type) {
        return find_victim(triggering_cpu,instr_id,set,current_set,ip,full_addr,champsim::to_underlying<access_type>(type));
      }
      virtual long find_victim([[maybe_unused]] uint32_t triggering_cpu, [[maybe_unused]] uint64_t instr_id, [[maybe_unused]] long set, [[maybe_unused]] const champsim::cache_block* current_set, [[maybe_unused]] champsim::address ip,
                                      [[maybe_unused]] champsim::address full_addr, [[maybe_unused]] std::underlying_type_t<access_type> type) {
        return find_victim(triggering_cpu, instr_id, set, current_set, ip.to<uint64_t>(), full_addr.to<uint64_t>(), static_cast<access_type>(type));
      }
      virtual long find_victim([[maybe_unused]] uint32_t triggering_cpu, [[maybe_unused]] uint64_t instr_id, [[maybe_unused]] long set, [[maybe_unused]] const champsim::cache_block* current_set, [[maybe_unused]] uint64_t ip,
                                      [[maybe_unused]] uint64_t full_addr, [[maybe_unused]] access_type type) {
        return find_victim(triggering_cpu, instr_id, set, current_set, ip, full_addr, champsim::to_underlying<access_type>(type));
      }
      virtual long find_victim([[maybe_unused]] uint32_t triggering_cpu, [[maybe_unused]] uint64_t instr_id, [[maybe_unused]] long set, [[maybe_unused]] const champsim::cache_block* current_set, [[maybe_unused]] uint64_t ip,
                                      [[maybe_unused]] uint64_t full_addr, [[maybe_unused]] std::underlying_type_t<access_type> type) { return -1;};

      //update replacement state
      virtual void update_replacement_state([[maybe_unused]] uint32_t triggering_cpu, [[maybe_unused]] long set, [[maybe_unused]] long way, [[maybe_unused]] champsim::address full_addr,
                                                  [[maybe_unused]] champsim::address ip, [[maybe_unused]] champsim::address victim_addr, [[maybe_unused]] access_type type, [[maybe_unused]] bool hit) {
        champsim::address repl_victim = hit ? champsim::address{} : victim_addr;
        update_replacement_state(triggering_cpu,set,way,full_addr,ip,repl_victim,type,(uint8_t)hit);
      }
      virtual void update_replacement_state([[maybe_unused]] uint32_t triggering_cpu, [[maybe_unused]] long set, [[maybe_unused]] long way, [[maybe_unused]] champsim::address full_addr,
                                                  [[maybe_unused]] champsim::address ip, [[maybe_unused]] champsim::address victim_addr, [[maybe_unused]] access_type type, [[maybe_unused]] uint8_t hit) {
        update_replacement_state(triggering_cpu,set,way,full_addr,ip,victim_addr,champsim::to_underlying<access_type>(type),hit);
      }
      virtual void update_replacement_state([[maybe_unused]] uint32_t triggering_cpu, [[maybe_unused]] long set, [[maybe_unused]] long way, [[maybe_unused]] champsim::address full_addr,
                                                  [[maybe_unused]] champsim::address ip, [[maybe_unused]] champsim::address victim_addr, [[maybe_unused]] std::underlying_type_t<access_type> type, [[maybe_unused]] bool hit) {
        update_replacement_state(triggering_cpu,set,way,full_addr.to<uint64_t>(),ip.to<uint64_t>(),victim_addr.to<uint64_t>(),type,hit);
      }
      virtual void update_replacement_state([[maybe_unused]] uint32_t triggering_cpu, [[maybe_unused]] long set, [[maybe_unused]] long way, [[maybe_unused]] uint64_t full_addr,
                                                  [[maybe_unused]] uint64_t ip, [[maybe_unused]] uint64_t victim_addr, [[maybe_unused]] std::underlying_type_t<access_type> type, [[maybe_unused]] bool hit) {
        update_replacement_state(triggering_cpu,set,way,champsim::address{full_addr},champsim::address{ip},static_cast<access_type>(type),hit);
      }
      virtual void update_replacement_state([[maybe_unused]] uint32_t triggering_cpu, [[maybe_unused]] long set, [[maybe_unused]] long way, [[maybe_unused]] champsim::address full_addr,
                                                  [[maybe_unused]] champsim::address ip, [[maybe_unused]] access_type type, [[maybe_unused]] bool hit) {}


      //replacement cache fill
      virtual void replacement_cache_fill([[maybe_unused]] uint32_t triggering_cpu, [[maybe_unused]] long set, [[maybe_unused]] long way, [[maybe_unused]] champsim::address full_addr, 
                                                  [[maybe_unused]] champsim::address ip, [[maybe_unused]] champsim::address victim_addr, [[maybe_unused]] access_type type);

      //replacement final stats
      virtual void replacement_final_stats() {}

  };

  struct branch_predictor: public module_base<branch_predictor,core_module> {

    virtual ~branch_predictor() = default;

    //initialize branch predictor
    virtual void initialize_branch_predictor() {}

    //last branch result
    virtual void last_branch_result([[maybe_unused]] champsim::address ip, [[maybe_unused]] champsim::address target, [[maybe_unused]] bool taken, [[maybe_unused]] uint8_t branch_type) {
      last_branch_result(ip.to<uint64_t>(),target.to<uint64_t>(),taken,branch_type);
    }
    virtual void last_branch_result([[maybe_unused]] uint64_t ip, [[maybe_unused]] uint64_t target, [[maybe_unused]] bool taken, [[maybe_unused]] uint8_t branch_type) {}

    //predict branch
    virtual bool predict_branch([[maybe_unused]] champsim::address ip, [[maybe_unused]] champsim::address predicted_target, [[maybe_unused]] bool always_taken, [[maybe_unused]] uint8_t branch_type) {
      return predict_branch(ip.to<uint64_t>(),predicted_target.to<uint64_t>(),always_taken,branch_type);
    }
    virtual bool predict_branch([[maybe_unused]] uint64_t ip, [[maybe_unused]] uint64_t predicted_target, [[maybe_unused]] bool always_taken, [[maybe_unused]] uint8_t branch_type) {
      return predict_branch(champsim::address{ip});
    }
    virtual bool predict_branch([[maybe_unused]] champsim::address ip) {
      return predict_branch(ip.to<uint64_t>());
    }
    virtual bool predict_branch([[maybe_unused]] uint64_t ip) {return false;}

  };

  struct btb: public module_base<btb,core_module> {

    virtual ~btb() = default;

    //initialize btb
    virtual void initialize_btb() {}

    //update btb
    virtual void update_btb([[maybe_unused]] champsim::address ip, [[maybe_unused]] champsim::address predicted_target, [[maybe_unused]] bool taken, [[maybe_unused]] uint8_t branch_type) {
      update_btb(ip.to<uint64_t>(),predicted_target.to<uint64_t>(),taken,branch_type);
    }
    virtual void update_btb([[maybe_unused]] uint64_t ip, [[maybe_unused]] uint64_t predicted_target, [[maybe_unused]] bool taken, [[maybe_unused]] uint8_t branch_type) {}

    //btb prediction
    virtual std::pair<champsim::address, bool> btb_prediction([[maybe_unused]] champsim::address ip, [[maybe_unused]] uint8_t branch_type) {
      return std::pair<champsim::address, bool>{btb_prediction(ip.to<uint64_t>(),branch_type)};
    }
    virtual std::pair<uint64_t, bool> btb_prediction([[maybe_unused]] uint64_t ip, [[maybe_unused]] uint8_t branch_type) {
      std::pair<champsim::address, bool> result = btb_prediction(champsim::address{ip});
      return std::pair<uint64_t, bool>{result.first.to<uint64_t>(),result.second};
    }
    virtual std::pair<champsim::address, bool> btb_prediction([[maybe_unused]] champsim::address ip) {
      return std::pair<champsim::address, bool>{btb_prediction(ip.to<uint64_t>())};  
    }
    virtual std::pair<uint64_t, bool> btb_prediction([[maybe_unused]] uint64_t ip) {return std::pair<uint64_t, bool>{};}
  };

  // Environment module interface - the top-level module that owns/constructs the entire simulation

  struct environment_module : public module_base<environment_module, environment_module> {
    // Single generic view function: returns all modules of the given interface type.
    // Special interface_type "operable" returns all operable modules across all interfaces.
    virtual std::vector<std::any> view(const std::string& interface_type) const = 0;

    // Typed convenience wrapper: casts the any values to T* and returns reference_wrappers.
    template<typename T>
    std::vector<std::reference_wrapper<T>> typed_view(const std::string& interface_type) const {
      auto raw = view(interface_type);
      std::vector<std::reference_wrapper<T>> result;
      for (auto& a : raw) result.push_back(std::ref(*std::any_cast<T*>(a)));
      return result;
    }

    virtual std::size_t get_num_cpus() const { return 0; }
    virtual unsigned get_block_size() const { return 64; }
    virtual unsigned get_page_size() const { return 4096; }

    // New: allow snooping of ModuleBuilder parameters by module name
    virtual const ModuleBuilder get_builder_params(const std::string& module_name) const = 0;

    virtual ~environment_module() = default;
  };

}

// Formatter for vector of pointers to types with a NAME member
template <typename T>
struct fmt::formatter<std::vector<T*>, std::enable_if_t<std::is_convertible_v<decltype(std::declval<T>().NAME), std::string>, char>>
    : fmt::formatter<std::string> {
  auto format(const std::vector<T*>& vec, fmt::format_context& ctx) const {
    std::string result = "[";
    for (std::size_t i = 0; i < vec.size(); i++) {
      if (i > 0) result += ", ";
      result += vec[i] ? ("@" + vec[i]->NAME) : "(null)";
    }
    result += "]";
    return fmt::formatter<std::string>::format(result, ctx);
  }
};


#endif
