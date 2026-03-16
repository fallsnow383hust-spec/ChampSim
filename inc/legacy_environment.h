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

#ifndef LEGACY_ENVIRONMENT_H
#define LEGACY_ENVIRONMENT_H

#include <string>
#include <vector>
#include <map>
#include <any>

#include "modules.h"

namespace champsim {

class legacy_environment final : public champsim::modules::environment_module {
  std::vector<champsim::modules::channel_module*> channels;
  champsim::modules::memory_controller_module* DRAM = nullptr;
  champsim::modules::vmem_module* vmem = nullptr;
  std::vector<champsim::modules::page_table_walker_module*> ptws;
  std::vector<champsim::modules::cache_module*> caches;
  std::vector<champsim::modules::core_module*> cores;

  size_t num_cpus_ = 0;
  unsigned block_size_ = 64;
  unsigned page_size_ = 4096;

  // New: map from module name to ModuleBuilder used for construction
  std::map<std::string, champsim::modules::ModuleBuilder> builder_params_;

public:
  explicit legacy_environment(champsim::modules::ModuleBuilder builder);

  std::vector<std::reference_wrapper<champsim::modules::core_module>> cpu_view() override;
  std::vector<std::reference_wrapper<champsim::modules::cache_module>> cache_view() override;
  std::vector<std::reference_wrapper<champsim::modules::page_table_walker_module>> ptw_view() override;
  champsim::modules::memory_controller_module& dram_view() override;
  std::vector<std::reference_wrapper<champsim::operable>> operable_view() override;

  size_t get_num_cpus() const override { return num_cpus_; }
  unsigned get_block_size() const override { return block_size_; }
  unsigned get_page_size() const override { return page_size_; }

  // New: expose builder params for test snooping
  const champsim::modules::ModuleBuilder get_builder_params(const std::string& module_name) const override {
    auto it = builder_params_.find(module_name);
    if (it != builder_params_.end()) return it->second;
    return champsim::modules::ModuleBuilder();
  }
};

} // namespace champsim

#endif // LEGACY_ENVIRONMENT_H
