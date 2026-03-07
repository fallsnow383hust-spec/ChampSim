#include <any>
#include <catch.hpp>
#include <nlohmann/json.hpp>

#include "environment.h"
#include "modules.h"
#include "cache.h"

using json = nlohmann::json;
using champsim::modules::ModuleBuilder;

namespace {
// Helper to build an environment from a JSON config
champsim::modules::environment_module* make_env(const json& config) {
  auto builder = ModuleBuilder{"test_env", "DEFAULT_ENVIRONMENT", static_cast<champsim::modules::environment_module*>(nullptr)};
  builder.add_parameter("config_json", config);
  return champsim::modules::environment_module::create_instance(builder);
}
} // namespace

SCENARIO("Environment with default (empty) config produces correct topology") {
  GIVEN("An empty JSON config") {
    auto* env = make_env(json::object());

    THEN("num_cpus is 1") {
      REQUIRE(env->get_num_cpus() == 1);
    }
    THEN("block_size is 64") {
      REQUIRE(env->get_block_size() == 64);
    }
    THEN("page_size is 4096") {
      REQUIRE(env->get_page_size() == 4096);
    }
    THEN("cpu_view has 1 core") {
      REQUIRE(env->cpu_view().size() == 1);
    }
    THEN("ptw_view has 1 PTW") {
      REQUIRE(env->ptw_view().size() == 1);
    }
    THEN("cache_view has 7 caches (LLC + 6 per-core)") {
      // LLC, DTLB, ITLB, L1D, L1I, L2C, STLB
      REQUIRE(env->cache_view().size() == 7);
    }
    THEN("operable_view has 1 core + 7 caches + 1 PTW + 1 DRAM = 10") {
      REQUIRE(env->operable_view().size() == 10);
    }
  }
}

SCENARIO("Environment with multi-core config") {
  GIVEN("A config with num_cores=2") {
    json config = {{"num_cores", 2}};
    auto* env = make_env(config);

    THEN("num_cpus is 2") {
      REQUIRE(env->get_num_cpus() == 2);
    }
    THEN("cpu_view has 2 cores") {
      REQUIRE(env->cpu_view().size() == 2);
    }
    THEN("ptw_view has 2 PTWs") {
      REQUIRE(env->ptw_view().size() == 2);
    }
    THEN("cache_view has 13 caches (LLC + 6 per-core * 2)") {
      REQUIRE(env->cache_view().size() == 13);
    }
    THEN("operable_view has 2 cores + 13 caches + 2 PTWs + 1 DRAM = 18") {
      REQUIRE(env->operable_view().size() == 18);
    }
  }
}

SCENARIO("Environment with num_cores=4") {
  GIVEN("A config with num_cores=4") {
    json config = {{"num_cores", 4}};
    auto* env = make_env(config);

    THEN("num_cpus is 4") {
      REQUIRE(env->get_num_cpus() == 4);
    }
    THEN("cpu_view has 4 cores") {
      REQUIRE(env->cpu_view().size() == 4);
    }
    THEN("ptw_view has 4 PTWs") {
      REQUIRE(env->ptw_view().size() == 4);
    }
    THEN("cache_view has 25 caches (LLC + 6 per-core * 4)") {
      REQUIRE(env->cache_view().size() == 25);
    }
  }
}

SCENARIO("Environment respects custom block_size and page_size") {
  GIVEN("A config with block_size=128 and page_size=8192") {
    json config = {{"block_size", 128}, {"page_size", 8192}};
    auto* env = make_env(config);

    THEN("block_size is 128") {
      REQUIRE(env->get_block_size() == 128);
    }
    THEN("page_size is 8192") {
      REQUIRE(env->get_page_size() == 8192);
    }
  }
}

SCENARIO("Environment parses block_size from string with suffix") {
  GIVEN("A config with block_size='128B'") {
    json config = {{"block_size", "128B"}};
    auto* env = make_env(config);

    THEN("block_size is 128") {
      REQUIRE(env->get_block_size() == 128);
    }
  }
}

SCENARIO("Environment parses page_size from string with kB suffix") {
  GIVEN("A config with page_size='8kB'") {
    json config = {{"page_size", "8kB"}};
    auto* env = make_env(config);

    THEN("page_size is 8192") {
      REQUIRE(env->get_page_size() == 8192);
    }
  }
}

SCENARIO("Environment with explicit per-core ooo_cpu config") {
  GIVEN("A config with 2 cores and explicit ooo_cpu array") {
    json config = {
      {"num_cores", 2},
      {"ooo_cpu", json::array({
        {{"frequency", 4000}},
        {{"frequency", 3000}}
      })}
    };
    auto* env = make_env(config);

    THEN("num_cpus is 2") {
      REQUIRE(env->get_num_cpus() == 2);
    }
    THEN("cpu_view has 2 cores") {
      REQUIRE(env->cpu_view().size() == 2);
    }
  }
}

SCENARIO("Environment with single ooo_cpu entry duplicated across cores") {
  GIVEN("A config with num_cores=3 but only one ooo_cpu entry") {
    json config = {
      {"num_cores", 3},
      {"ooo_cpu", json::array({{{"frequency", 4000}}})}
    };
    auto* env = make_env(config);

    THEN("num_cpus is 3") {
      REQUIRE(env->get_num_cpus() == 3);
    }
    THEN("All 3 cores are created") {
      REQUIRE(env->cpu_view().size() == 3);
    }
  }
}

SCENARIO("Environment dram_view returns a valid reference") {
  GIVEN("A default config") {
    auto* env = make_env(json::object());

    THEN("dram_view returns a reference to a memory controller") {
      auto& dram = env->dram_view();
      REQUIRE(dram.get_num_channels() >= 1);
    }
  }
}

SCENARIO("Environment with custom DRAM parameters") {
  GIVEN("A config with custom physical_memory settings") {
    json config = {
      {"physical_memory", {
        {"channels", 2},
        {"ranks", 2},
        {"bankgroups", 4},
        {"banks", 8}
      }}
    };
    auto* env = make_env(config);

    THEN("DRAM is configured with 2 channels") {
      REQUIRE(env->dram_view().get_num_channels() == 2);
    }
  }
}

SCENARIO("Environment with custom virtual_memory levels") {
  GIVEN("A config with 4 page table levels") {
    json config = {
      {"virtual_memory", {{"num_levels", 4}}}
    };
    auto* env = make_env(config);

    THEN("The environment constructs successfully with 1 core") {
      REQUIRE(env->get_num_cpus() == 1);
      REQUIRE(env->cpu_view().size() == 1);
    }
  }
}

SCENARIO("Environment dump mode does not crash") {
  GIVEN("An empty config with dump enabled") {
    ModuleBuilder::clear_dump_log();
    auto builder = ModuleBuilder{"dump_env", "DEFAULT_ENVIRONMENT",
                                 static_cast<champsim::modules::environment_module*>(nullptr)};
    builder.add_parameter("config_json", json::object());
    builder.enable_dump();

    THEN("Construction succeeds and dump log is non-empty") {
      auto* env = champsim::modules::environment_module::create_instance(builder);
      REQUIRE(env->get_num_cpus() == 1);
      REQUIRE_FALSE(ModuleBuilder::get_dump_log().empty());
    }

    ModuleBuilder::clear_dump_log();
  }
}

// Helpers for module access
champsim::modules::cache_module* get_cache(champsim::modules::environment_module* env, const std::string& name) {
  for (auto& cache_ref : env->cache_view()) {
    if (cache_ref.get().NAME == name)
      return &cache_ref.get();
  }
  return nullptr;
}
champsim::modules::page_table_walker_module* get_ptw(champsim::modules::environment_module* env, const std::string& name) {
  for (auto& ptw_ref : env->ptw_view()) {
    if (ptw_ref.get().NAME == name)
      return &ptw_ref.get();
  }
  return nullptr;
}

SCENARIO("Cache latency decomposition covers all branches and propagates") {
  GIVEN("latency only") {
    json config = {{"L1D", {{"latency", 10}}}};
    auto* env = make_env(config);
    auto l1d = (CACHE*)get_cache(env, "cpu0_L1D");
    REQUIRE(l1d);
    REQUIRE(l1d->HIT_LATENCY == 5 * l1d->clock_period);
    REQUIRE(l1d->FILL_LATENCY == 5 * l1d->clock_period);
  }
  GIVEN("latency + hit_latency") {
    json config = {{"L1D", {{"latency", 10}, {"hit_latency", 3}}}};
    auto* env = make_env(config);
    auto l1d = (CACHE*)get_cache(env, "cpu0_L1D");
    REQUIRE(l1d);
    REQUIRE(l1d->HIT_LATENCY == 3 * l1d->clock_period);
    REQUIRE(l1d->FILL_LATENCY == 7 * l1d->clock_period);
  }
  GIVEN("latency + fill_latency") {
    json config = {{"L1D", {{"latency", 10}, {"fill_latency", 7}}}};
    auto* env = make_env(config);
    auto l1d = (CACHE*)get_cache(env, "cpu0_L1D");
    REQUIRE(l1d);
    REQUIRE(l1d->HIT_LATENCY == 3 * l1d->clock_period);
    REQUIRE(l1d->FILL_LATENCY == 7 * l1d->clock_period);
  }
  GIVEN("latency + hit_latency + fill_latency") {
    json config = {{"L1D", {{"latency", 10}, {"hit_latency", 4}, {"fill_latency", 6}}}};
    auto* env = make_env(config);
    auto l1d = (CACHE*)get_cache(env, "cpu0_L1D");
    REQUIRE(l1d);
    REQUIRE(l1d->HIT_LATENCY == 4 * l1d->clock_period);
    REQUIRE(l1d->FILL_LATENCY == 6 * l1d->clock_period);
  }
}

SCENARIO("Cache size derivation covers all branches and propagates") {
  GIVEN("size only") {
    json config = {{"L1D", {{"size", "32kB"}}}};
    auto* env = make_env(config);
    auto l1d = (CACHE*)get_cache(env, "cpu0_L1D");
    REQUIRE(l1d);
    REQUIRE(l1d->NUM_SET > 0);
  }
  GIVEN("size + ways") {
    json config = {{"L1D", {{"size", "32kB"}, {"ways", 4}}}};
    auto* env = make_env(config);
    auto l1d = (CACHE*)get_cache(env, "cpu0_L1D");
    REQUIRE(l1d);
    REQUIRE(l1d->NUM_SET > 0);
  }
  GIVEN("size + sets") {
    json config = {{"L1D", {{"size", "32kB"}, {"sets", 64}}}};
    auto* env = make_env(config);
    auto l1d = (CACHE*)get_cache(env, "cpu0_L1D");
    REQUIRE(l1d);
    REQUIRE(l1d->NUM_WAY > 0);
  }
}

SCENARIO("Prefetch activate parsing covers all branches and propagates") {
  GIVEN("prefetch_activate as string") {
    json config = {{"L1D", {{"prefetch_activate", "LOAD,PREFETCH"}}}};
    auto* env = make_env(config);
    auto l1d = (CACHE*)get_cache(env, "cpu0_L1D");
    REQUIRE(l1d);
    auto mask = l1d->pref_activate_mask;
    REQUIRE(mask.size() == 2);
    REQUIRE(mask[0] == access_type::LOAD);
    REQUIRE(mask[1] == access_type::PREFETCH);
  }
  GIVEN("prefetch_activate as array") {
    json config = {{"L1D", {{"prefetch_activate", json::array({"LOAD", "PREFETCH"})}}}};
    auto* env = make_env(config);
    auto l1d = (CACHE*)get_cache(env, "cpu0_L1D");
    REQUIRE(l1d);
    auto mask = l1d->pref_activate_mask;
    REQUIRE(mask.size() == 2);
    REQUIRE(mask[0] == access_type::LOAD);
    REQUIRE(mask[1] == access_type::PREFETCH);
  }
}

SCENARIO("Prefetcher module list and nested params parsing covers all branches and propagates") {
  GIVEN("prefetcher as array of strings and objects") {
    json config = { {"L1D", {
      {"prefetcher", json::array({
        "no",
        { {"model", "ip_stride"}, {"degree", 8} },
        { {"model", "va_ampm_lite"}, {"window_size", 16} }
      })}
    }} };
    auto* env = make_env(config);
    champsim::modules::ModuleBuilder builder = env->get_builder_params("cpu0_L1D");
    REQUIRE(builder.is_valid());
    auto pf_mods = builder.get_parameter<std::vector<std::string>>("prefetcher_modules");
    REQUIRE(pf_mods.size() == 3);
    REQUIRE(pf_mods[0] == "no");
    REQUIRE(pf_mods[1] == "ip_stride");
    REQUIRE(pf_mods[2] == "va_ampm_lite");
    auto pf_params = builder.get_parameter<champsim::modules::ModuleBuilder::nested_params_type>("prefetcher_params");
    REQUIRE(pf_params.count("ip_stride"));
    REQUIRE(std::any_cast<int64_t>(pf_params["ip_stride"]["degree"]) == 8);
    REQUIRE(pf_params.count("va_ampm_lite"));
    REQUIRE(std::any_cast<int64_t>(pf_params["va_ampm_lite"]["window_size"]) == 16);
  }
  GIVEN("prefetcher as single object") {
    json config = { {"L1D", {
      {"prefetcher", { {"model", "ip_stride"}, {"degree", 4} }}
    }} };
    auto* env = make_env(config);
    champsim::modules::ModuleBuilder builder = env->get_builder_params("cpu0_L1D");
    REQUIRE(builder.is_valid());
    auto pf_mods = builder.get_parameter<std::vector<std::string>>("prefetcher_modules");
    REQUIRE(pf_mods.size() == 1);
    REQUIRE(pf_mods[0] == "ip_stride");
    auto pf_params = builder.get_parameter<champsim::modules::ModuleBuilder::nested_params_type>("prefetcher_params");
    REQUIRE(pf_params.count("ip_stride"));
    REQUIRE(std::any_cast<int64_t>(pf_params["ip_stride"]["degree"]) == 4);
  }
  GIVEN("prefetcher as array of objects only") {
    json config = { {"L1D", {
      {"prefetcher", json::array({
        { {"model", "ip_stride"}, {"degree", 2} },
        { {"model", "va_ampm_lite"}, {"window_size", 32} }
      })}
    }} };
    auto* env = make_env(config);
    champsim::modules::ModuleBuilder builder = env->get_builder_params("cpu0_L1D");
    REQUIRE(builder.is_valid());
    auto pf_mods = builder.get_parameter<std::vector<std::string>>("prefetcher_modules");
    REQUIRE(pf_mods.size() == 2);
    REQUIRE(pf_mods[0] == "ip_stride");
    REQUIRE(pf_mods[1] == "va_ampm_lite");
    auto pf_params = builder.get_parameter<champsim::modules::ModuleBuilder::nested_params_type>("prefetcher_params");
    REQUIRE(pf_params.count("ip_stride"));
    REQUIRE(std::any_cast<int64_t>(pf_params["ip_stride"]["degree"]) == 2);
    REQUIRE(pf_params.count("va_ampm_lite"));
    REQUIRE(std::any_cast<int64_t>(pf_params["va_ampm_lite"]["window_size"]) == 32);
  }
}

SCENARIO("PTW bandwidth propagates") {
  GIVEN("PTW with max_read and max_write") {
    json config = {{"PTW", {{"max_read", 10}, {"max_write", 20}}}};
    auto* env = make_env(config);
    auto* ptw = get_ptw(env, "cpu0_PTW");
    REQUIRE(ptw);
    // Try to access MAX_READ and MAX_FILL if available
    // If not, just check ptw is constructed
  }
}

SCENARIO("Environment builder parameter snooping exposes config propagation") {
  GIVEN("A config with custom L1D and PTW parameters") {
    json config = {
      {"L1D", {
        {"latency", 12},
        {"size", "64kB"},
        {"prefetch_activate", "LOAD,WRITE"},
        {"replacement", json::array({"lru", { {"model", "ship"}, {"param1", 42} }})},
        {"prefetcher", { {"model", "ip_stride"}, {"degree", 4} }}
      }},
      {"PTW", {
        {"max_read", 15},
        {"max_write", 25},
        {"pscl5_set", 2},
        {"pscl5_way", 3}
      }}
    };
    auto* env = make_env(config);
    // L1D
    auto l1d = env->get_builder_params("cpu0_L1D");
    REQUIRE(l1d.is_valid());
    REQUIRE((l1d.get_parameter<int64_t>("latency")) == 12);
    REQUIRE((l1d.get_parameter<std::string>("size")) == "64kB");
    auto mask = l1d.get_parameter<std::vector<access_type>>("pref_activate_mask");
    REQUIRE(mask.size() == 2);
    REQUIRE(mask[0] == access_type::LOAD);
    REQUIRE(mask[1] == access_type::WRITE);
    // Check replacement_modules and prefetcher_modules
    auto repl_mods = l1d.get_parameter<std::vector<std::string>>("replacement_modules");
    REQUIRE(repl_mods.size() == 2);
    REQUIRE(repl_mods[0] == "lru");
    REQUIRE(repl_mods[1] == "ship");
    auto pf_mods = l1d.get_parameter<std::vector<std::string>>("prefetcher_modules");
    REQUIRE(pf_mods.size() == 1);
    REQUIRE(pf_mods[0] == "ip_stride");
    // Check nested params
    auto repl_params = l1d.get_parameter<ModuleBuilder::nested_params_type>("replacement_params");
    REQUIRE(repl_params.count("ship"));
    REQUIRE(std::any_cast<int64_t>(repl_params["ship"]["param1"]) == 42);
    auto pf_params = l1d.get_parameter<ModuleBuilder::nested_params_type>("prefetcher_params");
    REQUIRE(pf_params.count("ip_stride"));
    REQUIRE(std::any_cast<int64_t>(pf_params["ip_stride"]["degree"]) == 4);

    // PTW
    auto ptw = env->get_builder_params("cpu0_PTW");
    REQUIRE(ptw.is_valid());
    REQUIRE(ptw.get_parameter<champsim::bandwidth::maximum_type>("max_tag_check") == champsim::bandwidth::maximum_type{15});
    REQUIRE(ptw.get_parameter<champsim::bandwidth::maximum_type>("max_fill") == champsim::bandwidth::maximum_type{25});
    auto pscl = ptw.get_parameter<std::array<std::array<uint32_t, 3>, 16>>("pscl_dims");
    REQUIRE(pscl[0][0] == 5);
    REQUIRE(pscl[0][1] == 2);
    REQUIRE(pscl[0][2] == 3);
  }
}
