#include <string>
#include <catch.hpp>

#include "modules.h"

using champsim::modules::ModuleBuilder;

// A minimal concrete module type for testing create_instance + dump
struct dummy_module : public champsim::modules::prefetcher {
  int val_a = 0;
  int val_b = 0;
  dummy_module(ModuleBuilder builder)
    : val_a(builder.get_parameter<int>("param_a")),
      val_b(builder.get_parameter<int>("param_b", true, 42)) {}

  void prefetcher_initialize() override {}
  uint32_t prefetcher_cache_operate(champsim::address, champsim::address, bool, bool, access_type, uint32_t metadata_in) override { return metadata_in; }
  uint32_t prefetcher_cache_fill(champsim::address, long, long, bool, champsim::address, uint32_t metadata_in) override { return metadata_in; }
  void prefetcher_cycle_operate() override {}
  void prefetcher_final_stats() override {}
  void prefetcher_branch_operate(champsim::address, uint8_t, champsim::address) override {}
};

// Register the dummy module so create_instance can find it
static champsim::modules::prefetcher::register_module<dummy_module> dummy_reg("test_dummy_093");

SCENARIO("ModuleBuilder dump logs parameter accesses when enabled")
{
  ModuleBuilder::clear_dump_log();
  ModuleBuilder::set_dump_enabled(false);

  GIVEN("A builder with dump enabled, one param set and one left to default")
  {
    ModuleBuilder::set_dump_enabled(true);
    auto builder = ModuleBuilder{"test_mod", "test_dummy_093"};
    builder.add_parameter("param_a", 10);

    WHEN("get_parameter is called for a set param and an optional defaulted param")
    {
      auto a = builder.get_parameter<int>("param_a");
      auto b = builder.get_parameter<int>("param_b", true, 42);

      THEN("The dump log contains entries for both parameters")
      {
        auto& log = ModuleBuilder::get_dump_log();
        REQUIRE(log.find("param_a") != std::string::npos);
        REQUIRE(log.find("param_b") != std::string::npos);
        REQUIRE(log.find("[test_mod]") != std::string::npos);
      }

      THEN("param_a shows value 10 (set) and param_b shows value 42 (default)")
      {
        auto& log = ModuleBuilder::get_dump_log();
        auto pos_a = log.find("param_a");
        REQUIRE(log.find("10 (set)", pos_a) != std::string::npos);

        auto pos_b = log.find("param_b");
        REQUIRE(log.find("42 (default)", pos_b) != std::string::npos);
      }

      THEN("The returned values are correct")
      {
        REQUIRE(a == 10);
        REQUIRE(b == 42);
      }
    }
  }

  GIVEN("A builder without dump enabled")
  {
    ModuleBuilder::clear_dump_log();
    ModuleBuilder::set_dump_enabled(false);
    auto builder = ModuleBuilder{"test_mod2", "test_dummy_093"}
      .add_parameter("param_a", 5);

    WHEN("get_parameter is called")
    {
      builder.get_parameter<int>("param_a");

      THEN("The dump log remains empty")
      {
        REQUIRE(ModuleBuilder::get_dump_log().empty());
      }
    }
  }

  ModuleBuilder::clear_dump_log();
  ModuleBuilder::set_dump_enabled(false);
}

SCENARIO("ModuleBuilder dump works through create_instance")
{
  ModuleBuilder::clear_dump_log();
  ModuleBuilder::set_dump_enabled(false);

  GIVEN("A registered dummy module with dump enabled, one explicit and one defaulted param")
  {
    ModuleBuilder::set_dump_enabled(true);
    WHEN("create_instance is called")
    {
      auto builder = ModuleBuilder{"dump_test_mod", "test_dummy_093"};
      builder.add_parameter("param_a", 7);
      auto* instance = champsim::modules::prefetcher::create_instance(builder, static_cast<champsim::modules::cache_module*>(nullptr));

      THEN("The dump log shows param_a as 7 (set) and param_b as 42 (default)")
      {
        auto& log = ModuleBuilder::get_dump_log();
        REQUIRE(log.find("[dump_test_mod]") != std::string::npos);
        REQUIRE(log.find("param_a") != std::string::npos);
        REQUIRE(log.find("param_b") != std::string::npos);

        auto pos_a = log.find("param_a");
        REQUIRE(log.find("7 (set)", pos_a) != std::string::npos);

        auto pos_b = log.find("param_b");
        REQUIRE(log.find("42 (default)", pos_b) != std::string::npos);
      }

      THEN("The constructed module received the correct values")
      {
        auto* d = dynamic_cast<dummy_module*>(instance);
        REQUIRE(d != nullptr);
        CHECK(d->val_a == 7);
        CHECK(d->val_b == 42);
      }
    }
  }

  ModuleBuilder::clear_dump_log();
  ModuleBuilder::set_dump_enabled(false);
}

SCENARIO("ModuleBuilder dump flag propagates through defaults constructor")
{
  ModuleBuilder::clear_dump_log();
  ModuleBuilder::set_dump_enabled(false);

  GIVEN("A defaults builder with dump enabled")
  {
    ModuleBuilder defaults;
    ModuleBuilder::set_dump_enabled(true);

    WHEN("A new builder is constructed with those defaults")
    {
      auto builder = ModuleBuilder{"prop_test", "test_dummy_093", defaults};
      builder.add_parameter("param_a", 99);
      builder.get_parameter<int>("param_a");

      THEN("The dump log has an entry from the new builder")
      {
        auto& log = ModuleBuilder::get_dump_log();
        REQUIRE(log.find("[prop_test]") != std::string::npos);
        REQUIRE(log.find("param_a") != std::string::npos);
      }
    }
  }

  ModuleBuilder::clear_dump_log();
  ModuleBuilder::set_dump_enabled(false);
}
