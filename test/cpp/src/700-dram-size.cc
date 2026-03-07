#include <catch.hpp>

#include "defaults.hpp"
#include "dram_controller.h"
#include "mocks.hpp"

SCENARIO("A dram controller reports its size accurately")
{
  GIVEN("A DRAM configuration of 8GB")
  {
    const auto clock_period = champsim::chrono::picoseconds{3200};
    const std::size_t trp_cycles = 18;
    const std::size_t trcd_cycles = 18;
    const std::size_t tcas_cycles = 18;
    const std::size_t tras_cycles = 38;
    const std::size_t DRAM_CHANNELS = 1;
    const std::size_t DRAM_BANKGROUPS = 2;
    const std::size_t DRAM_BANKS = 4;
    const std::size_t DRAM_RANKS = 2;
    const std::size_t DRAM_COLUMNS = 1024;
    const std::size_t DRAM_ROWS = 65536;
    const std::size_t DRAM_CHANNEL_WIDTH = 8;
    const champsim::data::bytes expected_size{1ul << 33ul};

    MEMORY_CONTROLLER uut{champsim::modules::ModuleBuilder{"uut", "DRAM", nullptr, champsim::defaults::default_memory_controller()}
                              .add_parameter("mc_period", clock_period)
                              .add_parameter("t_rp", trp_cycles)
                              .add_parameter("t_rcd", trcd_cycles)
                              .add_parameter("t_cas", tcas_cycles)
                              .add_parameter("t_ras", tras_cycles)
                              .add_parameter("channels", DRAM_CHANNELS)
                              .add_parameter("banks", DRAM_BANKS)
                              .add_parameter("bankgroups", DRAM_BANKGROUPS)
                              .add_parameter("ranks", DRAM_RANKS)
                              .add_parameter("columns", DRAM_COLUMNS)
                              .add_parameter("rows", DRAM_ROWS)
                              .add_parameter("channel_width", champsim::data::bytes{DRAM_CHANNEL_WIDTH})};
    WHEN("The memory controller is queried for size")
    {
      champsim::data::bytes actual_size = uut.size();
      THEN("The memory controller reports the correct size") { REQUIRE(actual_size.count() == expected_size.count()); }
    }
  }
  GIVEN("A DRAM configuration of 16GB")
  {
    const auto clock_period = champsim::chrono::picoseconds{3200};
    const std::size_t trp_cycles = 18;
    const std::size_t trcd_cycles = 18;
    const std::size_t tcas_cycles = 18;
    const std::size_t tras_cycles = 38;
    const std::size_t DRAM_CHANNELS = 2;
    const std::size_t DRAM_BANKS = 4;
    const std::size_t DRAM_BANKGROUPS = 2;
    const std::size_t DRAM_RANKS = 2;
    const std::size_t DRAM_COLUMNS = 1024;
    const std::size_t DRAM_ROWS = 65536;
    const std::size_t DRAM_CHANNEL_WIDTH = 8;
    const champsim::data::gibibytes expected_size{1ul << 34ul};

    MEMORY_CONTROLLER uut{champsim::modules::ModuleBuilder{"uut", "DRAM", nullptr, champsim::defaults::default_memory_controller()}
                              .add_parameter("mc_period", clock_period)
                              .add_parameter("t_rp", trp_cycles)
                              .add_parameter("t_rcd", trcd_cycles)
                              .add_parameter("t_cas", tcas_cycles)
                              .add_parameter("t_ras", tras_cycles)
                              .add_parameter("channels", DRAM_CHANNELS)
                              .add_parameter("banks", DRAM_BANKS)
                              .add_parameter("bankgroups", DRAM_BANKGROUPS)
                              .add_parameter("ranks", DRAM_RANKS)
                              .add_parameter("columns", DRAM_COLUMNS)
                              .add_parameter("rows", DRAM_ROWS)
                              .add_parameter("channel_width", champsim::data::bytes{DRAM_CHANNEL_WIDTH})};
    WHEN("The memory controller is queried for size")
    {
      champsim::data::bytes actual_size = uut.size();
      THEN("The memory controller reports the correct size") { REQUIRE(actual_size.count() == expected_size.count()); }
    }
  }
  GIVEN("A DRAM configuration of 32GB")
  {
    const auto clock_period = champsim::chrono::picoseconds{3200};
    const std::size_t trp_cycles = 18;
    const std::size_t trcd_cycles = 18;
    const std::size_t tcas_cycles = 18;
    const std::size_t tras_cycles = 38;
    const std::size_t DRAM_CHANNELS = 2;
    const std::size_t DRAM_BANKS = 8;
    const std::size_t DRAM_BANKGROUPS = 2;
    const std::size_t DRAM_RANKS = 2;
    const std::size_t DRAM_COLUMNS = 1024;
    const std::size_t DRAM_ROWS = 65536;
    const std::size_t DRAM_CHANNEL_WIDTH = 8;
    const champsim::data::gibibytes expected_size{1ul << 35ul};

    MEMORY_CONTROLLER uut{champsim::modules::ModuleBuilder{"uut", "DRAM", nullptr, champsim::defaults::default_memory_controller()}
                              .add_parameter("mc_period", clock_period)
                              .add_parameter("t_rp", trp_cycles)
                              .add_parameter("t_rcd", trcd_cycles)
                              .add_parameter("t_cas", tcas_cycles)
                              .add_parameter("t_ras", tras_cycles)
                              .add_parameter("channels", DRAM_CHANNELS)
                              .add_parameter("banks", DRAM_BANKS)
                              .add_parameter("bankgroups", DRAM_BANKGROUPS)
                              .add_parameter("ranks", DRAM_RANKS)
                              .add_parameter("columns", DRAM_COLUMNS)
                              .add_parameter("rows", DRAM_ROWS)
                              .add_parameter("channel_width", champsim::data::bytes{DRAM_CHANNEL_WIDTH})};
    WHEN("The memory controller is asked for the size of memory")
    {
      champsim::data::bytes actual_size = uut.size();
      THEN("The memory controller reports the correct size") { REQUIRE(actual_size.count() == expected_size.count()); }
    }
  }
}
