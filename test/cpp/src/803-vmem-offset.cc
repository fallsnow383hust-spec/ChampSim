#include <catch.hpp>

#include "dram_controller.h"
#include "vmem.h"
#include "defaults.hpp"

TEST_CASE("The virtual memory evaluates the correct shift amounts")
{
  constexpr unsigned log2_pte_page_size = 12;

  auto level = GENERATE(as<std::size_t>{}, 1, 2, 3, 4, 5);

  MEMORY_CONTROLLER dram{champsim::modules::ModuleBuilder{"dram", "DRAM", nullptr, champsim::defaults::default_memory_controller()}};
  VirtualMemory uut{champsim::modules::ModuleBuilder{"uut", "VMEM", nullptr, champsim::defaults::default_vmem()}
      .add_parameter("dram", static_cast<champsim::modules::memory_controller_module*>(&dram))};

  champsim::data::bits expected_value{LOG2_PAGE_SIZE + (log2_pte_page_size - champsim::lg2(pte_entry::byte_multiple)) * (level - 1)};
  REQUIRE(uut.shamt(level) == expected_value);
}

TEST_CASE("The virtual memory evaluates the correct offsets")
{
  constexpr std::size_t log2_pte_page_size = 12;

  auto level = GENERATE(as<unsigned>{}, 1, 2, 3, 4, 5);

  MEMORY_CONTROLLER dram{champsim::modules::ModuleBuilder{"dram", "DRAM", nullptr, champsim::defaults::default_memory_controller()}};
  VirtualMemory uut{champsim::modules::ModuleBuilder{"uut", "VMEM", nullptr, champsim::defaults::default_vmem()}
      .add_parameter("dram", static_cast<champsim::modules::memory_controller_module*>(&dram))};

  champsim::address addr{(0xffff'ffff'ffe0'0000 | (level << LOG2_PAGE_SIZE)) << ((level - 1) * 9)};
  REQUIRE(uut.get_offset(addr, level) == level);
}
