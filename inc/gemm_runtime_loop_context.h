#ifndef GEMM_RUNTIME_LOOP_CONTEXT_H
#define GEMM_RUNTIME_LOOP_CONTEXT_H

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace gemm_runtime_loop_context
{
// Single-core experiment sideband. It models a small hardware loop detector:
// a predicted taken backward branch allocates a runtime context id, and the id
// is stamped onto following PIM uops by dynamic instruction id. No GEMM loop
// coordinate or trace-provided phase is consumed by the prefetcher.
struct runtime_state {
  static constexpr uint64_t PIM_PC_BEGIN = 0x400000;
  static constexpr uint64_t PIM_PC_END = 0x500000;
  static constexpr uint64_t LOOP_BRANCH_PC_BEGIN = 0x500000;
  static constexpr uint64_t LOOP_BRANCH_PC_END = 0x500100;
  static constexpr uint8_t MAX_CONTEXTS = 6;

  std::unordered_map<uint64_t, uint8_t> instruction_context{};
  std::unordered_map<uint64_t, uint8_t> branch_context{};
  std::array<uint64_t, MAX_CONTEXTS + 1> context_branch_pc{};
  uint8_t current_context = 0;
  uint8_t next_context = 1;
  uint64_t predicted_backedges = 0;
  uint64_t actual_backedges = 0;
  uint64_t correctly_predicted_backedges = 0;
  uint64_t missed_backedges = 0;
  uint64_t false_backedges = 0;
  uint64_t context_overflow = 0;

  void reset()
  {
    instruction_context.clear();
    branch_context.clear();
    context_branch_pc = {};
    current_context = 0;
    next_context = 1;
    predicted_backedges = 0;
    actual_backedges = 0;
    correctly_predicted_backedges = 0;
    missed_backedges = 0;
    false_backedges = 0;
    context_overflow = 0;
  }

  void observe_branch(uint64_t branch_pc, bool predicted_taken, uint64_t predicted_target, bool actual_taken, uint64_t actual_target)
  {
    if (branch_pc < LOOP_BRANCH_PC_BEGIN || branch_pc >= LOOP_BRANCH_PC_END)
      return;

    const bool predicted_backedge = predicted_taken && predicted_target >= PIM_PC_BEGIN && predicted_target < PIM_PC_END
        && predicted_target < branch_pc;
    const bool actual_backedge = actual_taken && actual_target >= PIM_PC_BEGIN && actual_target < PIM_PC_END && actual_target < branch_pc;
    actual_backedges += actual_backedge;
    predicted_backedges += predicted_backedge;
    correctly_predicted_backedges += predicted_backedge && actual_backedge;
    missed_backedges += !predicted_backedge && actual_backedge;
    false_backedges += predicted_backedge && !actual_backedge;
    if (!predicted_backedge)
      return;

    const auto found = branch_context.find(branch_pc);
    if (found != branch_context.end()) {
      current_context = found->second;
      return;
    }
    if (next_context > MAX_CONTEXTS) {
      ++context_overflow;
      current_context = 0;
      return;
    }
    current_context = next_context++;
    branch_context.emplace(branch_pc, current_context);
    context_branch_pc[current_context] = branch_pc;
  }

  void stamp_instruction(uint64_t instr_id, uint64_t ip)
  {
    if (ip >= PIM_PC_BEGIN && ip < PIM_PC_END)
      instruction_context.insert_or_assign(instr_id, current_context);
  }

  [[nodiscard]] std::optional<uint8_t> context_for(uint64_t instr_id) const
  {
    const auto found = instruction_context.find(instr_id);
    if (found == instruction_context.end())
      return std::nullopt;
    return found->second;
  }
};

inline runtime_state state{};
} // namespace gemm_runtime_loop_context

#endif
