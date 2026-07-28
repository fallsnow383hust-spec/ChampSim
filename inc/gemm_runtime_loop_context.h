#ifndef GEMM_RUNTIME_LOOP_CONTEXT_H
#define GEMM_RUNTIME_LOOP_CONTEXT_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace gemm_runtime_loop_context
{
using predicted_backedge_observer_type = void (*)(uint64_t, uint8_t);
using descriptor_dispatch_observer_type = void (*)(uint64_t, uint64_t, uint8_t);
using descriptor_observer_type = void (*)(uint64_t, uint64_t, uint8_t);
inline predicted_backedge_observer_type predicted_backedge_observer = nullptr;
inline descriptor_dispatch_observer_type descriptor_dispatch_observer = nullptr;
inline descriptor_observer_type descriptor_observer = nullptr;

// Single-core experiment sideband. It models a small hardware loop detector:
// a predicted taken backward branch allocates a runtime context id, and the id
// is stamped onto following PIM uops by dynamic instruction id. No GEMM loop
// coordinate or trace-provided phase is consumed by the prefetcher.
struct runtime_state {
  static constexpr uint64_t PIM_PC_BEGIN = 0x400000;
  static constexpr uint64_t PIM_PC_END = 0x500000;
  static constexpr uint64_t LOOP_BRANCH_PC_BEGIN = 0x500000;
  static constexpr uint64_t LOOP_BRANCH_PC_END = 0x500100;
  static constexpr uint64_t PIMCFG_MARKER_BIT = 0x4;
  static constexpr uint8_t MAX_CONTEXTS = 6;

  std::unordered_map<uint64_t, uint8_t> instruction_context{};
  std::unordered_map<uint64_t, uint64_t> instruction_descriptor{};
  std::unordered_map<uint64_t, uint8_t> branch_context{};
  std::array<uint64_t, MAX_CONTEXTS + 1> context_branch_pc{};
  uint8_t current_context = 0;
  uint8_t next_context = 1;
  uint64_t next_dispatched_descriptor = 0;
  uint64_t next_descriptor = 0;
  uint64_t predicted_backedges = 0;
  uint64_t actual_backedges = 0;
  uint64_t correctly_predicted_backedges = 0;
  uint64_t missed_backedges = 0;
  uint64_t false_backedges = 0;
  uint64_t context_overflow = 0;

  void reset()
  {
    instruction_context.clear();
    instruction_descriptor.clear();
    branch_context.clear();
    context_branch_pc = {};
    current_context = 0;
    next_context = 1;
    next_dispatched_descriptor = 0;
    next_descriptor = 0;
    predicted_backedges = 0;
    actual_backedges = 0;
    correctly_predicted_backedges = 0;
    missed_backedges = 0;
    false_backedges = 0;
    context_overflow = 0;
  }

  void observe_branch(uint64_t instr_id, uint64_t branch_pc, bool predicted_taken, uint64_t predicted_target, bool actual_taken,
                      uint64_t actual_target)
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
    if (!predicted_backedge && !actual_backedge)
      return;

    uint8_t context = 0;
    const auto found = branch_context.find(branch_pc);
    if (found != branch_context.end()) {
      context = found->second;
    } else if (next_context > MAX_CONTEXTS) {
      ++context_overflow;
    } else {
      context = next_context++;
      branch_context.emplace(branch_pc, context);
      context_branch_pc[context] = branch_pc;
    }

    // Prediction may trigger an early prefetch. The correct-path trace is
    // stamped with the resolved context, modeling checkpoint restoration.
    if (predicted_backedge && predicted_backedge_observer != nullptr)
      predicted_backedge_observer(instr_id, context);
    if (actual_backedge)
      current_context = context;
  }

  void stamp_instruction(uint64_t instr_id, uint64_t ip)
  {
    if (ip >= PIM_PC_BEGIN && ip < PIM_PC_END) {
      instruction_context.insert_or_assign(instr_id, current_context);
      if ((ip & PIMCFG_MARKER_BIT) != 0) {
        const auto descriptor_index = next_dispatched_descriptor++;
        instruction_descriptor.insert_or_assign(instr_id, descriptor_index);
        if (descriptor_dispatch_observer != nullptr)
          descriptor_dispatch_observer(descriptor_index, instr_id, current_context);
      }
    }
  }

  void retire_instruction(uint64_t instr_id, uint64_t ip)
  {
    const auto found = instruction_context.find(instr_id);
    if (ip >= PIM_PC_BEGIN && ip < PIM_PC_END && (ip & PIMCFG_MARKER_BIT) != 0) {
      const auto context = found == instruction_context.end() ? uint8_t{0} : found->second;
      const auto descriptor_found = instruction_descriptor.find(instr_id);
      const auto descriptor_index = descriptor_found == instruction_descriptor.end() ? next_descriptor : descriptor_found->second;
      if (descriptor_observer != nullptr)
        descriptor_observer(descriptor_index, instr_id, context);
      next_descriptor = std::max(next_descriptor, descriptor_index + 1);
      if (descriptor_found != instruction_descriptor.end())
        instruction_descriptor.erase(descriptor_found);
    }

    // A retired instruction has completed every memory operation, so its
    // issue-time loop-context stamp is no longer needed by the STLB.
    if (found != instruction_context.end())
      instruction_context.erase(found);
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
