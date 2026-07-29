#ifndef GEMM_RUNTIME_LOOP_CONTEXT_H
#define GEMM_RUNTIME_LOOP_CONTEXT_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace gemm_runtime_loop_context
{
using asid_type = uint16_t;

struct loop_identity {
  asid_type asid = 0;
  uint64_t branch_pc = 0;
  uint64_t target_pc = 0;

  bool operator==(const loop_identity& other) const
  {
    return asid == other.asid && branch_pc == other.branch_pc && target_pc == other.target_pc;
  }
};

struct loop_identity_hash {
  std::size_t operator()(const loop_identity& identity) const
  {
    auto mixed = identity.branch_pc ^ (identity.branch_pc >> 17) ^ identity.target_pc ^ (identity.target_pc >> 29);
    mixed ^= static_cast<uint64_t>(identity.asid) * 0x9e3779b97f4a7c15ULL;
    return static_cast<std::size_t>(mixed);
  }
};

struct pim_descriptor {
  uint64_t site_pc = 0;
  uint64_t a_base = 0;
  uint64_t b_base = 0;
  uint64_t c_base = 0;
};

struct resolved_backedge_event {
  uint64_t instr_id = 0;
  uint64_t branch_pc = 0;
  uint64_t predicted_target = 0;
  uint64_t actual_target = 0;
  asid_type asid = 0;
  uint8_t predicted_context = 0;
  uint8_t actual_context = 0;
  bool predicted_taken = false;
  bool actual_taken = false;
  bool predicted_backedge = false;
  bool actual_backedge = false;
};

using predicted_backedge_observer_type = void (*)(uint64_t, uint8_t);
using resolved_backedge_observer_type = void (*)(const resolved_backedge_event&);
using descriptor_dispatch_observer_type =
    void (*)(uint64_t, uint64_t, asid_type, uint8_t, const pim_descriptor&);
using descriptor_observer_type =
    void (*)(uint64_t, uint64_t, asid_type, uint8_t, const pim_descriptor&);
inline predicted_backedge_observer_type predicted_backedge_observer = nullptr;
inline resolved_backedge_observer_type resolved_backedge_observer = nullptr;
inline descriptor_dispatch_observer_type descriptor_dispatch_observer = nullptr;
inline descriptor_observer_type descriptor_observer = nullptr;

// Hardware-visible loop detector. Context ids are opaque: they are learned from
// resolved (ASID, branch PC, backward target) tuples and stamped onto following
// PIM uops. Predicted backedges may only look up an already learned tuple; they
// never allocate from the actual outcome. No source-level loop coordinate,
// GEMM phase name, or trace-provided semantic label is consumed.
struct runtime_state {
  static constexpr uint64_t PIM_PC_BEGIN = 0x400000;
  static constexpr uint64_t PIM_PC_END = 0x500000;
  static constexpr uint64_t PIMCFG_MARKER_BIT = 0x4;
  static constexpr uint8_t MAX_CONTEXTS = 63;

  struct instruction_stamp {
    asid_type asid = 0;
    uint8_t context = 0;
  };

  struct descriptor_record {
    uint64_t index = 0;
    pim_descriptor value{};
  };

  struct pending_branch {
    asid_type asid = 0;
    uint64_t branch_pc = 0;
    uint64_t predicted_target = 0;
    uint8_t source_context = 0;
    uint8_t predicted_context = 0;
    bool predicted_taken = false;
    bool predicted_backedge = false;
  };

  std::unordered_map<uint64_t, instruction_stamp> instruction_context{};
  std::unordered_map<uint64_t, descriptor_record> instruction_descriptor{};
  std::unordered_map<uint64_t, std::array<uint64_t, 3>> instruction_operands{};
  std::unordered_map<uint64_t, pending_branch> pending_branches{};
  std::unordered_map<loop_identity, uint8_t, loop_identity_hash> loop_context{};
  std::unordered_map<asid_type, uint8_t> current_context{};
  std::array<uint64_t, MAX_CONTEXTS + 1> context_branch_pc{};
  std::array<uint64_t, MAX_CONTEXTS + 1> context_branch_target{};
  std::array<asid_type, MAX_CONTEXTS + 1> context_asid{};
  uint8_t next_context = 1;
  uint64_t next_dispatched_descriptor = 0;
  uint64_t next_descriptor = 0;
  uint64_t predicted_backedges = 0;
  uint64_t actual_backedges = 0;
  uint64_t correctly_predicted_backedges = 0;
  uint64_t missed_backedges = 0;
  uint64_t false_backedges = 0;
  uint64_t context_overflow = 0;
  uint64_t malformed_descriptors = 0;

  void reset()
  {
    instruction_context.clear();
    instruction_descriptor.clear();
    instruction_operands.clear();
    pending_branches.clear();
    loop_context.clear();
    current_context.clear();
    context_branch_pc = {};
    context_branch_target = {};
    context_asid = {};
    next_context = 1;
    next_dispatched_descriptor = 0;
    next_descriptor = 0;
    predicted_backedges = 0;
    actual_backedges = 0;
    correctly_predicted_backedges = 0;
    missed_backedges = 0;
    false_backedges = 0;
    context_overflow = 0;
    malformed_descriptors = 0;
  }

  [[nodiscard]] static bool is_backedge(bool taken, uint64_t branch_pc, uint64_t target_pc)
  {
    return taken && target_pc != 0 && target_pc < branch_pc;
  }

  [[nodiscard]] uint8_t find_context(asid_type asid, uint64_t branch_pc, uint64_t target_pc) const
  {
    const auto found = loop_context.find({asid, branch_pc, target_pc});
    return found == loop_context.end() ? uint8_t{0} : found->second;
  }

  uint8_t learn_context(asid_type asid, uint64_t branch_pc, uint64_t target_pc)
  {
    if (const auto known = find_context(asid, branch_pc, target_pc); known != 0)
      return known;
    if (next_context > MAX_CONTEXTS) {
      ++context_overflow;
      return 0;
    }

    const auto context = next_context++;
    loop_context.emplace(loop_identity{asid, branch_pc, target_pc}, context);
    context_branch_pc[context] = branch_pc;
    context_branch_target[context] = target_pc;
    context_asid[context] = asid;
    return context;
  }

  void predict_branch(uint64_t instr_id, asid_type asid, uint64_t branch_pc, bool predicted_taken,
                      uint64_t predicted_target)
  {
    const bool predicted_backedge = is_backedge(predicted_taken, branch_pc, predicted_target);
    predicted_backedges += predicted_backedge;
    const auto predicted_context =
        predicted_backedge ? find_context(asid, branch_pc, predicted_target) : uint8_t{0};
    const auto source_found = current_context.find(asid);
    const auto source_context = source_found == current_context.end() ? uint8_t{0} : source_found->second;
    pending_branches.insert_or_assign(
        instr_id,
        pending_branch{asid, branch_pc, predicted_target, source_context, predicted_context,
                       predicted_taken, predicted_backedge});

    // Only an identity learned by an older resolved branch may affect the
    // speculative path. The first encounter can never consume its own result.
    if (predicted_context != 0 && predicted_backedge_observer != nullptr)
      predicted_backedge_observer(instr_id, predicted_context);
    if (predicted_context != 0)
      current_context.insert_or_assign(asid, predicted_context);
  }

  void resolve_branch(uint64_t instr_id, bool actual_taken, uint64_t actual_target)
  {
    const auto found = pending_branches.find(instr_id);
    if (found == pending_branches.end())
      return;
    const auto prediction = found->second;
    pending_branches.erase(found);

    const bool actual_backedge = is_backedge(actual_taken, prediction.branch_pc, actual_target);
    const auto actual_context =
        actual_backedge ? learn_context(prediction.asid, prediction.branch_pc, actual_target) : uint8_t{0};
    actual_backedges += actual_backedge;
    correctly_predicted_backedges +=
        prediction.predicted_backedge && actual_backedge && prediction.predicted_target == actual_target;
    missed_backedges += !prediction.predicted_backedge && actual_backedge;
    false_backedges += prediction.predicted_backedge && !actual_backedge;

    const bool control_mispredicted = prediction.predicted_taken != actual_taken
        || (actual_taken && prediction.predicted_target != actual_target);
    if (control_mispredicted) {
      // Fetch is blocked for a misprediction, so resolution can safely restore
      // the correct checkpoint before correct-path fetching resumes.
      if (actual_context != 0)
        current_context.insert_or_assign(prediction.asid, actual_context);
      else if (prediction.source_context != 0)
        current_context.insert_or_assign(prediction.asid, prediction.source_context);
      else
        current_context.erase(prediction.asid);
    } else if (actual_context != 0 && prediction.predicted_context == 0) {
      // Correct first encounter: install the learned identity only if no
      // younger predicted backedge has advanced the speculative context.
      const auto current = current_context.find(prediction.asid);
      const auto unchanged = current == current_context.end()
          ? prediction.source_context == 0
          : current->second == prediction.source_context;
      if (unchanged)
        current_context.insert_or_assign(prediction.asid, actual_context);
    }

    if (resolved_backedge_observer != nullptr
        && (prediction.predicted_backedge || actual_backedge)) {
      const resolved_backedge_event event{
          instr_id, prediction.branch_pc, prediction.predicted_target, actual_target,
          prediction.asid, prediction.predicted_context, actual_context,
          prediction.predicted_taken, actual_taken,
          prediction.predicted_backedge, actual_backedge};
      resolved_backedge_observer(event);
    }
  }

  void stamp_instruction(uint64_t instr_id, asid_type asid, uint64_t ip, bool descriptor_valid = false,
                         uint64_t a_base = 0, uint64_t b_base = 0, uint64_t c_base = 0)
  {
    if (ip < PIM_PC_BEGIN || ip >= PIM_PC_END)
      return;

    const auto found = current_context.find(asid);
    const auto context = found == current_context.end() ? uint8_t{0} : found->second;
    instruction_context.insert_or_assign(instr_id, instruction_stamp{asid, context});
    if ((ip & PIMCFG_MARKER_BIT) == 0)
      return;
    if (!descriptor_valid) {
      ++malformed_descriptors;
      return;
    }

    const auto descriptor_index = next_dispatched_descriptor++;
    const pim_descriptor descriptor{
        ip & ~PIMCFG_MARKER_BIT, a_base, b_base, c_base};
    instruction_descriptor.insert_or_assign(instr_id, descriptor_record{descriptor_index, descriptor});
    instruction_operands.insert_or_assign(instr_id, std::array<uint64_t, 3>{a_base, b_base, c_base});
    if (descriptor_dispatch_observer != nullptr)
      descriptor_dispatch_observer(descriptor_index, instr_id, asid, context, descriptor);
  }

  void retire_instruction(uint64_t instr_id, uint64_t ip)
  {
    const auto found = instruction_context.find(instr_id);
    const auto descriptor_found = instruction_descriptor.find(instr_id);
    if (descriptor_found != instruction_descriptor.end()) {
      const auto stamp = found == instruction_context.end() ? instruction_stamp{} : found->second;
      if (descriptor_observer != nullptr)
        descriptor_observer(descriptor_found->second.index, instr_id, stamp.asid, stamp.context,
                            descriptor_found->second.value);
      next_descriptor = std::max(next_descriptor, descriptor_found->second.index + 1);
      instruction_descriptor.erase(descriptor_found);
    }

    // A retired instruction has completed every memory operation, so its
    // issue-time loop-context stamp is no longer needed by the STLB.
    if (found != instruction_context.end())
      instruction_context.erase(found);
    instruction_operands.erase(instr_id);
    (void)ip;
  }

  [[nodiscard]] std::optional<uint8_t> context_for(uint64_t instr_id) const
  {
    const auto found = instruction_context.find(instr_id);
    if (found == instruction_context.end())
      return std::nullopt;
    return found->second.context;
  }

  [[nodiscard]] std::optional<uint8_t> operand_role_for(uint64_t instr_id, uint64_t virtual_address) const
  {
    const auto found = instruction_operands.find(instr_id);
    if (found == instruction_operands.end())
      return std::nullopt;
    for (uint8_t role = 0; role < found->second.size(); ++role) {
      if (found->second[role] == virtual_address)
        return role;
    }
    return std::nullopt;
  }
};

inline runtime_state state{};
} // namespace gemm_runtime_loop_context

#endif
