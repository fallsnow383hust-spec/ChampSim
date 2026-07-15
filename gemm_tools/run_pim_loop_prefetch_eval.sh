#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
csv="${1:?usage: bash gemm_tools/run_pim_loop_prefetch_eval.sh TRACE.csv [OUT_DIR]}"
stamp="$(date +%Y%m%d-%H%M%S)"
out="${2:-${root}/results/pim_loop_boundary_${stamp}}"
mkdir -p "${out}" "${root}/traces"

base="$(basename "${csv}" .csv)"
runtime_trace="${root}/traces/${base}.loop-runtime.champsimtrace.xz"
runtime_manifest="${runtime_trace}.json"

python3 "${root}/gemm_tools/pim_loop_csv_to_champsim_trace.py" \
  "${csv}" "${runtime_trace}" --manifest "${runtime_manifest}" --no-phase-context

sim_instr="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["simulation_instructions"])' "${runtime_manifest}")"

build_one() {
  local config="$1"
  local binary="$2"
  "${root}/config.sh" "${config}"
  make -C "${root}" -j"${BUILD_JOBS:-$(nproc)}"
  cp "${root}/bin/champsim" "${binary}"
}

baseline_bin="${root}/bin/champsim-pim-loop-baseline"
loop_bin="${root}/bin/champsim-pim-loop-boundary"
build_one "${root}/gemm_configs/stlb_pim_loop_baseline.json" "${baseline_bin}"
build_one "${root}/gemm_configs/stlb_loop_boundary_tlb_realfill.json" "${loop_bin}"

run_one() {
  local binary="$1"
  local trace="$2"
  local result="$3"
  local event_log="${4:-}"
  if [[ -n "${event_log}" ]]; then
    GEMM_TLB_EVENT_LOG="${event_log}" "${binary}" --warmup-instructions 0 \
      --simulation-instructions "${sim_instr}" "${trace}" | tee "${result}"
  else
    "${binary}" --warmup-instructions 0 \
      --simulation-instructions "${sim_instr}" "${trace}" | tee "${result}"
  fi
}

# Both predictor runs consume exactly the same PCs, addresses, and branches.
# The first disables the runtime backedge detector; the second enables it.
run_one "${baseline_bin}" "${runtime_trace}" "${out}/baseline.txt"
GEMM_RUNTIME_LOOP_CONTEXT=0 run_one "${loop_bin}" "${runtime_trace}" "${out}/loop_pc_role.txt" \
  "${out}/loop_pc_role.prefetch-events.csv"
GEMM_RUNTIME_LOOP_CONTEXT=1 run_one "${loop_bin}" "${runtime_trace}" "${out}/loop_boundary.txt" \
  "${out}/loop_boundary.prefetch-events.csv"

python3 "${root}/gemm_tools/summarize_pim_loop_prefetch.py" \
  "${out}/baseline.txt" "${out}/loop_pc_role.txt" \
  "${out}/loop_boundary.txt" --output "${out}/summary.txt"

echo "ChampSim instructions: ${sim_instr}"
echo "results: ${out}"
echo "summary: ${out}/summary.txt"
echo "per-request timing: ${out}/loop_pc_role.prefetch-events.csv"
echo "per-request timing: ${out}/loop_boundary.prefetch-events.csv"
