#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
csv="${1:?usage: bash gemm_tools/run_pim_loop_prefetch_eval.sh TRACE.csv [OUT_DIR]}"
stamp="$(date +%Y%m%d-%H%M%S)"
out="${2:-${root}/results/pim_loop_boundary_${stamp}}"
mkdir -p "${out}" "${root}/traces"

base="$(basename "${csv}" .csv)"
plain="${root}/traces/${base}.loop-plain.champsimtrace.xz"
context="${root}/traces/${base}.loop-context.champsimtrace.xz"
plain_manifest="${plain}.json"
context_manifest="${context}.json"

python3 "${root}/gemm_tools/pim_loop_csv_to_champsim_trace.py" \
  "${csv}" "${plain}" --manifest "${plain_manifest}" --no-phase-context
python3 "${root}/gemm_tools/pim_loop_csv_to_champsim_trace.py" \
  "${csv}" "${context}" --manifest "${context_manifest}"

sim_instr="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["simulation_instructions"])' "${plain_manifest}")"

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
  "${binary}" --warmup-instructions 0 \
    --simulation-instructions "${sim_instr}" "${trace}" | tee "${result}"
}

# The same prefetcher binary is run twice. loop_pc_role has phase bits forced
# to zero; loop_boundary receives the actual loop-context bits. This isolates
# the benefit of boundary context without changing predictor resources.
run_one "${baseline_bin}" "${plain}" "${out}/baseline.txt"
run_one "${loop_bin}" "${plain}" "${out}/loop_pc_role.txt"
run_one "${loop_bin}" "${context}" "${out}/loop_boundary.txt"

python3 "${root}/gemm_tools/summarize_pim_loop_prefetch.py" \
  "${out}/baseline.txt" "${out}/loop_pc_role.txt" \
  "${out}/loop_boundary.txt" --output "${out}/summary.txt"

echo "ChampSim instructions: ${sim_instr}"
echo "results: ${out}"
echo "summary: ${out}/summary.txt"
