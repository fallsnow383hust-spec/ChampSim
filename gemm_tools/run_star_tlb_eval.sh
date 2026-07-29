#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
csv="${1:?usage: bash gemm_tools/run_star_tlb_eval.sh TRACE.csv [OUT_DIR]}"
stamp="$(date +%Y%m%d-%H%M%S)"
out="${2:-${root}/results/pim_star_tlb_${stamp}}"
mkdir -p "${out}" "${root}/traces"

csv="$(realpath "${csv}")"
base="$(basename "${csv}" .csv)"
trace="${root}/traces/${base}.star-full-pages.champsimtrace.xz"
manifest="${trace}.json"

convert_args=()
if [[ -n "${MAX_PIM_RECORDS:-}" ]]; then
  convert_args+=(--max-records "${MAX_PIM_RECORDS}")
fi
python3 "${root}/gemm_tools/star_pim_csv_to_champsim_trace.py" \
  "${csv}" "${trace}" --manifest "${manifest}" "${convert_args[@]}"

sim_instr="$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["simulation_instructions"])' \
    "${manifest}"
)"

build_one() {
  local config="$1"
  local binary="$2"
  "${root}/config.sh" "${config}"
  make -C "${root}" -j"${BUILD_JOBS:-$(nproc)}"
  cp "${root}/bin/champsim" "${binary}"
}

baseline_bin="${root}/bin/champsim-star-baseline"
star_bin="${root}/bin/champsim-star-tlb"
build_one "${root}/gemm_configs/stlb_pim_loop_baseline.json" "${baseline_bin}"
build_one "${root}/gemm_configs/stlb_star_tlb.json" "${star_bin}"

"${baseline_bin}" --warmup-instructions 0 \
  --simulation-instructions "${sim_instr}" "${trace}" \
  | tee "${out}/baseline.txt"

STAR_TLB_EVENT_LOG="${out}/star_tlb.prefetch-events.csv" \
"${star_bin}" --warmup-instructions 0 \
  --simulation-instructions "${sim_instr}" "${trace}" \
  | tee "${out}/star_tlb.txt"

python3 "${root}/gemm_tools/summarize_star_tlb.py" \
  "${out}/baseline.txt" "${out}/star_tlb.txt" \
  --manifest "${manifest}" --output "${out}/summary.txt"

echo "ChampSim instructions: ${sim_instr}"
echo "full-page trace: ${trace}"
echo "results: ${out}"
echo "summary: ${out}/summary.txt"
echo "per-prefetch timing: ${out}/star_tlb.prefetch-events.csv"
