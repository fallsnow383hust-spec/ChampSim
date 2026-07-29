#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
csv="${1:?usage: bash gemm_tools/run_star_tlb_accuracy.sh TRACE.csv [OUT_DIR]}"
stamp="$(date +%Y%m%d-%H%M%S)"
out="${2:-${root}/results/pim_star_tlb_accuracy_${stamp}}"
mkdir -p "${out}" "${root}/traces"

csv="$(realpath "${csv}")"
base="$(basename "${csv}" .csv)"
trace="${root}/traces/${base}.star-base-only.champsimtrace.xz"
manifest="${trace}.json"

convert_args=(--base-only)
if [[ -n "${MAX_PIM_RECORDS:-}" ]]; then
  convert_args+=(--max-records "${MAX_PIM_RECORDS}")
fi
python3 "${root}/gemm_tools/star_pim_csv_to_champsim_trace.py" \
  "${csv}" "${trace}" --manifest "${manifest}" "${convert_args[@]}"

read -r sim_instr descriptor_count < <(
  python3 -c \
    'import json,sys; m=json.load(open(sys.argv[1])); print(m["simulation_instructions"], m["input_pim_records"])' \
    "${manifest}"
)

star_bin="${root}/bin/champsim-star-tlb-accuracy"
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  "${root}/config.sh" "${root}/gemm_configs/stlb_star_tlb.json"
  make -C "${root}" -j"${BUILD_JOBS:-$(nproc)}"
  cp "${root}/bin/champsim" "${star_bin}"
fi

STAR_TLB_DESCRIPTOR_CSV="${csv}" \
STAR_TLB_DESCRIPTOR_LIMIT="${descriptor_count}" \
STAR_TLB_ACCURACY_ONLY=1 \
STAR_TLB_ACCURACY_LOG="${out}/rprg-acc1-events.csv" \
STAR_TLB_GRAPH_LOG="${out}/rprg-graph-events.csv" \
STAR_TLB_BRANCH_LOG="${out}/branch-prediction-events.csv" \
"${star_bin}" --warmup-instructions 0 \
  --simulation-instructions "${sim_instr}" "${trace}" \
  | tee "${out}/star_tlb_accuracy.txt"

pdq_dispatch="$(
  sed -n 's/.* pdq_dispatch:\([0-9][0-9]*\) .*/\1/p' "${out}/star_tlb_accuracy.txt" | tail -n 1
)"
pdq_commit="$(
  sed -n 's/.* pdq_commit:\([0-9][0-9]*\) .*/\1/p' "${out}/star_tlb_accuracy.txt" | tail -n 1
)"
pdq_stall="$(
  sed -n 's/.* pdq_stall:\([0-9][0-9]*\) .*/\1/p' "${out}/star_tlb_accuracy.txt" | tail -n 1
)"
pdq_high_watermark="$(
  sed -n 's/.* pdq_high_watermark:\([0-9][0-9]*\) .*/\1/p' "${out}/star_tlb_accuracy.txt" | tail -n 1
)"
seq_conflict="$(
  sed -n 's/.* seq_conflict:\([0-9][0-9]*\) .*/\1/p' "${out}/star_tlb_accuracy.txt" | tail -n 1
)"
if [[ -z "${pdq_dispatch}" || -z "${pdq_commit}" || -z "${pdq_stall}" || -z "${pdq_high_watermark}" || -z "${seq_conflict}" ]]; then
  echo "ERROR: PDQ loss-free acceptance failed: missing simulator counters" >&2
  exit 2
fi
if (( pdq_dispatch != descriptor_count || pdq_commit != descriptor_count || pdq_stall != 0 || seq_conflict != 0 )); then
  echo "ERROR: PDQ loss-free acceptance failed: descriptors=${descriptor_count} dispatch=${pdq_dispatch} commit=${pdq_commit} drops=${pdq_stall} seq_conflict=${seq_conflict}" >&2
  exit 2
fi

python3 "${root}/gemm_tools/summarize_star_tlb_accuracy.py" \
  "${out}/rprg-acc1-events.csv" "${out}/rprg-graph-events.csv" \
  --manifest "${manifest}" --output-dir "${out}"

python3 "${root}/gemm_tools/summarize_star_tlb_branch_accuracy.py" \
  "${out}/branch-prediction-events.csv" "${out}/rprg-acc1-events.csv" \
  --output-dir "${out}"

echo "accuracy-only mode: no translation prefetches issued"
echo "PIM descriptors: ${descriptor_count}"
echo "base-only trace: ${trace}"
echo "PDQ loss-free acceptance: PASS (capacity=128, high_watermark=${pdq_high_watermark}, dispatch=${pdq_dispatch}, commit=${pdq_commit}, drops=0, seq_conflict=0)"
echo "summary: ${out}/rprg-accuracy-summary.txt"
echo "Acc@1 events: ${out}/rprg-acc1-events.csv"
echo "RPRG CRUD events: ${out}/rprg-graph-events.csv"
echo "branch summary: ${out}/branch-prediction-summary.txt"
echo "branch errors: ${out}/branch-prediction-errors.csv"
echo "branch/RPRG correlation: ${out}/branch-rprg-context-error-correlation.csv"
