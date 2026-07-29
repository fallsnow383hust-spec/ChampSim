#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
csv="${1:?usage: bash gemm_tools/run_star_tlb_graph_accuracy.sh TRACE.csv [OUT_DIR]}"
stamp="$(date +%Y%m%d-%H%M%S)"
out="${2:-${root}/results/pim_star_tlb_graph_accuracy_${stamp}}"
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

star_bin="${root}/bin/champsim-star-tlb-graph-accuracy"
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  "${root}/config.sh" "${root}/gemm_configs/stlb_star_tlb.json"
  make -C "${root}" -j"${BUILD_JOBS:-$(nproc)}"
  cp "${root}/bin/champsim" "${star_bin}"
fi

STAR_TLB_DESCRIPTOR_CSV="${csv}" \
STAR_TLB_DESCRIPTOR_LIMIT="${descriptor_count}" \
STAR_TLB_GRAPH_ORACLE_ONLY=1 \
STAR_TLB_ORACLE_GRAPH_LOG="${out}/rprg-oracle-acc1-events.csv" \
STAR_TLB_GRAPH_LOG="${out}/rprg-oracle-graph-events.csv" \
"${star_bin}" --warmup-instructions 0 \
  --simulation-instructions "${sim_instr}" "${trace}" \
  | tee "${out}/star_tlb_graph_accuracy.txt"

python3 "${root}/gemm_tools/summarize_star_tlb_graph_accuracy.py" \
  "${out}/rprg-oracle-acc1-events.csv" "${out}/rprg-oracle-graph-events.csv" \
  --manifest "${manifest}" --output-dir "${out}"

echo "pure graph mode: branch/LBQ/PDQ bypassed; no translation prefetches issued"
echo "query order: query D_i->D_(i+1), score, then train with D_(i+1)"
echo "PIM descriptors: ${descriptor_count}"
echo "base-only trace: ${trace}"
echo "summary: ${out}/rprg-oracle-accuracy-summary.txt"
echo "oracle Acc@1 events: ${out}/rprg-oracle-acc1-events.csv"
echo "oracle RPRG CRUD events: ${out}/rprg-oracle-graph-events.csv"
