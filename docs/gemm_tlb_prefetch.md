# GEMM TLB prefetcher experiment on ChampSim

This tree contains a DTLB prefetcher for the PIM/AMX GEMM trace experiment:

- `prefetcher/pc_role_tlb_stride/`: PC+role indexed VPN-stride TLB prefetcher.
- `gemm_configs/baseline_no_tlb_prefetch.json`: same CPU/TLB/PTW setting, no DTLB prefetcher.
- `gemm_configs/pc_role_tlb_stride.json`: enables `pc_role_tlb_stride` on DTLB.

The implemented prefetcher is intentionally close to the offline model used in `gemm_realamx`: it observes DTLB demand misses from LOAD/WRITE/RFO, indexes the stride table by instruction PC, learns VPN stride, and issues a degree-1 page prefetch after confidence reaches 2. For true `PC+role`, encode role into the trace IP before running ChampSim, for example `ip = (real_pc << 2) | role_id`, where `role_id = 0/1/2` means A/B/C. If the trace keeps the original machine PC unchanged, ChampSim will evaluate only a PC-indexed prefetcher, not PC+role.

## Build on the server

```bash
cd ~/projects/ChampSim
git submodule update --init --recursive

# First-time dependency setup. If vcpkg was already bootstrapped, skip this block.
./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg install

# Baseline binary
./config.sh gemm_configs/baseline_no_tlb_prefetch.json
make -j"$(nproc)"
cp bin/champsim bin/champsim-baseline-no-tlb-prefetch

# Prefetcher binary
./config.sh gemm_configs/pc_role_tlb_stride.json
make -j"$(nproc)"
cp bin/champsim bin/champsim-pc-role-tlb-stride
```

## Run baseline and prefetcher

```bash
trace=/path/to/your/gemm_pim_pc_role_trace.champsimtrace.xz
out=results/gemm_tlb_prefetch_$(date +%Y%m%d-%H%M%S)
mkdir -p "$out"

bin/champsim-baseline-no-tlb-prefetch \
  --warmup-instructions 20000000 \
  --simulation-instructions 200000000 \
  "$trace" | tee "$out/baseline.txt"

bin/champsim-pc-role-tlb-stride \
  --warmup-instructions 20000000 \
  --simulation-instructions 200000000 \
  "$trace" | tee "$out/pc_role_tlb_stride.txt"
```

If the generated GEMM trace is short, set warmup to 0 and simulate about one trace pass. For a CSV with 524288 GEMM records, the converter emits 524288 * 3 = 1572864 ChampSim instructions, so use 1600000 rather than 1000000000:

```bash
bin/champsim-baseline-no-tlb-prefetch \
  --warmup-instructions 0 \
  --simulation-instructions 1600000 \
  "$trace" | tee "$out/baseline.onepass.txt"

bin/champsim-pc-role-tlb-stride \
  --warmup-instructions 0 \
  --simulation-instructions 1600000 \
  "$trace" | tee "$out/pc_role_tlb_stride.full.txt"
```

## Trace requirement

Standard ChampSim traces do not know the GEMM operand role. To test `PC+role`, the trace generator or CSV-to-ChampSim converter must emit three synthetic memory references per fused GEMM instruction:

```text
ip = (pim_gemm_pc << 2) | 0, load A_tile_base
ip = (pim_gemm_pc << 2) | 1, load B_tile_base_or_packed_B_tile_base
ip = (pim_gemm_pc << 2) | 2, load/store C_tile_base
```

For the high-level oneDNN matmul path, B means the packed-B scratchpad address used by the compute microkernel, not the original raw `B` matrix address. If you want to model full tile footprint instead of only tile base translation, expand each tile base into all pages touched by the tile before writing the ChampSim trace.

This repository includes a tile-base converter for the CSV produced by `gemm_realamx`:

```bash
cd ~/projects/ChampSim
mkdir -p traces

python3 gemm_tools/gemm_csv_to_champsim_trace.py \
  /path/to/matmul-attn-out-m256-n4096-k4096.csv \
  traces/matmul-attn-m256-n4096-k4096.pc_role.champsimtrace.xz \
  --pc-period 4
```

Use `--max-records 100000` for a quick debug trace. The converter writes A and packed-B as loads and C as a store, so the DTLB prefetcher configuration uses `prefetch_activate: "LOAD,WRITE"`.

## Metrics to compare

Read these fields from `baseline.txt` and `pc_role_tlb_stride.txt`: IPC/cycles, DTLB hit/miss, STLB hit/miss, PTW activity/page-walk count if printed by the current ChampSim version, and the module lines `pc_role_tlb_stride trained/predictions/issued/useful`. Improvement is valid only if DTLB misses or exposed cycles go down without excessive useless prefetches, PTW queue pressure, or cache/PWC pollution.

Change address-translation hardware in the JSON files: `DTLB`/`STLB` `sets`, `ways`, `rq_size`, `wq_size`, `pq_size`, `mshr_size`, `latency`; `PTW` `rq_size`, `mshr_size`, `max_read`, `max_write`, and `pscl5/4/3/2` set/way fields. For Xeon-like stress tests, keep 4 KiB pages and 5-level page tables, then sweep DTLB/STLB capacity and PTW `mshr_size`.
