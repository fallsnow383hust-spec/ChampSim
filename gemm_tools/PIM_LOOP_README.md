# PIM five-loop / loop-boundary STLB experiment

The repositories stay separate:

- `~/projects/gemm_realamx`: compile the canonical five-loop driver and write
  one CSV row per logical three-address PIM32 instruction.
- `~/projects/ChampSim`: convert that CSV, build the STLB prefetcher, and run
  the three-way comparison below.

```bash
cd ~/projects/ChampSim
bash gemm_tools/run_pim_loop_prefetch_eval.sh \
  ~/projects/gemm_realamx/outputs/<tag>/<tag>.csv
```

The command generates one ChampSim trace and runs all configurations on that
identical PC/address/branch stream. Memory PCs encode only static PIM site and
A/B/C role; no loop phase is encoded. It then runs:

1. no STLB prefetch;
2. the new predictor with PC+role only;
3. the same predictor/resources with runtime backedge context.

Results are written under `results/pim_loop_boundary_<time>/`.
Both configurations use the paper-style 4-level translation model: 64-entry
L1 DTLB, 1536-entry 6-way STLB, four translation MSHRs, and the three PSC
levels PML4/PDP/PD. Only the prefetch-enabled STLB adds a 32-entry PQ.

One CSV row remains one logical PIM instruction. The converter expands its
A/B/C translation requests into three ChampSim memory micro-ops only because
the unmodified ChampSim prefetcher callback otherwise cannot identify the
operand role. It reconstructs the canonical loop-latch branches before each
PIM record, but the predictor receives neither `jc/pc/ic/jr/ir/q` nor the CSV
phase label. At fetch/decode, a predicted taken branch whose target is lower
than its PC allocates a runtime backedge context. That context is stamped by
dynamic instruction id so OoO arrival at the STLB cannot associate an older
memory request with a newer branch.

The runtime predictor keeps one shared `(PC,role)` history instead of splitting the
whole table by phase. It learns byte strides, but injects a request only when
the predicted byte address is on a different 4-KiB page. The context-free
steady stream requires confidence 2. Every dynamically
identified backedge context uses confidence 1 and may issue two causal predictions: the address
after this boundary and the next recurrence of this boundary. A request is
issued only when the predicted VPN differs from the current VPN. Before issue,
the module checks the local STLB tags, MSHRs/fill queue, and its own pending
table.

`inc/cache.h` and `src/cache.cc` extend the prefetch callback with the original
byte address and `instr_id`. The former preserves page offsets for byte-stride
training; the latter suppresses repeated tag-check callbacks for one demand.
A per-cycle clock then classifies every accepted prediction as timely, late, too early, or never
demanded before the trace ends. For completed late requests, `late_by` is the
translation fill cycle minus the first demand cycle.

`GEMM_RUNTIME_LOOP_CONTEXT=0` disables context lookup for the PC+role control;
`=1` enables it. Both runs use the same trace and predictor resources.

Read `summary.txt` for global IPC/cycle/DTLB/STLB results, A/B/C outcomes,
local-filter counts, prediction-source results, and per-boundary cycle timing.
Raw counters are in `baseline.txt`, `loop_pc_role.txt`, and
`loop_boundary.txt`. The two `*.prefetch-events.csv` files contain one row per
accepted cross-page request with issue/demand/fill cycles and its final outcome;
`never_demanded` means no later demand appeared before this finite trace ended.
