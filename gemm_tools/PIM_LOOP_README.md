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

The command generates two ChampSim traces with identical addresses and branch
events. `loop-plain` sets phase bits to zero; `loop-context` encodes the actual
`K_PROGRESS/K_TO_IR/.../PC_TO_JC` context. It then runs:

1. no STLB prefetch;
2. the new predictor with PC+role only;
3. the same predictor/resources with PC+role+loop-boundary.

Results are written under `results/pim_loop_boundary_<time>/`.
Both configurations use the paper-style 4-level translation model: 64-entry
L1 DTLB, 1536-entry 6-way STLB, four translation MSHRs, and the three PSC
levels PML4/PDP/PD. Only the prefetch-enabled STLB adds a 32-entry PQ.

One CSV row remains one logical PIM instruction. The converter expands its
A/B/C translation requests into three ChampSim memory micro-ops only because
the unmodified ChampSim prefetcher callback otherwise cannot identify the
operand role. The phase-tagged run is an information-value/upper-bound test:
`loop_boundary_before` is encoded into low PC bits as a proxy for the loop-exit
level that a hardware branch/loop detector would supply; it is not a claim
that the architectural PIM PC changes at each boundary.

Read `summary.txt` for global IPC/cycle/STLB results, per-role A/B/C coverage
and accuracy, and the per-boundary lead-distance breakdown. Raw counters are
in `baseline.txt`, `loop_pc_role.txt`, and `loop_boundary.txt`.
