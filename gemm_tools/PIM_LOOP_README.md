# PIM five-loop / G-LBTP STLB experiment

The repositories stay separate:

- `~/projects/pim-gemm-isa-sim`: run the canonical five-loop driver and write
  one CSV row per logical three-address PIM32 instruction.
- `~/projects/ChampSim`: reconstruct the branch stream, build G-LBTP, and run
  the comparison.

```bash
cd ~/projects/ChampSim
bash gemm_tools/run_pim_loop_prefetch_eval.sh \
  ~/projects/pim-gemm-isa-sim/results/<tag>/<tag>.instruction.csv
```

One CSV row remains one logical PIM instruction. The converter emits A-load,
B-load, and C-store micro-ops only so the STLB callback can identify the role.
Their PCs encode the static PIM site and a two-bit A/B/C role; loop phase and
loop variables are not encoded in the memory PC.

The command reconstructs conditional loop-latch branches before each PIM
record and runs the identical PC/address/branch stream three times:

1. no STLB prefetch;
2. G-LBTP with runtime context disabled, reducing the graph to PC+role;
3. G-LBTP with dynamically detected loop-PC vertices.

At fetch/decode, a predicted taken branch whose target is lower than its PC
allocates a small runtime context. It both triggers G-LBTP immediately and is
stamped onto subsequent PIM uops by dynamic instruction id. The trigger gives
the translation prefetch a loop-body head start; the stamp ensures that
out-of-order STLB callbacks train the correct boundary transition.

G-LBTP keeps one stream node per `(PIM PC, role)` and a four-way outgoing-edge
set per `(stream, source loop context)`. An edge points to the next loop context
and stores the observed signed byte change of the instruction-carried base.
On a predicted taken boundary, the new context selects a matching edge from
each live A/B/C stream's previous context. Confidence, occurrence count, and
timeliness feedback select a single edge. Prediction is suppressed when
confidence is below 2 or two different byte deltas are within the score margin.

The predictor adds the selected byte delta to the last observed A/B/C base, but
sends a translation request only when the result crosses a 4-KiB page. It then
checks the STLB, MSHR/fill queues, its 64-entry pending CAM, and the 32-entry
STLB PQ before issuing one page-aligned request.

`GEMM_RUNTIME_LOOP_CONTEXT=0` disables loop-PC lookup and uses a base-triggered
PC+role control; `=1` enables branch-triggered full dynamic-graph prediction.
Both modes use the same predictor storage and the same trace.

Results are written under `results/pim_g_lbtp_<time>/`:

- `summary.txt`: IPC, cycles, TLB misses, graph occupancy/feedback, A/B/C
  coverage, ambiguity, and timeliness.
- `g_lbtp_pc_role.txt` and `g_lbtp_graph.txt`: raw ChampSim counters.
- `*.prefetch-events.csv`: source/target loop contexts and PCs, signed byte
  delta, edge confidence/score, predicted VPN, and timely/late/early outcome.

See `docs/g_lbtp.md` for the graph definition, microarchitecture, execution
mechanism, and storage estimate.
