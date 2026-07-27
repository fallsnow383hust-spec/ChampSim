# STAR-TLB v1

STAR-TLB is a PIM-GEMM-aware translation prefetcher for the ChampSim fork in
this repository. It predicts an entire `A/B/C` descriptor transition at a
detected loop boundary, expands that descriptor into 4-KiB operand pages, and
schedules page walks against the predicted first-use deadline.

## Architectural contract

The PIM instruction trace carries ordinary memory accesses. The environment
variable `STAR_TLB_DESCRIPTOR_CSV` supplies the architecturally visible PIM
descriptor stream (equivalent to a `PIMCFG` sideband):

- PIM site PC and A/B/C virtual base addresses;
- A/B/C row strides in bytes;
- valid M/N/K tile dimensions and flags.

No loop coordinate is supplied to the prefetcher. Synthetic loop branches are
observed by the existing runtime loop detector and mapped to six small loop
contexts. The first A-page uop of every descriptor carries a reserved
`PIMCFG-start` bit in its synthetic trace PC. The CPU frontend observes this
bit in program order and notifies RPRG; therefore out-of-order
TLB requests cannot desynchronize the descriptor stream. This bit carries no
loop coordinate and no dynamic address—the actual descriptor fields still
come from the architectural sideband. The descriptor CSV must be the same CSV
used to generate the binary trace.

## Data path

1. **Descriptor observer** consumes exactly one descriptor when the marked
   first A-page uop reaches the frontend.
2. **RPRG** stores four weighted transitions per set. An edge is
   `(site, source-loop-context) -> (target-context, delta A/B/C, next shape)`.
   Confidence, frequency, usefulness, and LRU jointly select an edge.
3. **Page-footprint generator** enumerates the exact pages touched by the
   strided tile:
   `A=(valid_m, valid_k*2, lda)`,
   `B=(valid_k, valid_n*2, ldb)`, and
   `C=(valid_m, valid_n*4, ldc)`.
4. **Reuse filter** merges equal VPNs across roles and future descriptors and
   rejects translations already resident or in flight.
5. **Deadline scheduler** tracks moving averages for the PIM interval and page
   walk latency. It uses
   `lookahead = clamp(ceil(walk_latency / PIM_interval)+1, 2, 4)` and issues at
   most one eligible page per cycle, prioritizing the earliest deadline,
   higher reuse, and stronger edge.
6. **Feedback** marks a prefetch timely, late, redundant, too early, or unused
   and updates both the walk-latency estimate and edge usefulness.

The v1 implementation installs completed translations in the STLB using
ChampSim's standard prefetch path. A dedicated optimistic translation buffer
is intentionally left for a later version because it requires a cache-core
lookup hook, not only a prefetcher module.

## Evaluation

```bash
cd ~/projects/ChampSim
csv=~/projects/pim-gemm-isa-sim/results/.../trace.instruction.csv

BUILD_JOBS="$(nproc)" \
bash gemm_tools/run_star_tlb_eval.sh "$csv"
```

For a short functional run:

```bash
MAX_PIM_RECORDS=256 BUILD_JOBS="$(nproc)" \
bash gemm_tools/run_star_tlb_eval.sh "$csv" results/star-smoke
```

The script generates a full-page trace once and runs the baseline and
STAR-TLB binaries against exactly that trace. It writes aggregate results to
`summary.txt` and per-prefetch deadlines/outcomes to
`star_tlb.prefetch-events.csv`.
