# Epoch-aligned STAR-TLB v2

STAR-TLB v2 is a PIM-GEMM-aware translation prefetcher. It predicts the
complete `A/B/C` descriptor transition at a loop boundary, expands the
predicted descriptor into 4-KiB operand pages, and schedules page walks against
the predicted first-use deadline.

The v2 change removes a control-descriptor epoch mismatch in v1. A speculative
loop boundary no longer reads a global `last_descriptor` that may lag by
several dynamic PIM instructions. Instead, a PDQ and LBQ explicitly pair the
branch with its nearest older PIM descriptor in program order.

## One-sentence mechanism

Allocate a sequence-tagged PIM descriptor in the PDQ before retirement, place a
predicted backward branch in the LBQ, pair them by dynamic instruction order,
speculatively issue the predicted `A/B/C` translations, and train RPRG only
after the descriptor commits.

## Architectural contract

The environment variable `STAR_TLB_DESCRIPTOR_CSV` supplies the fields that a
real PIM instruction or `PIMCFG` would expose:

- PIM site PC and A/B/C virtual base addresses;
- A/B/C row strides in bytes;
- valid M/N/K tile dimensions and flags.

The binary trace supplies demand translations and dynamic loop branches. The
CSV must be the same file used to generate that trace. It is a simulator
sideband for fields that would be carried by the real ISA; it is not a source
of loop coordinates or future addresses.

## Dynamic sequence alignment

### PDQ: PIM Descriptor Queue

- 32 entries in the current model.
- Allocated when the marked first A-page uop enters the frontend model.
- Stores `descriptor_seq`, dynamic `instr_id`, site, A/B/C bases,
  stride/shape, loop context, and speculative/committed state.
- Retirement validates the same `descriptor_seq`; only that event can train
  RPRG.

### LBQ: Loop Boundary Queue

- 16 entries in the current model.
- A predicted taken backward branch contributes its dynamic `instr_id` and
  target loop context.
- The matcher selects the PDQ entry with the largest instruction ID satisfying
  `descriptor.instr_id < branch.instr_id`.
- If the descriptor has not reached the PDQ yet, the LBQ entry waits.

This relation is the key invariant:

```text
source = arg max D.instr_id, subject to D.instr_id < B.instr_id
```

The predictor therefore uses the descriptor that actually precedes the
dynamic loop boundary, rather than whichever descriptor happened to retire
most recently.

## Prediction and training timing

Prediction and training deliberately occur at different times:

1. Descriptor dispatch allocates a PDQ entry.
2. A predicted backward branch allocates an LBQ entry.
3. The program-order matcher forms a pair.
4. RPRG is read immediately and translation prefetches may be generated.
5. ROB retirement marks the PDQ entry committed.
6. Only committed adjacent descriptors update RPRG edges and confidence.

This preserves most of the pre-retirement lead time while preventing
wrong-path or cross-epoch state from becoming persistent training data.

## RPRG, footprint generation, and scheduling

RPRG trains an atomic three-role edge:

```text
(PIM site, source loop context)
  -> (target loop context, delta A/B/C, next stride/shape)
```

A selected edge predicts a complete descriptor. The footprint generator
enumerates pages for:

- `A = (valid_m rows, valid_k * 2 bytes, lda)`;
- `B = (valid_k rows, valid_n * 2 bytes, ldb)`;
- `C = (valid_m rows, valid_n * 4 bytes, ldc)`.

Resident, in-flight, pending, and duplicate VPNs are filtered. The deadline
scheduler tracks PIM interval and translation latency moving averages and
issues at most one eligible STLB translation prefetch per cycle.

## Branch recovery

The supplied ChampSim trace contains only correct-path instructions. The model
still separates predicted and resolved loop context: a predicted backedge may
trigger an early prefetch, while following correct-path instructions receive
the resolved context.

A real wrong-path-fetch implementation additionally needs a branch checkpoint:

- squash younger LBQ and speculative PDQ entries;
- remove not-yet-issued candidate translations from the squashed epoch;
- do not train RPRG from squashed descriptors;
- already issued page walks cannot be recalled and are handled as ordinary
  prefetch pollution.

## Hardware cost

Approximate packed per-core storage:

- PDQ: about 2.0-2.5 KiB for 32 full three-address descriptors and sequence
  metadata;
- LBQ and checkpoints: about 0.25-0.4 KiB for 16 branches;
- matcher: 32 age comparisons plus a priority encoder, off the DTLB/STLB hit
  path;
- existing 256-edge RPRG: approximately 13-16 KiB, depending on tag and delta
  compression;
- complete predictor: approximately 16-20 KiB per core.

The synchronization fix itself adds roughly 2.4-3.0 KiB per core. Its critical
operation is a queue lookup at a loop boundary, not on every TLB hit.

## New statistics

The final statistics include:

- `pdq_dispatch`, `pdq_commit`, `pdq_occupancy`, and `pdq_stall`;
- `lbq_alloc`, `lbq_pair`, `lbq_unpaired`, and `lbq_drop`;
- `seq_conflict`;
- `committed_base_lag_avg` and `committed_base_lag_max`.

`committed_base_lag` measures how far the correct PDQ source descriptor is
ahead of the descriptor that a v1 global committed-base design could have
used.

## Evaluation

```bash
cd ~/projects/ChampSim
csv=~/projects/pim-gemm-isa-sim/results/.../trace.instruction.csv

BUILD_JOBS="$(nproc)" \
bash gemm_tools/run_star_tlb_eval.sh "$csv"
```

For a functional run:

```bash
MAX_PIM_RECORDS=256 BUILD_JOBS="$(nproc)" \
bash gemm_tools/run_star_tlb_eval.sh "$csv" results/star-v2-smoke
```

The key correctness conditions are:

```text
descriptor_mismatch = 0
seq_conflict = 0
pdq_stall = 0
lbq_unpaired = 0 at the end of a complete run
```
