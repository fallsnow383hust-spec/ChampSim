#!/usr/bin/env python3
"""Convert canonical PIM-loop CSV into role- and boundary-aware ChampSim trace.

Each PIM32 record becomes A-load, B-load, and C-store instructions. Conditional
branch records reproduce the loop latches that executed before the current PIM
instruction. The synthetic memory IP encodes only:

    [static PIM site][2-bit operand role]

The STLB predictor must recover loop context causally from the dynamic backedge
stream. GEMM coordinates and phase labels are used only to reconstruct those
branch outcomes; they are never encoded in a memory PC.
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import os
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Iterable


INSTR_STRUCT = struct.Struct("<QBB2B4B2Q4Q")
REG_FLAGS = 25
REG_IP = 26
ROLE_BITS = 2
PHASE_BITS = 3
CONTEXT_BITS = ROLE_BITS + PHASE_BITS

PHASES = {
    "START": 0,
    "K_PROGRESS": 1,
    "K_TO_IR": 2,
    "IR_TO_JR": 3,
    "JR_TO_IC": 4,
    "IC_TO_PC": 5,
    "PC_TO_JC": 6,
}

LOOP_LEVELS = ("K", "IR", "JR", "IC", "PC", "JC")


def parse_int(value: str) -> int:
    return int(value.strip(), 0)


def open_output(path: Path) -> BinaryIO:
    if path.suffix == ".xz":
        return lzma.open(path, "wb", preset=6)
    return path.open("wb")


def pack_memory(ip: int, *, src_addr: int = 0, dst_addr: int = 0) -> bytes:
    return INSTR_STRUCT.pack(
        ip,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        dst_addr,
        0,
        src_addr,
        0,
        0,
        0,
    )


def pack_conditional_branch(ip: int, taken: bool) -> bytes:
    # dst={IP}, src={IP,FLAGS} makes ChampSim classify it as conditional.
    return INSTR_STRUCT.pack(
        ip,
        1,
        int(taken),
        REG_IP,
        0,
        REG_IP,
        REG_FLAGS,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def branch_chain(phase: str) -> list[tuple[int, bool]]:
    """Return loop-latch evaluations immediately before this PIM record."""
    code = PHASES[phase]
    if code == 0:
        return []
    # K_PROGRESS means the K latch was taken. A carry to outer level L means
    # every inner latch was not taken and L was taken.
    target_level = code - 1
    return [
        (level, level == target_level)
        for level in range(target_level + 1)
    ]


def convert_rows(
    args: argparse.Namespace, rows: Iterable[dict], output: BinaryIO
) -> dict:
    sites: dict[int, int] = {}
    phase_counts = {name: 0 for name in PHASES}
    records = 0
    memory_instructions = 0
    branch_instructions = 0

    for row_index, row in enumerate(rows):
        if args.max_records and records >= args.max_records:
            break
        phase = row[args.phase_column]
        if phase not in PHASES:
            raise ValueError(f"unknown loop phase at row {row_index}: {phase}")
        phase_counts[phase] += 1

        if args.emit_branches:
            for level, taken in branch_chain(phase):
                branch_ip = args.branch_pc_base + level * 0x10
                output.write(pack_conditional_branch(branch_ip, taken))
                branch_instructions += 1

        raw_site = parse_int(row[args.pc_column])
        site_id = sites.setdefault(raw_site, len(sites))
        ip_base = args.base_pc + (site_id << CONTEXT_BITS)

        a_addr = parse_int(row[args.a_column])
        b_addr = parse_int(row[args.b_column])
        c_addr = parse_int(row[args.c_column])
        output.write(pack_memory(ip_base | 0, src_addr=a_addr))
        output.write(pack_memory(ip_base | 1, src_addr=b_addr))
        output.write(pack_memory(ip_base | 2, dst_addr=c_addr))
        memory_instructions += 3
        records += 1

    if args.emit_branches and records:
        # Final loop termination has no following PIM record from which it
        # could be inferred. Emit the six not-taken latches explicitly.
        for level in range(len(LOOP_LEVELS)):
            branch_ip = args.branch_pc_base + level * 0x10
            output.write(pack_conditional_branch(branch_ip, False))
            branch_instructions += 1

    return {
        "input_pim_records": records,
        "memory_instructions": memory_instructions,
        "branch_instructions": branch_instructions,
        "simulation_instructions": memory_instructions + branch_instructions,
        "static_pim_sites": len(sites),
        "phase_context_encoded": False,
        "runtime_backedge_context": args.emit_branches,
        "branches_emitted": args.emit_branches,
        "phase_counts": phase_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_trace", type=Path)
    parser.add_argument("output_trace", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pc-column", default="pim_site_pc")
    parser.add_argument("--phase-column", default="loop_boundary_before")
    parser.add_argument("--a-column", default="a_tile_base")
    parser.add_argument("--b-column", default="b_tile_base")
    parser.add_argument("--c-column", default="c_tile_base")
    parser.add_argument("--base-pc", type=lambda x: int(x, 0), default=0x400000)
    parser.add_argument(
        "--branch-pc-base", type=lambda x: int(x, 0), default=0x500000
    )
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--no-phase-context", action="store_true")
    parser.add_argument(
        "--no-branches", dest="emit_branches", action="store_false"
    )
    parser.set_defaults(emit_branches=True)
    args = parser.parse_args()

    args.output_trace.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or Path(str(args.output_trace) + ".json")
    with args.csv_trace.open("r", newline="", encoding="utf-8-sig") as src:
        reader = csv.DictReader(src)
        required = {
            args.pc_column,
            args.phase_column,
            args.a_column,
            args.b_column,
            args.c_column,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit("missing CSV columns: " + ", ".join(sorted(missing)))
        with open_output(args.output_trace) as output:
            result = convert_rows(args, reader, output)

    result.update(
        {
            "input": str(args.csv_trace),
            "output": str(args.output_trace),
            "ip_encoding": "[pim_site][role:2]",
            "roles": {"A": 0, "B": 1, "C": 2},
            "phases": PHASES,
            "branch_levels": list(LOOP_LEVELS),
        }
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
