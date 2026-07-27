#!/usr/bin/env python3
"""Build a full-page PIM-GEMM trace for STAR-TLB.

Each canonical PIMGEMM CSV record is expanded into one demand translation per
unique 4-KiB page in the strided A/B/C tile footprints. Up to four load pages
or two store pages share one ChampSim instruction. Synthetic conditional
branches reproduce the six-level JC/PC/IC/JR/IR/K loop nest without encoding
loop coordinates in memory PCs.
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


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
REQUIRED_COLUMNS = {
    "a_tile_base",
    "b_tile_base",
    "c_tile_base",
    "a_row_stride_bytes",
    "b_row_stride_bytes",
    "c_row_stride_bytes",
    "valid_m",
    "valid_n",
    "valid_k",
    "flags",
}


@dataclass(frozen=True)
class Iteration:
    jc: int
    pc: int
    ic: int
    jr: int
    ir: int
    k0: int


def parse_int(value: str) -> int:
    return int(value.strip(), 0)


def infer_shape(path: Path) -> tuple[int, int, int] | None:
    match = re.search(
        r"(?:^|[-_])m(\d+)[-_]n(\d+)[-_]k(\d+)(?:[-_.]|$)",
        path.name,
        re.IGNORECASE,
    )
    return tuple(map(int, match.groups())) if match else None


def loop_iterations(args: argparse.Namespace) -> Iterator[Iteration]:
    for jc in range(0, args.n, args.nc):
        for pc in range(0, args.k, args.kc):
            for ic in range(0, args.m, args.mc):
                for jr in range(jc, min(jc + args.nc, args.n), args.nr):
                    for ir in range(
                        ic, min(ic + args.mc, args.m), args.mr
                    ):
                        for k0 in range(
                            pc, min(pc + args.kc, args.k), args.kr
                        ):
                            yield Iteration(jc, pc, ic, jr, ir, k0)


def phase_between(
    previous: Iteration | None, current: Iteration
) -> str:
    if previous is None:
        return "START"
    if current.jc != previous.jc:
        return "PC_TO_JC"
    if current.pc != previous.pc:
        return "IC_TO_PC"
    if current.ic != previous.ic:
        return "JR_TO_IC"
    if current.jr != previous.jr:
        return "IR_TO_JR"
    if current.ir != previous.ir:
        return "K_TO_IR"
    if current.k0 != previous.k0:
        return "K_PROGRESS"
    raise ValueError(f"duplicate reconstructed iteration: {current}")


def reconstructed_phases(args: argparse.Namespace) -> Iterator[str]:
    previous = None
    for current in loop_iterations(args):
        yield phase_between(previous, current)
        previous = current


def branch_chain(phase: str) -> list[tuple[int, bool]]:
    code = PHASES[phase]
    if code == 0:
        return []
    target_level = code - 1
    return [
        (level, level == target_level)
        for level in range(target_level + 1)
    ]


def pack_conditional_branch(ip: int, taken: bool) -> bytes:
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


def pack_memory_batch(
    ip: int, addresses: list[int], *, is_store: bool
) -> bytes:
    capacity = 2 if is_store else 4
    if not addresses or len(addresses) > capacity:
        raise ValueError(
            f"memory batch must contain 1..{capacity} addresses"
        )
    destinations = [0, 0]
    sources = [0, 0, 0, 0]
    target = destinations if is_store else sources
    target[: len(addresses)] = addresses
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
        *destinations,
        *sources,
    )


def page_footprint_addresses(
    base: int,
    rows: int,
    row_width_bytes: int,
    row_stride_bytes: int,
    page_bytes: int,
) -> list[int]:
    """Return one actually touched byte address per unique operand page."""
    if rows <= 0 or row_width_bytes <= 0 or row_stride_bytes <= 0:
        raise ValueError("tile footprint parameters must be positive")
    seen: set[int] = set()
    result: list[int] = []
    for row in range(rows):
        begin = base + row * row_stride_bytes
        end = begin + row_width_bytes - 1
        for vpn in range(begin // page_bytes, end // page_bytes + 1):
            if vpn in seen:
                continue
            seen.add(vpn)
            result.append(
                base if not result else max(begin, vpn * page_bytes)
            )
    return result


def emit_batches(
    output: BinaryIO,
    ip: int,
    addresses: list[int],
    *,
    is_store: bool,
    first_ip: int | None = None,
) -> int:
    capacity = 2 if is_store else 4
    count = 0
    for offset in range(0, len(addresses), capacity):
        batch_ip = first_ip if offset == 0 and first_ip is not None else ip
        output.write(
            pack_memory_batch(
                batch_ip,
                addresses[offset : offset + capacity],
                is_store=is_store,
            )
        )
        count += 1
    return count


def open_output(path: Path) -> BinaryIO:
    if path.suffix == ".xz":
        return lzma.open(path, "wb", preset=6)
    return path.open("wb")


def convert(
    args: argparse.Namespace,
    reader: csv.DictReader,
    output: BinaryIO,
) -> dict:
    phases = iter(reconstructed_phases(args))
    sites: dict[int, int] = {}
    phase_counts = {name: 0 for name in PHASES}
    role_page_demands = {"A": 0, "B": 0, "C": 0}
    records = 0
    memory_instructions = 0
    branch_instructions = 0

    for row_index, row in enumerate(reader):
        if args.max_records and records >= args.max_records:
            break
        try:
            phase = next(phases)
        except StopIteration as exc:
            raise ValueError(
                "CSV has more records than the reconstructed loop nest"
            ) from exc
        if "global_seq" in row and parse_int(row["global_seq"]) != row_index:
            raise ValueError("global_seq must be contiguous from zero")
        phase_counts[phase] += 1

        for level, taken in branch_chain(phase):
            output.write(
                pack_conditional_branch(
                    args.branch_pc_base + level * 0x10, taken
                )
            )
            branch_instructions += 1

        raw_site = parse_int(row[args.pc_column])
        site_id = sites.setdefault(raw_site, len(sites))
        ip_base = args.base_pc + (site_id << CONTEXT_BITS)

        valid_m = parse_int(row["valid_m"])
        valid_n = parse_int(row["valid_n"])
        valid_k = parse_int(row["valid_k"])
        if not (
            1 <= valid_m <= args.mr
            and 1 <= valid_n <= args.nr
            and 1 <= valid_k <= args.kr
        ):
            raise ValueError(
                f"invalid tile dimensions at CSV row {row_index + 2}"
            )

        role_addresses = {
            "A": page_footprint_addresses(
                parse_int(row["a_tile_base"]),
                valid_m,
                valid_k * 2,
                parse_int(row["a_row_stride_bytes"]),
                args.page_bytes,
            ),
            "B": page_footprint_addresses(
                parse_int(row["b_tile_base"]),
                valid_k,
                valid_n * 2,
                parse_int(row["b_row_stride_bytes"]),
                args.page_bytes,
            ),
            "C": page_footprint_addresses(
                parse_int(row["c_tile_base"]),
                valid_m,
                valid_n * 4,
                parse_int(row["c_row_stride_bytes"]),
                args.page_bytes,
            ),
        }
        memory_instructions += emit_batches(
            output,
            ip_base | 0,
            role_addresses["A"],
            is_store=False,
            first_ip=ip_base | args.pimcfg_marker_bit,
        )
        memory_instructions += emit_batches(
            output, ip_base | 1, role_addresses["B"], is_store=False
        )
        memory_instructions += emit_batches(
            output, ip_base | 2, role_addresses["C"], is_store=True
        )
        for role, addresses in role_addresses.items():
            role_page_demands[role] += len(addresses)
        records += 1

    if not args.max_records:
        try:
            next(phases)
        except StopIteration:
            pass
        else:
            raise ValueError(
                "CSV has fewer records than the reconstructed loop nest"
            )

    if records:
        for level in range(len(LOOP_LEVELS)):
            output.write(
                pack_conditional_branch(
                    args.branch_pc_base + level * 0x10, False
                )
            )
            branch_instructions += 1

    return {
        "input_pim_records": records,
        "memory_instructions": memory_instructions,
        "branch_instructions": branch_instructions,
        "simulation_instructions": memory_instructions
        + branch_instructions,
        "translation_demands": sum(role_page_demands.values()),
        "role_page_demands": role_page_demands,
        "static_pim_sites": len(sites),
        "phase_counts": phase_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_trace", type=Path)
    parser.add_argument("output_trace", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pc-column")
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--mc", type=int, default=128)
    parser.add_argument("--nc", type=int, default=256)
    parser.add_argument("--kc", type=int, default=256)
    parser.add_argument("--mr", type=int, default=32)
    parser.add_argument("--nr", type=int, default=32)
    parser.add_argument("--kr", type=int, default=32)
    parser.add_argument("--page-bytes", type=int, default=4096)
    parser.add_argument(
        "--base-pc", type=lambda value: int(value, 0), default=0x400000
    )
    parser.add_argument(
        "--branch-pc-base",
        type=lambda value: int(value, 0),
        default=0x500000,
    )
    parser.add_argument(
        "--pimcfg-marker-bit",
        type=lambda value: int(value, 0),
        default=0x4,
    )
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    inferred = infer_shape(args.csv_trace)
    supplied = (args.m, args.n, args.k)
    if inferred is not None:
        args.m, args.n, args.k = tuple(
            explicit if explicit is not None else detected
            for explicit, detected in zip(supplied, inferred)
        )
    if any(value is None for value in (args.m, args.n, args.k)):
        raise SystemExit(
            "M/N/K cannot be inferred from filename; pass --m, --n, --k"
        )
    blocking = (
        args.mc,
        args.nc,
        args.kc,
        args.mr,
        args.nr,
        args.kr,
    )
    if any(value <= 0 for value in blocking):
        raise SystemExit("blocking parameters must be positive")
    if (
        args.pimcfg_marker_bit <= 0
        or args.pimcfg_marker_bit & ((1 << ROLE_BITS) - 1)
    ):
        raise SystemExit("--pimcfg-marker-bit must not overlap role bits")
    if (
        args.page_bytes <= 0
        or args.page_bytes & (args.page_bytes - 1)
    ):
        raise SystemExit("--page-bytes must be a positive power of two")

    args.output_trace.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or Path(str(args.output_trace) + ".json")
    with args.csv_trace.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or [])
        if args.pc_column is None:
            args.pc_column = next(
                (
                    name
                    for name in ("pim_pc", "pim_site_pc")
                    if name in fields
                ),
                None,
            )
        if args.pc_column is None:
            raise SystemExit("missing CSV column: pim_pc")
        missing = (REQUIRED_COLUMNS | {args.pc_column}) - fields
        if missing:
            raise SystemExit(
                "missing CSV columns: " + ", ".join(sorted(missing))
            )
        with open_output(args.output_trace) as output:
            result = convert(args, reader, output)

    result.update(
        {
            "input": str(args.csv_trace),
            "output": str(args.output_trace),
            "translation_granularity": "full_page_footprint",
            "page_bytes": args.page_bytes,
            "ip_encoding": "[pim_site][role:2]",
            "roles": {"A": 0, "B": 1, "C": 2},
            "phases": PHASES,
            "pimcfg_marker_bit": hex(args.pimcfg_marker_bit),
            "shape": {"m": args.m, "n": args.n, "k": args.k},
            "blocking": {
                "mc": args.mc,
                "nc": args.nc,
                "kc": args.kc,
                "mr": args.mr,
                "nr": args.nr,
                "kr": args.kr,
            },
        }
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
