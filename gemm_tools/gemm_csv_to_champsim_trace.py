#!/usr/bin/env python3
"""Convert gemm_realamx AMX/PIM tile-base CSV traces to ChampSim traces.

The generated trace is deliberately synthetic: each GEMM trace record is
expanded into three memory instructions so that ChampSim can evaluate DTLB
behavior for A/B/C separately. The role is encoded into IP:

    ip = base_pc + (micro_pc << role_bits) + role

role 0 = A load, role 1 = B load, role 2 = C store by default.
"""

import argparse
import csv
import lzma
import os
import struct
import sys
from typing import BinaryIO, Iterable


INSTR_STRUCT = struct.Struct("<QBB2B4B2Q4Q")


def parse_int(value: str) -> int:
    value = value.strip()
    if not value:
        return 0
    return int(value, 0)


def pack_instr(ip: int, *, src_addr: int = 0, dst_addr: int = 0) -> bytes:
    return INSTR_STRUCT.pack(
        ip,
        0,  # is_branch
        0,  # branch_taken
        0,
        0,  # destination_registers[2]
        0,
        0,
        0,
        0,  # source_registers[4]
        dst_addr,
        0,  # destination_memory[2]
        src_addr,
        0,
        0,
        0,  # source_memory[4]
    )


def open_output(path: str) -> BinaryIO:
    if path.endswith(".xz"):
        return lzma.open(path, "wb", preset=6)
    return open(path, "wb")


def choose_micro_pc(row: dict, row_index: int, thread_seq_column: str, pc_period: int) -> int:
    if thread_seq_column and row.get(thread_seq_column):
        return parse_int(row[thread_seq_column]) % pc_period
    return row_index % pc_period


def convert_rows(args: argparse.Namespace, rows: Iterable[dict], out: BinaryIO) -> tuple[int, int]:
    input_records = 0
    output_instrs = 0
    role_bits = 2

    for row_index, row in enumerate(rows):
        if row_index < args.skip_records:
            continue
        if args.max_records is not None and input_records >= args.max_records:
            break

        micro_pc = choose_micro_pc(row, row_index, args.thread_seq_column, args.pc_period)
        ip_base = args.base_pc + (micro_pc << role_bits)

        a_addr = parse_int(row[args.a_column])
        b_addr = parse_int(row[args.b_column])
        c_addr = parse_int(row[args.c_column])

        out.write(pack_instr(ip_base | 0, src_addr=a_addr))
        out.write(pack_instr(ip_base | 1, src_addr=b_addr))
        if args.c_as_load:
            out.write(pack_instr(ip_base | 2, src_addr=c_addr))
        else:
            out.write(pack_instr(ip_base | 2, dst_addr=c_addr))

        input_records += 1
        output_instrs += 3

    return input_records, output_instrs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_trace", help="Input gemm_realamx CSV trace.")
    parser.add_argument("output_trace", help="Output ChampSim trace; use .xz for xz compression.")
    parser.add_argument("--a-column", default="a_tile_base")
    parser.add_argument("--b-column", default="b_tile_base")
    parser.add_argument("--c-column", default="c_tile_base")
    parser.add_argument("--thread-seq-column", default="thread_seq")
    parser.add_argument("--base-pc", type=lambda x: int(x, 0), default=0x400000)
    parser.add_argument("--pc-period", type=int, default=4, help="Number of static PIM-GEMM PCs in the microkernel loop.")
    parser.add_argument("--skip-records", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--c-as-load", action="store_true", help="Emit C as a load instead of a store.")
    args = parser.parse_args()

    if args.pc_period <= 0:
        raise SystemExit("--pc-period must be positive")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_trace)) or ".", exist_ok=True)

    with open(args.csv_trace, newline="") as csv_file, open_output(args.output_trace) as out:
        reader = csv.DictReader(csv_file)
        required = {args.a_column, args.b_column, args.c_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"missing required CSV columns: {', '.join(sorted(missing))}")
        input_records, output_instrs = convert_rows(args, reader, out)

    print(f"input GEMM records: {input_records}", file=sys.stderr)
    print(f"output ChampSim instructions: {output_instrs}", file=sys.stderr)
    print(f"output trace: {args.output_trace}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
