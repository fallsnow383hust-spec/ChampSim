#!/usr/bin/env python3
"""Summarize baseline, PC+role, and loop-boundary ChampSim outputs."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


MODULE_ROW = re.compile(
    r"loop_boundary_tlb_realfill phase (?P<phase>\S+) role (?P<role>[ABC]) "
    r"access:(?P<access>\d+) miss:(?P<miss>\d+) trained:(?P<trained>\d+) "
    r"prediction:(?P<prediction>\d+) issued:(?P<issued>\d+) "
    r"rejected:(?P<rejected>\d+) pending_hit:(?P<pending_hit>\d+) "
    r"real_useful:(?P<real_useful>\d+) lead_avg:(?P<lead_avg>[0-9.]+) "
    r"lead_max:(?P<lead_max>\d+)"
)


def last_number(pattern: str, text: str, default: str = "n/a") -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else default


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    phase_role = []
    for match in MODULE_ROW.finditer(text):
        row = match.groupdict()
        for key in (
            "access",
            "miss",
            "trained",
            "prediction",
            "issued",
            "rejected",
            "pending_hit",
            "real_useful",
            "lead_max",
        ):
            row[key] = int(row[key])
        row["lead_avg"] = float(row["lead_avg"])
        phase_role.append(row)
    return {
        "path": path,
        "ipc": last_number(r"CPU 0 cumulative IPC:\s*([0-9.]+)", text),
        "cycles": last_number(
            r"CPU 0 cumulative IPC:.*?cycles:\s*([0-9]+)", text
        ),
        "stlb_miss": last_number(
            r"cpu0->cpu0_STLB LOAD\s+ACCESS:.*?MISS:\s*([0-9]+)", text
        ),
        "pf_requested": last_number(
            r"cpu0->cpu0_STLB PREFETCH REQUESTED:\s*([0-9]+)", text
        ),
        "pf_useful": last_number(
            r"cpu0->cpu0_STLB PREFETCH REQUESTED:.*?USEFUL:\s*([0-9]+)",
            text,
        ),
        "phase_role": phase_role,
    }


def ratio(numerator: int, denominator: int) -> str:
    if not denominator:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}%"


def aggregate_role(rows: list[dict]) -> dict[str, dict]:
    result = defaultdict(
        lambda: {
            "access": 0,
            "miss": 0,
            "prediction": 0,
            "issued": 0,
            "rejected": 0,
            "pending_hit": 0,
            "real_useful": 0,
            "lead_weighted": 0.0,
            "lead_max": 0,
        }
    )
    for row in rows:
        target = result[row["role"]]
        for key in (
            "access",
            "miss",
            "prediction",
            "issued",
            "rejected",
            "pending_hit",
            "real_useful",
        ):
            target[key] += row[key]
        target["lead_weighted"] += row["lead_avg"] * row["pending_hit"]
        target["lead_max"] = max(target["lead_max"], row["lead_max"])
    return result


def detail_lines(name: str, parsed: dict) -> list[str]:
    rows = parsed["phase_role"]
    if not rows:
        return []
    lines = [
        "",
        f"{name}: per-role real STLB prefetch behavior",
        "role  access    residual_miss issued  useful  coverage* accuracy  timely/pending lead_avg",
    ]
    for role, value in sorted(aggregate_role(rows).items()):
        useful = value["real_useful"]
        potential = value["miss"] + useful
        lead_avg = (
            value["lead_weighted"] / value["pending_hit"]
            if value["pending_hit"]
            else 0.0
        )
        lines.append(
            f"{role:<6}{value['access']:<10}{value['miss']:<14}"
            f"{value['issued']:<8}{useful:<8}{ratio(useful, potential):<10}"
            f"{ratio(useful, value['issued']):<10}"
            f"{ratio(useful, value['pending_hit']):<15}{lead_avg:.2f}"
        )
    lines.extend(
        [
            "* coverage = useful / (residual STLB misses + useful prefetch hits).",
            "  It is a role-local estimate of misses covered by timely real fills.",
            "",
            f"{name}: per-loop-boundary and role",
            "phase       role access   miss   issued useful pending_hit lead_avg lead_max",
        ]
    )
    for row in rows:
        lines.append(
            f"{row['phase']:<12}{row['role']:<5}{row['access']:<9}"
            f"{row['miss']:<7}{row['issued']:<7}{row['real_useful']:<7}"
            f"{row['pending_hit']:<12}{row['lead_avg']:<9.2f}{row['lead_max']}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("pc_role", type=Path)
    parser.add_argument("loop_boundary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        ("baseline", parse(args.baseline)),
        ("pc_role_same_engine", parse(args.pc_role)),
        ("loop_boundary", parse(args.loop_boundary)),
    ]
    lines = [
        "PIM five-loop STLB prefetch comparison",
        "name                    IPC       cycles       STLB_miss   pf_requested   pf_useful",
    ]
    for name, values in rows:
        lines.append(
            f"{name:<24}{values['ipc']:<10}{values['cycles']:<13}"
            f"{values['stlb_miss']:<12}{values['pf_requested']:<15}"
            f"{values['pf_useful']}"
        )
    lines += detail_lines(rows[1][0], rows[1][1])
    lines += detail_lines(rows[2][0], rows[2][1])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
