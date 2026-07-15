#!/usr/bin/env python3
"""Summarize baseline, PC+role, and loop-boundary STLB experiments."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


PHASE_LINE = re.compile(
    r"^loop_boundary_tlb_realfill_v2 phase (?P<phase>\S+) "
    r"role (?P<role>[ABC]) (?P<counters>.*)$",
    re.MULTILINE,
)
PATTERN_LINE = re.compile(
    r"^loop_boundary_tlb_realfill_v2 pattern (?P<pattern>\S+) "
    r"(?P<counters>.*)$",
    re.MULTILINE,
)
COUNTER = re.compile(r"(?P<name>[a-z_]+):(?P<value>[0-9.]+)")


def last_number(pattern: str, text: str, default: str = "n/a") -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else default


def counters(text: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for match in COUNTER.finditer(text):
        raw = match.group("value")
        result[match.group("name")] = float(raw) if "." in raw else int(raw)
    return result


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    phase_rows = []
    for match in PHASE_LINE.finditer(text):
        phase_rows.append(
            {
                "phase": match.group("phase"),
                "role": match.group("role"),
                **counters(match.group("counters")),
            }
        )
    pattern_rows = []
    for match in PATTERN_LINE.finditer(text):
        pattern_rows.append(
            {
                "pattern": match.group("pattern"),
                **counters(match.group("counters")),
            }
        )
    return {
        "path": path,
        "ipc": last_number(r"CPU 0 cumulative IPC:\s*([0-9.]+)", text),
        "cycles": last_number(
            r"CPU 0 cumulative IPC:.*?cycles:\s*([0-9]+)", text
        ),
        "dtlb_miss": last_number(
            r"cpu0->cpu0_DTLB LOAD\s+ACCESS:.*?MISS:\s*([0-9]+)", text
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
        "pf_useless": last_number(
            r"cpu0->cpu0_STLB PREFETCH REQUESTED:.*?USELESS:\s*([0-9]+)",
            text,
        ),
        "phase_rows": phase_rows,
        "pattern_rows": pattern_rows,
    }


def integer(row: dict, name: str) -> int:
    return int(row.get(name, 0))


def ratio(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{100.0 * numerator / denominator:.2f}%"


def aggregate_roles(rows: list[dict]) -> dict[str, dict]:
    result = defaultdict(lambda: defaultdict(float))
    summed = (
        "access",
        "miss",
        "candidate",
        "cross_page",
        "same_page",
        "resident_filter",
        "inflight_filter",
        "pending_filter",
        "issued",
        "rejected",
        "demanded",
        "timely",
        "late",
        "late_completed",
        "nonuseful_hit",
        "too_early",
        "never",
        "unresolved_late",
    )
    for row in rows:
        target = result[row["role"]]
        for key in summed:
            target[key] += integer(row, key)
        target["issue_weight"] += float(row.get("issue_to_demand_avg", 0.0)) * integer(row, "demanded")
        target["lead_weight"] += float(row.get("ready_lead_avg", 0.0)) * integer(row, "timely")
        target["late_weight"] += float(row.get("late_by_avg", 0.0)) * integer(row, "late_completed")
    return result


def weighted(value: dict, sum_name: str, count_name: str) -> float:
    count = value[count_name]
    return 0.0 if count == 0 else value[sum_name] / count


def predictor_lines(name: str, parsed: dict) -> list[str]:
    if not parsed["phase_rows"]:
        return []
    lines = [
        "",
        f"{name}: per-role prediction outcome",
        "role issued timely late late_by never never% too_early issue_to_demand ready_lead resident_filter inflight_filter same_page",
    ]
    for role, value in sorted(aggregate_roles(parsed["phase_rows"]).items()):
        lines.append(
            f"{role:<5}{int(value['issued']):<7}{int(value['timely']):<7}"
            f"{int(value['late']):<5}{weighted(value, 'late_weight', 'late_completed'):<8.2f}"
            f"{int(value['never']):<6}{ratio(int(value['never']), int(value['issued'])):<7}"
            f"{int(value['too_early']):<10}{weighted(value, 'issue_weight', 'demanded'):<16.2f}"
            f"{weighted(value, 'lead_weight', 'timely'):<11.2f}"
            f"{int(value['resident_filter']):<16}{int(value['inflight_filter']):<16}"
            f"{int(value['same_page'])}"
        )
    lines += [
        "",
        f"{name}: prediction source",
        "pattern              candidate cross_page resident inflight pending issued timely late never issue_to_demand ready_lead late_by",
    ]
    for row in parsed["pattern_rows"]:
        lines.append(
            f"{row['pattern']:<21}{integer(row, 'candidate'):<10}"
            f"{integer(row, 'cross_page'):<11}{integer(row, 'resident_filter'):<9}"
            f"{integer(row, 'inflight_filter'):<9}{integer(row, 'pending_filter'):<8}"
            f"{integer(row, 'issued'):<7}{integer(row, 'timely'):<7}"
            f"{integer(row, 'late'):<5}{integer(row, 'never'):<6}"
            f"{float(row.get('issue_to_demand_avg', 0.0)):<16.2f}"
            f"{float(row.get('ready_lead_avg', 0.0)):<11.2f}"
            f"{float(row.get('late_by_avg', 0.0)):.2f}"
        )
    lines += [
        "",
        f"{name}: per-loop-boundary and role",
        "phase       role access miss candidate issued timely late late_by never resident inflight",
    ]
    for row in parsed["phase_rows"]:
        lines.append(
            f"{row['phase']:<12}{row['role']:<5}{integer(row, 'access'):<7}"
            f"{integer(row, 'miss'):<5}{integer(row, 'candidate'):<10}"
            f"{integer(row, 'issued'):<7}{integer(row, 'timely'):<7}"
            f"{integer(row, 'late'):<5}{float(row.get('late_by_avg', 0.0)):<8.2f}"
            f"{integer(row, 'never'):<6}{integer(row, 'resident_filter'):<9}"
            f"{integer(row, 'inflight_filter')}"
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
        ("loop_boundary_v2", parse(args.loop_boundary)),
    ]
    lines = [
        "PIM five-loop STLB prefetch comparison",
        "name                    IPC       cycles       DTLB_miss   STLB_miss   pf_request   pf_useful   pf_useless",
    ]
    for name, values in rows:
        lines.append(
            f"{name:<24}{values['ipc']:<10}{values['cycles']:<13}"
            f"{values['dtlb_miss']:<12}{values['stlb_miss']:<12}"
            f"{values['pf_requested']:<13}{values['pf_useful']:<12}"
            f"{values['pf_useless']}"
        )
    lines += predictor_lines(rows[1][0], rows[1][1])
    lines += predictor_lines(rows[2][0], rows[2][1])
    lines += [
        "",
        "Definitions:",
        "timely: prefetched STLB entry was ready and marked useful when demand arrived.",
        "late: demand missed before the predicted translation became ready.",
        "late_by: fill cycle minus first demand cycle for completed late requests.",
        "never: no later demand for the predicted VPN before the end of this trace.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
