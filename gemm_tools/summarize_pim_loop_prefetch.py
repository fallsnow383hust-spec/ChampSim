#!/usr/bin/env python3
"""Summarize baseline, PC+role, and G-LBTP dynamic-graph experiments."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


LEGACY_SOURCE_LINE = re.compile(
    r"^loop_boundary_tlb_realfill_v3 phase (?P<phase>\S+) "
    r"role (?P<role>[ABC]) (?P<counters>.*)$",
    re.MULTILINE,
)
LEGACY_PATTERN_LINE = re.compile(
    r"^loop_boundary_tlb_realfill_v3 pattern (?P<pattern>\S+) "
    r"(?P<counters>.*)$",
    re.MULTILINE,
)
G_SOURCE_LINE = re.compile(
    r"^g_lbtp_v1 source (?P<phase>\S+) "
    r"role (?P<role>[ABC]) (?P<counters>.*)$",
    re.MULTILINE,
)
G_GRAPH_LINE = re.compile(
    r"^g_lbtp_v1 graph (?P<counters>.*)$",
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
    source_rows = []
    for expression in (LEGACY_SOURCE_LINE, G_SOURCE_LINE):
        for match in expression.finditer(text):
            source_rows.append(
                {
                    "phase": match.group("phase"),
                    "role": match.group("role"),
                    **counters(match.group("counters")),
                }
            )

    pattern_rows = [
        {
            "pattern": match.group("pattern"),
            **counters(match.group("counters")),
        }
        for match in LEGACY_PATTERN_LINE.finditer(text)
    ]
    graph_matches = list(G_GRAPH_LINE.finditer(text))
    graph_row = (
        counters(graph_matches[-1].group("counters")) if graph_matches else {}
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
        "source_rows": source_rows,
        "pattern_rows": pattern_rows,
        "graph": graph_row,
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
        "graph_lookup",
        "no_edge",
        "low_confidence",
        "ambiguous",
        "selected",
        "candidate",
        "cross_page",
        "same_page",
        "resident_filter",
        "inflight_filter",
        "pending_filter",
        "capacity_filter",
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
        target["issue_weight"] += float(
            row.get("issue_to_demand_avg", 0.0)
        ) * integer(row, "demanded")
        target["lead_weight"] += float(
            row.get("ready_lead_avg", 0.0)
        ) * integer(row, "timely")
        target["late_weight"] += float(
            row.get("late_by_avg", 0.0)
        ) * integer(row, "late_completed")
    return result


def weighted(value: dict, sum_name: str, count_name: str) -> float:
    count = value[count_name]
    return 0.0 if count == 0 else value[sum_name] / count


def predictor_lines(name: str, parsed: dict) -> list[str]:
    if not parsed["source_rows"]:
        return []

    lines = [
        "",
        f"{name}: per-role prediction outcome",
        "role issued timely late late_by never never% too_early "
        "issue_to_demand ready_lead resident inflight same_page ambiguous",
    ]
    for role, value in sorted(
        aggregate_roles(parsed["source_rows"]).items()
    ):
        lines.append(
            f"{role:<5}{int(value['issued']):<7}{int(value['timely']):<7}"
            f"{int(value['late']):<5}"
            f"{weighted(value, 'late_weight', 'late_completed'):<8.2f}"
            f"{int(value['never']):<6}"
            f"{ratio(int(value['never']), int(value['issued'])):<7}"
            f"{int(value['too_early']):<10}"
            f"{weighted(value, 'issue_weight', 'demanded'):<16.2f}"
            f"{weighted(value, 'lead_weight', 'timely'):<11.2f}"
            f"{int(value['resident_filter']):<9}"
            f"{int(value['inflight_filter']):<9}"
            f"{int(value['same_page']):<10}"
            f"{int(value['ambiguous'])}"
        )

    if parsed["graph"]:
        graph = parsed["graph"]
        lines += [
            "",
            f"{name}: dynamic graph state",
            "transitions edge_alloc reinforced evicted valid_edges capacity "
            "stream_collision positive_fb negative_fb stale_fb",
            f"{integer(graph, 'transitions'):<12}"
            f"{integer(graph, 'edge_allocations'):<11}"
            f"{integer(graph, 'edge_reinforcements'):<11}"
            f"{integer(graph, 'edge_evictions'):<8}"
            f"{integer(graph, 'valid_edges'):<12}"
            f"{integer(graph, 'edge_capacity'):<9}"
            f"{integer(graph, 'stream_collisions'):<17}"
            f"{integer(graph, 'positive_feedback'):<12}"
            f"{integer(graph, 'negative_feedback'):<12}"
            f"{integer(graph, 'stale_feedback')}",
        ]

    if parsed["pattern_rows"]:
        lines += [
            "",
            f"{name}: legacy prediction source",
            "pattern candidate cross_page resident inflight pending issued "
            "timely late never",
        ]
        for row in parsed["pattern_rows"]:
            lines.append(
                f"{row['pattern']:<21}{integer(row, 'candidate'):<10}"
                f"{integer(row, 'cross_page'):<11}"
                f"{integer(row, 'resident_filter'):<9}"
                f"{integer(row, 'inflight_filter'):<9}"
                f"{integer(row, 'pending_filter'):<8}"
                f"{integer(row, 'issued'):<7}"
                f"{integer(row, 'timely'):<7}"
                f"{integer(row, 'late'):<5}"
                f"{integer(row, 'never')}"
            )

    lines += [
        "",
        f"{name}: per-graph-source and role",
        "source role access miss lookup no_edge low_conf ambiguous selected "
        "candidate issued timely late never",
    ]
    for row in parsed["source_rows"]:
        lines.append(
            f"{row['phase']:<12}{row['role']:<5}"
            f"{integer(row, 'access'):<7}{integer(row, 'miss'):<5}"
            f"{integer(row, 'graph_lookup'):<7}"
            f"{integer(row, 'no_edge'):<8}"
            f"{integer(row, 'low_confidence'):<9}"
            f"{integer(row, 'ambiguous'):<10}"
            f"{integer(row, 'selected'):<9}"
            f"{integer(row, 'candidate'):<10}"
            f"{integer(row, 'issued'):<7}"
            f"{integer(row, 'timely'):<7}"
            f"{integer(row, 'late'):<5}"
            f"{integer(row, 'never')}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("pc_role", type=Path)
    parser.add_argument("dynamic_graph", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        ("baseline", parse(args.baseline)),
        ("g_lbtp_pc_role", parse(args.pc_role)),
        ("g_lbtp_dynamic_graph", parse(args.dynamic_graph)),
    ]
    lines = [
        "PIM five-loop G-LBTP STLB prefetch comparison",
        "name                    IPC       cycles       DTLB_miss   "
        "STLB_miss   pf_request   pf_useful   pf_useless",
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
        "G-LBTP edge: (PIM PC, role, source loop PC) -> "
        "(target loop PC, signed byte delta).",
        "A prediction is issued only after confidence/margin checks and "
        "a 4-KiB page crossing.",
        "timely: prefetched STLB entry was ready and marked useful when "
        "demand arrived.",
        "late: demand missed before the predicted translation became ready.",
        "late_by: fill cycle minus first demand cycle for completed late "
        "requests.",
        "never: no later demand for the predicted VPN before the end of "
        "this trace.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
