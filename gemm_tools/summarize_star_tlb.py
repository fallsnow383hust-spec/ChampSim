#!/usr/bin/env python3
"""Summarize a full-page baseline versus STAR-TLB experiment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


COUNTER = re.compile(r"([a-z_]+):([0-9.]+)")
ROLE_LINE = re.compile(
    r"^star_tlb_v1 role (?P<role>[ABC]) (?P<counters>.*)$",
    re.MULTILINE,
)
GRAPH_LINE = re.compile(
    r"^star_tlb_v1 graph (?P<counters>.*)$", re.MULTILINE
)


def counters(text: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name, raw in COUNTER.findall(text):
        result[name] = float(raw) if "." in raw else int(raw)
    return result


def last(pattern: str, text: str, default: str = "n/a") -> str:
    values = re.findall(pattern, text, flags=re.MULTILINE)
    return values[-1] if values else default


def access_misses(cache: str, text: str) -> int:
    expression = re.compile(
        rf"^cpu0->cpu0_{cache} (?:LOAD|WRITE|RFO)\s+"
        rf"ACCESS:.*?MISS:\s*([0-9]+)",
        re.MULTILINE,
    )
    return sum(int(value) for value in expression.findall(text))


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    graph_match = list(GRAPH_LINE.finditer(text))
    return {
        "ipc": last(r"CPU 0 cumulative IPC:\s*([0-9.]+)", text),
        "cycles": last(
            r"CPU 0 cumulative IPC:.*?cycles:\s*([0-9]+)", text
        ),
        "dtlb_miss": access_misses("DTLB", text),
        "stlb_miss": access_misses("STLB", text),
        "pf_requested": last(
            r"cpu0->cpu0_STLB PREFETCH REQUESTED:\s*([0-9]+)", text
        ),
        "pf_useful": last(
            r"cpu0->cpu0_STLB PREFETCH REQUESTED:.*?USEFUL:\s*([0-9]+)",
            text,
        ),
        "pf_useless": last(
            r"cpu0->cpu0_STLB PREFETCH REQUESTED:.*?USELESS:\s*([0-9]+)",
            text,
        ),
        "graph": (
            counters(graph_match[-1].group("counters"))
            if graph_match
            else {}
        ),
        "roles": {
            match.group("role"): counters(match.group("counters"))
            for match in ROLE_LINE.finditer(text)
        },
    }


def integer(row: dict, key: str) -> int:
    return int(row.get(key, 0))


def pct(numerator: int, denominator: int) -> str:
    return (
        "n/a"
        if denominator == 0
        else f"{100.0 * numerator / denominator:.2f}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("star", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = parse(args.baseline)
    star = parse(args.star)
    lines = [
        "STAR-TLB full-page translation comparison",
        "name       IPC       cycles       DTLB_miss   STLB_miss   "
        "PF_req   PF_useful PF_useless",
        f"baseline   {baseline['ipc']:<9}{baseline['cycles']:<13}"
        f"{baseline['dtlb_miss']:<12}{baseline['stlb_miss']:<12}"
        f"{baseline['pf_requested']:<9}{baseline['pf_useful']:<10}"
        f"{baseline['pf_useless']}",
        f"star_tlb   {star['ipc']:<9}{star['cycles']:<13}"
        f"{star['dtlb_miss']:<12}{star['stlb_miss']:<12}"
        f"{star['pf_requested']:<9}{star['pf_useful']:<10}"
        f"{star['pf_useless']}",
    ]

    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        lines += [
            "",
            "Trace coverage",
            f"PIM descriptors: {manifest['input_pim_records']}",
            f"translation demands: {manifest['translation_demands']}",
            "role pages: "
            + " ".join(
                f"{role}={count}"
                for role, count in manifest["role_page_demands"].items()
            ),
        ]

    graph = star["graph"]
    if graph:
        lines += [
            "",
            "RPRG",
            f"descriptors seen/loaded: "
            f"{integer(graph, 'descriptors_seen')}/"
            f"{integer(graph, 'descriptors_loaded')}",
            f"edges valid/capacity: {integer(graph, 'valid_edges')}/"
            f"{integer(graph, 'edge_capacity')}",
            f"boundary triggers: {integer(graph, 'boundary_triggers')}",
            f"predicted descriptors: "
            f"{integer(graph, 'predicted_descriptors')}",
            f"no-edge/low-confidence/ambiguous: "
            f"{integer(graph, 'no_edge')}/"
            f"{integer(graph, 'low_confidence')}/"
            f"{integer(graph, 'ambiguous')}",
            f"adaptive interval/walk/lookahead: "
            f"{integer(graph, 'interval_ema')}/"
            f"{integer(graph, 'walk_latency_ema')}/"
            f"{integer(graph, 'lookahead')}",
        ]

    if star["roles"]:
        lines += [
            "",
            "role issued demanded timely late redundant too_early never "
            "timely/issued timely/demanded demand_coverage",
        ]
        for role in "ABC":
            row = star["roles"].get(role, {})
            issued = integer(row, "issued")
            demanded = integer(row, "demanded")
            timely = integer(row, "timely")
            late = integer(row, "late")
            accesses = integer(row, "access")
            lines.append(
                f"{role:<5}{issued:<7}{demanded:<9}{timely:<7}{late:<5}"
                f"{integer(row, 'redundant'):<10}"
                f"{integer(row, 'too_early'):<10}"
                f"{integer(row, 'never'):<6}"
                f"{pct(timely, issued):<14}"
                f"{pct(timely, demanded):<16}"
                f"{pct(demanded, accesses)}"
            )

    output = "\n".join(lines) + "\n"
    print(output, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
