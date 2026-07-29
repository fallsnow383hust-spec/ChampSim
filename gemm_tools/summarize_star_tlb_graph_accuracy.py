#!/usr/bin/env python3
"""Summarize online RPRG Acc@1 with branch, LBQ, and PDQ deliberately bypassed."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

CONTEXTS = ("K_PROGRESS", "K_TO_IR", "IR_TO_JR", "JR_TO_IC", "IC_TO_PC", "PC_TO_JC")
ROLES = ("a", "b", "c")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pct(numerator: int, denominator: int) -> str:
    return "n/a" if not denominator else f"{100.0 * numerator / denominator:.4f}%"


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def wilson(successes: int, samples: int, z: float = 1.96) -> tuple[float, float]:
    if not samples:
        return (0.0, 0.0)
    p = successes / samples
    denominator = 1.0 + z * z / samples
    center = (p + z * z / (2.0 * samples)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / samples + z * z / (4.0 * samples * samples)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle_log", type=Path)
    parser.add_argument("graph_log", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_eligible = max(0, int(manifest["input_pim_records"]) - 1)
    rows = read_rows(args.oracle_log)
    selected = [row for row in rows if row["outcome"] == "selected"]
    outcomes = Counter(row["outcome"] for row in rows)
    adjacent = sum(int(row["target_seq"]) == int(row["source_seq"]) + 1 for row in rows)
    triplet_vpn = sum(row["triplet_vpn_correct"] == "1" for row in selected)
    triplet_byte = sum(row["triplet_byte_correct"] == "1" for row in selected)
    descriptor_exact = sum(row["descriptor_exact"] == "1" for row in selected)

    graph_rows = read_rows(args.graph_log)
    operations = Counter(row["operation"] for row in graph_rows)
    train_observations = operations["allocate"] + operations["reinforce"]

    lines = [
        "STAR-TLB pure RPRG online Acc@1 report",
        "isolation: true adjacent committed descriptors; actual source/target loop contexts",
        "bypassed: branch prediction, LBQ, PDQ, translation prefetch issue",
        "anti-leakage order: query -> score actual target -> train target",
        f"trace granularity: {manifest.get('translation_granularity', 'unknown')}",
        f"PIM descriptors: {manifest['input_pim_records']}",
        f"expected eligible transitions: {expected_eligible}",
        f"observed eligible transitions: {len(rows)}",
        f"adjacent D_i->D_(i+1): {adjacent}/{len(rows)} ({pct(adjacent, len(rows))})",
        "",
        "Online graph prediction",
        f"selected: {len(selected)}",
        f"coverage: {pct(len(selected), expected_eligible)}",
        f"no-edge/low-confidence/ambiguous/address-overflow: "
        f"{outcomes['no_edge']}/{outcomes['low_confidence']}/{outcomes['ambiguous']}/{outcomes['address_overflow']}",
    ]
    for role in ROLES:
        byte_correct = sum(row[f"{role}_byte_correct"] == "1" for row in selected)
        vpn_correct = sum(row[f"{role}_vpn_correct"] == "1" for row in selected)
        lines.append(f"{role.upper()} byte/VPN Acc@1: {pct(byte_correct, len(selected))} / {pct(vpn_correct, len(selected))}")
    low, high = wilson(triplet_vpn, len(selected))
    lines += [
        f"triplet byte Acc@1: {pct(triplet_byte, len(selected))}",
        f"triplet VPN Acc@1: {pct(triplet_vpn, len(selected))}",
        f"triplet VPN 95% Wilson CI: [{100.0 * low:.4f}%, {100.0 * high:.4f}%]",
        f"descriptor exact Acc@1: {pct(descriptor_exact, len(selected))}",
        f"effective correct coverage: {pct(triplet_vpn, expected_eligible)}",
        "",
        "Online graph training",
        f"train observations: {train_observations}",
        f"allocate/reinforce: {operations['allocate']}/{operations['reinforce']}",
        f"exact-mode recurrence: {pct(operations['reinforce'], train_observations)}",
        f"decay/evict: {operations['decay']}/{operations['evict']}",
    ]

    context_rows: list[dict[str, object]] = []
    for context in CONTEXTS:
        eligible_rows = [row for row in rows if row["target_context"] == context]
        predictions = [row for row in eligible_rows if row["outcome"] == "selected"]
        correct = sum(row["triplet_vpn_correct"] == "1" for row in predictions)
        low, high = wilson(correct, len(predictions))
        context_rows.append({
            "target_context": context,
            "eligible": len(eligible_rows),
            "selected": len(predictions),
            "coverage": ratio(len(predictions), len(eligible_rows)),
            "triplet_vpn_correct": correct,
            "triplet_vpn_accuracy": ratio(correct, len(predictions)),
            "effective_correct_coverage": ratio(correct, len(eligible_rows)),
            "wilson_low": low,
            "wilson_high": high,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = "\n".join(lines) + "\n"
    print(summary, end="")
    (args.output_dir / "rprg-oracle-accuracy-summary.txt").write_text(summary, encoding="utf-8")
    write_csv(args.output_dir / "rprg-oracle-accuracy-by-context.csv", list(context_rows[0]), context_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
