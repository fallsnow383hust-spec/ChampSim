#!/usr/bin/env python3
"""Summarize side-effect-free STAR-TLB RPRG Acc@1 and graph CRUD logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROLES = ("a", "b", "c")
CONTEXTS = (
    "K_PROGRESS",
    "K_TO_IR",
    "IR_TO_JR",
    "JR_TO_IC",
    "IC_TO_PC",
    "PC_TO_JC",
)


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def pct(numerator: int, denominator: int) -> str:
    return "n/a" if not denominator else f"{100.0 * numerator / denominator:.4f}%"


def wilson(successes: int, samples: int, z: float = 1.96) -> tuple[float, float]:
    if not samples:
        return (0.0, 0.0)
    p = successes / samples
    denominator = 1.0 + z * z / samples
    center = (p + z * z / (2.0 * samples)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / samples + z * z / (4.0 * samples * samples)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def correct(row: dict[str, str], field: str) -> bool:
    return row["source_committed"] == "1" and row[field] == "1"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accuracy_log", type=Path)
    parser.add_argument("graph_log", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    eligible = max(0, int(manifest["input_pim_records"]) - 1)
    offline = manifest.get("offline_ground_truth", {})
    identity_by_pc = {
        int(row["branch_pc"], 0): row
        for row in offline.get("loop_identities", [])
    }
    phase_by_level = {
        "K": "K_PROGRESS",
        "IR": "K_TO_IR",
        "JR": "IR_TO_JR",
        "IC": "JR_TO_IC",
        "PC": "IC_TO_PC",
        "JC": "PC_TO_JC",
    }
    events = read_rows(args.accuracy_log)
    primary_events = [row for row in events if row["primary"] == "1" and row["outcome"] != "out_of_range"]
    primary = [row for row in primary_events if row["actual_context"] != "NONE"]
    committed = [row for row in primary if row["source_committed"] == "1"]
    correct_triplet_vpn = sum(correct(row, "triplet_vpn_correct") and correct(row, "context_correct") for row in primary)
    correct_triplet_byte = sum(correct(row, "triplet_byte_correct") and correct(row, "context_correct") for row in primary)

    graph_rows = read_rows(args.graph_log)
    operations = Counter(row["operation"] for row in graph_rows)
    train_observations = operations["allocate"] + operations["reinforce"]
    lookup_attempts = sum(
        operations[name]
        for name in ("select", "lookup_no_edge", "lookup_low_confidence", "lookup_ambiguous")
    )

    lines = [
        "STAR-TLB side-effect-free RPRG Acc@1 report",
        "runtime contexts: opaque IDs learned from (ASID, branch PC, backward target)",
        f"control-flow source: {manifest.get('control_flow', {}).get('source', 'unknown')}",
        f"paper-valid control flow: {manifest.get('control_flow', {}).get('paper_valid', 'unknown')}",
        f"trace granularity: {manifest.get('translation_granularity', 'unknown')}",
        f"PIM descriptors: {manifest['input_pim_records']}",
        f"eligible one-step transitions: {eligible}",
        "",
        "Direct one-step prediction",
        f"primary predictions/resolved: {len(primary_events)}/{len(primary)}",
        f"unresolved primary predictions: {len(primary_events) - len(primary)}",
        f"target coverage: {pct(len(primary), eligible)}",
        f"source committed: {len(committed)}/{len(primary)} ({pct(len(committed), len(primary))})",
        f"context accuracy: {pct(sum(correct(row, 'context_correct') for row in primary), len(primary))}",
    ]
    for role in ROLES:
        lines.append(f"{role.upper()} byte/VPN accuracy: {pct(sum(correct(row, f'{role}_byte_correct') for row in primary), len(primary))} / "
                     f"{pct(sum(correct(row, f'{role}_vpn_correct') for row in primary), len(primary))}")
    lines += [
        f"triplet byte accuracy: {pct(correct_triplet_byte, len(primary))}",
        f"triplet VPN accuracy: {pct(correct_triplet_vpn, len(primary))}",
        f"correct target coverage (context + triplet VPN): {pct(correct_triplet_vpn, eligible)}",
        "",
        "Observed graph behavior",
        f"train observations: {train_observations}",
        f"allocate/reinforce: {operations['allocate']}/{operations['reinforce']}",
        f"exact-mode recurrence: {pct(operations['reinforce'], train_observations)}",
        f"decay/evict: {operations['decay']}/{operations['evict']}",
        f"lookup attempts/select: {lookup_attempts}/{operations['select']}",
        f"selection coverage: {pct(operations['select'], lookup_attempts)}",
        f"lookup no-edge/low-confidence/ambiguous: {operations['lookup_no_edge']}/"
        f"{operations['lookup_low_confidence']}/{operations['lookup_ambiguous']}",
    ]

    context_rows: list[dict[str, object]] = []
    contexts = sorted({row["actual_context"] for row in primary}, key=int)
    for context in contexts:
        rows = [row for row in primary if row["actual_context"] == context]
        triplet = sum(correct(row, "triplet_vpn_correct") and correct(row, "context_correct") for row in rows)
        low, high = wilson(triplet, len(rows))
        loop_pc_values = [
            int(row["actual_loop_pc"])
            for row in rows
            if int(row["actual_loop_pc"]) != 0
        ]
        loop_pc = Counter(loop_pc_values).most_common(1)[0][0] if loop_pc_values else 0
        identity = identity_by_pc.get(loop_pc, {})
        loop_level = identity.get("loop_level", "")
        ground_truth_phase = phase_by_level.get(loop_level, "")
        denominator = int(
            offline.get("phase_counts", {}).get(ground_truth_phase, 0)
        )
        entry: dict[str, object] = {
            "context_id": int(context),
            "loop_branch_pc": loop_pc,
            "offline_loop_level": loop_level,
            "offline_ground_truth_phase": ground_truth_phase,
            "predictions": len(rows),
            "coverage_denominator": denominator,
            "context_accuracy": ratio(sum(correct(row, "context_correct") for row in rows), len(rows)),
            "triplet_byte_accuracy": ratio(sum(correct(row, "triplet_byte_correct") and correct(row, "context_correct") for row in rows), len(rows)),
            "triplet_vpn_accuracy": ratio(triplet, len(rows)),
            "triplet_vpn_wilson_low": low,
            "triplet_vpn_wilson_high": high,
        }
        for role in ROLES:
            entry[f"{role}_vpn_accuracy"] = ratio(sum(correct(row, f"{role}_vpn_correct") for row in rows), len(rows))
        context_rows.append(entry)

    confidence_rows: list[dict[str, object]] = []
    for confidence in sorted({int(row["edge_confidence"]) for row in primary}):
        rows = [row for row in primary if int(row["edge_confidence"]) == confidence]
        triplet = sum(correct(row, "triplet_vpn_correct") and correct(row, "context_correct") for row in rows)
        low, high = wilson(triplet, len(rows))
        confidence_rows.append({
            "edge_confidence": confidence,
            "predictions": len(rows),
            "triplet_vpn_accuracy": ratio(triplet, len(rows)),
            "wilson_low": low,
            "wilson_high": high,
        })

    convergence_rows: list[dict[str, object]] = []
    ordered = sorted(primary, key=lambda row: int(row["target_seq"]))
    step = 8192
    for boundary in range(step, eligible + step, step):
        rows = [row for row in ordered if int(row["target_seq"]) <= min(boundary, eligible)]
        triplet = sum(correct(row, "triplet_vpn_correct") and correct(row, "context_correct") for row in rows)
        convergence_rows.append({
            "target_seq_upto": min(boundary, eligible),
            "predictions": len(rows),
            "coverage": ratio(len(rows), min(boundary, eligible)),
            "triplet_vpn_accuracy": ratio(triplet, len(rows)),
            "correct_target_coverage": ratio(triplet, min(boundary, eligible)),
        })
        if boundary >= eligible:
            break

    graph_summary_rows = [
        {"operation": operation, "count": count}
        for operation, count in sorted(operations.items())
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = "\n".join(lines) + "\n"
    print(summary, end="")
    (args.output_dir / "rprg-accuracy-summary.txt").write_text(summary, encoding="utf-8")
    if context_rows:
        write_csv(args.output_dir / "rprg-accuracy-by-context.csv", list(context_rows[0]), context_rows)
    if confidence_rows:
        write_csv(args.output_dir / "rprg-accuracy-by-confidence.csv", list(confidence_rows[0]), confidence_rows)
    if convergence_rows:
        write_csv(args.output_dir / "rprg-accuracy-convergence.csv", list(convergence_rows[0]), convergence_rows)
    write_csv(args.output_dir / "rprg-graph-operation-summary.csv", ["operation", "count"], graph_summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
