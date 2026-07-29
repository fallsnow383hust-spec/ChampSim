#!/usr/bin/env python3
"""Summarize resolved loop-branch prediction errors and correlate them with RPRG context errors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def pct(numerator: int, denominator: int) -> str:
    return "n/a" if not denominator else f"{100.0 * numerator / denominator:.4f}%"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch_log", type=Path)
    parser.add_argument("accuracy_log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    branches = read_rows(args.branch_log)
    accuracy = read_rows(args.accuracy_log)
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.manifest is not None
        else {}
    )
    offline_identity = {
        (int(row["branch_pc"], 0), int(row["target_pc"], 0)): row
        for row in manifest.get("offline_ground_truth", {}).get("loop_identities", [])
    }
    errors = [row for row in branches if row["outcome"] != "correct"]
    outcomes = Counter(row["outcome"] for row in branches)
    predicted = sum(row["predicted_backedge"] == "1" for row in branches)
    actual = sum(row["actual_backedge"] == "1" for row in branches)
    exact = outcomes["correct"]

    branch_by_instr = {row["branch_instr_id"]: row for row in branches}
    primary = [
        row for row in accuracy
        if row["primary"] == "1" and row["actual_context"] != "NONE" and row["source_committed"] == "1"
    ]
    context_errors = [row for row in primary if row["context_correct"] != "1"]
    correlations: Counter[tuple[str, str, str, str, str]] = Counter()
    for row in context_errors:
        branch = branch_by_instr.get(row["branch_instr_id"])
        branch_outcome = branch["outcome"] if branch is not None else "missing_branch_event"
        branch_pc = branch["branch_pc"] if branch is not None else "0"
        correlations[(
            branch_outcome,
            branch_pc,
            row["predicted_context"],
            row["actual_context"],
            row["outcome"],
        )] += 1

    lines = [
        "STAR-TLB resolved loop-branch prediction audit",
        "runtime contexts: opaque IDs; semantic annotations are offline-only",
        f"control-flow source: {manifest.get('control_flow', {}).get('source', 'unknown')}",
        f"audited branch events: {len(branches)}",
        f"predicted/actual backedges: {predicted}/{actual}",
        f"exact direction+target: {exact}",
        f"false positive / false negative / wrong target: "
        f"{outcomes['false_positive']} / {outcomes['false_negative']} / {outcomes['wrong_target']}",
        f"backedge precision: {pct(exact, predicted)}",
        f"backedge recall: {pct(exact, actual)}",
        "",
        "Correlation with primary RPRG Acc@1",
        f"primary resolved predictions: {len(primary)}",
        f"context mismatches: {len(context_errors)}",
    ]
    linked = Counter()
    for row in context_errors:
        branch = branch_by_instr.get(row["branch_instr_id"])
        linked[branch["outcome"] if branch is not None else "missing_branch_event"] += 1
    for outcome in ("correct", "false_positive", "false_negative", "wrong_target", "missing_branch_event"):
        lines.append(f"context mismatches linked to {outcome}: {linked[outcome]}")

    counters: dict[tuple[int, int, int, int], Counter[str]] = {}

    def bucket(
        context: int, asid: int, pc: int, target: int
    ) -> Counter[str]:
        key = (context, asid, pc, target)
        return counters.setdefault(key, Counter())

    for row in branches:
        asid = int(row["asid"])
        pc = int(row["branch_pc"])
        predicted_context = int(row["predicted_context"])
        actual_context = int(row["actual_context"])
        predicted_target = int(row["predicted_target"])
        actual_target = int(row["actual_target"])
        event_context = actual_context or predicted_context
        event_target = actual_target if actual_context else predicted_target
        if event_context:
            bucket(event_context, asid, pc, event_target)["events"] += 1
        # A correct first encounter is scored against the resolved identity for
        # offline accounting, but that newly allocated context was never made
        # visible to the speculative hardware callback.
        prediction_context = predicted_context or (
            actual_context if row["outcome"] == "correct" else 0
        )
        if row["predicted_backedge"] == "1" and prediction_context:
            bucket(prediction_context, asid, pc, actual_target)["predicted"] += 1
        if row["actual_backedge"] == "1" and actual_context:
            bucket(actual_context, asid, pc, actual_target)["actual"] += 1
        if row["outcome"] == "correct" and actual_context:
            bucket(actual_context, asid, pc, actual_target)["correct"] += 1
        elif row["outcome"] == "false_positive" and predicted_context:
            bucket(predicted_context, asid, pc, predicted_target)["false_positive"] += 1
        elif row["outcome"] == "false_negative" and actual_context:
            bucket(actual_context, asid, pc, actual_target)["false_negative"] += 1
        elif row["outcome"] == "wrong_target" and predicted_context:
            bucket(predicted_context, asid, pc, predicted_target)["wrong_target"] += 1

    context_rows: list[dict[str, object]] = []
    for (context, asid, branch_pc, target_pc), values in sorted(counters.items()):
        identity = offline_identity.get((branch_pc, target_pc), {})
        row_predicted = values["predicted"]
        row_actual = values["actual"]
        row_exact = values["correct"]
        context_rows.append({
            "context_id": context,
            "asid": asid,
            "branch_pc": branch_pc,
            "target_pc": target_pc,
            "offline_loop_level": identity.get("loop_level", ""),
            "events": values["events"],
            "predicted_backedges": row_predicted,
            "actual_backedges": row_actual,
            "correct": row_exact,
            "false_positive": values["false_positive"],
            "false_negative": values["false_negative"],
            "wrong_target": values["wrong_target"],
            "precision": ratio(row_exact, row_predicted),
            "recall": ratio(row_exact, row_actual),
        })

    correlation_rows = [
        {
            "branch_outcome": key[0],
            "branch_pc": key[1],
            "predicted_context": key[2],
            "actual_context": key[3],
            "rprg_outcome": key[4],
            "count": count,
        }
        for key, count in sorted(correlations.items())
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = "\n".join(lines) + "\n"
    print(summary, end="")
    (args.output_dir / "branch-prediction-summary.txt").write_text(summary, encoding="utf-8")
    if context_rows:
        write_csv(args.output_dir / "branch-prediction-by-context.csv", list(context_rows[0]), context_rows)
    if errors:
        write_csv(args.output_dir / "branch-prediction-errors.csv", list(errors[0]), errors)
    else:
        write_csv(args.output_dir / "branch-prediction-errors.csv", list(branches[0]) if branches else ["outcome"], [])
    if correlation_rows:
        write_csv(args.output_dir / "branch-rprg-context-error-correlation.csv", list(correlation_rows[0]), correlation_rows)
    else:
        write_csv(
            args.output_dir / "branch-rprg-context-error-correlation.csv",
            ["branch_outcome", "branch_pc", "predicted_context", "actual_context", "rprg_outcome", "count"],
            [],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
