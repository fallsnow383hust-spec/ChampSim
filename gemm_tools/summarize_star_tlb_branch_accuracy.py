#!/usr/bin/env python3
"""Summarize resolved loop-branch prediction errors and correlate them with RPRG context errors."""

from __future__ import annotations

import argparse
import csv
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
    args = parser.parse_args()

    branches = read_rows(args.branch_log)
    accuracy = read_rows(args.accuracy_log)
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

    contexts = sorted({(row["context"], row["branch_pc"]) for row in branches})
    context_rows: list[dict[str, object]] = []
    for context, branch_pc in contexts:
        rows = [row for row in branches if row["context"] == context and row["branch_pc"] == branch_pc]
        row_outcomes = Counter(row["outcome"] for row in rows)
        row_predicted = sum(row["predicted_backedge"] == "1" for row in rows)
        row_actual = sum(row["actual_backedge"] == "1" for row in rows)
        row_exact = row_outcomes["correct"]
        context_rows.append({
            "context": context,
            "branch_pc": branch_pc,
            "events": len(rows),
            "predicted_backedges": row_predicted,
            "actual_backedges": row_actual,
            "correct": row_exact,
            "false_positive": row_outcomes["false_positive"],
            "false_negative": row_outcomes["false_negative"],
            "wrong_target": row_outcomes["wrong_target"],
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
