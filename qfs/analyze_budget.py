"""Analyze a completed gold-step execution-budget curve.

The generator records every transition separately, but transitions from the
same objective are correlated.  This analysis therefore reports ordinary
step accuracy, exact paired McNemar tests, and item-clustered bootstrap
intervals for paired accuracy deltas.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random


def load_records(results_dir, budget):
    directory = Path(results_dir) / f"budget-{budget:04d}"
    records = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json":
            continue
        record = json.loads(path.read_text())
        identity = record["identity"]
        if identity["budget"] != budget:
            raise ValueError(f"budget identity mismatch in {path}")
        key = (identity["item_id"], identity["cycle"])
        if key in records:
            raise ValueError(f"duplicate transition {key} for budget {budget}")
        records[key] = record
    return records


def exact_mcnemar(improvements, regressions):
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    tail = min(improvements, regressions)
    probability = sum(math.comb(discordant, index)
                      for index in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2 * probability)


def _percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def clustered_delta_ci(low, high, samples=10000, seed=0):
    item_ids = sorted({item_id for item_id, _ in low})
    by_item = {}
    for item_id in item_ids:
        keys = [key for key in low if key[0] == item_id]
        by_item[item_id] = (sum(low[key]["correct"] for key in keys),
                            sum(high[key]["correct"] for key in keys), len(keys))
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        low_correct = high_correct = total = 0
        for _ in item_ids:
            item_id = rng.choice(item_ids)
            low_count, high_count, count = by_item[item_id]
            low_correct += low_count
            high_correct += high_count
            total += count
        deltas.append((high_correct - low_correct) / total)
    return [round(_percentile(deltas, 0.025), 6),
            round(_percentile(deltas, 0.975), 6)]


def paired_comparison(low_budget, high_budget, low, high, bootstrap_samples=10000):
    if set(low) != set(high):
        raise ValueError("paired budgets do not contain identical transitions")
    improvements = sum(not low[key]["correct"] and high[key]["correct"] for key in low)
    regressions = sum(low[key]["correct"] and not high[key]["correct"] for key in low)
    low_accuracy = sum(record["correct"] for record in low.values()) / len(low)
    high_accuracy = sum(record["correct"] for record in high.values()) / len(high)
    return {
        "low_budget": low_budget, "high_budget": high_budget,
        "steps": len(low),
        "low_accuracy": round(low_accuracy, 6),
        "high_accuracy": round(high_accuracy, 6),
        "delta": round(high_accuracy - low_accuracy, 6),
        "improvements": improvements, "regressions": regressions,
        "discordant": improvements + regressions,
        "exact_mcnemar_p": exact_mcnemar(improvements, regressions),
        "item_clustered_bootstrap_delta_95": clustered_delta_ci(
            low, high, samples=bootstrap_samples),
        "bootstrap_samples": bootstrap_samples,
    }


def grouped_accuracy(records, field):
    grouped = {}
    for record in records.values():
        key = str(record[field])
        group = grouped.setdefault(key, {"steps": 0, "correct": 0})
        group["steps"] += 1
        group["correct"] += bool(record["correct"])
    for group in grouped.values():
        group["accuracy"] = round(group["correct"] / group["steps"], 6)
    return grouped


def analyze(results_dir, bootstrap_samples=10000):
    results_dir = Path(results_dir)
    summary = json.loads((results_dir / "summary.json").read_text())
    budgets = summary["budgets"]
    records = {budget: load_records(results_dir, budget) for budget in budgets}
    expected = summary["selected_steps"]
    if any(len(rows) != expected for rows in records.values()):
        raise ValueError("one or more budget arms are incomplete")
    keys = set(records[budgets[0]])
    if any(set(rows) != keys for rows in records.values()):
        raise ValueError("budget arms do not share the same transitions")

    per_budget = {}
    for budget in budgets:
        rows = records[budget]
        per_budget[str(budget)] = {
            "by_category": grouped_accuracy(rows, "category"),
            "by_chain_length": grouped_accuracy(rows, "chain_length"),
            "parse_failures": sum(not row["parse_ok"] for row in rows.values()),
        }
    comparisons = []
    for low_budget, high_budget in zip(budgets, budgets[1:]):
        comparisons.append(paired_comparison(
            low_budget, high_budget, records[low_budget], records[high_budget],
            bootstrap_samples))
    if 128 in records and 256 in records:
        selected = paired_comparison(
            128, 256, records[128], records[256], bootstrap_samples)
    else:
        selected = None
    return {
        "schema_version": 1,
        "experiment": summary["experiment"],
        "suite": summary["suite"],
        "runner_sha256": summary["runner_sha256"],
        "budgets": budgets, "steps_per_budget": expected,
        "per_budget": per_budget,
        "adjacent_paired_comparisons": comparisons,
        "selected_128_vs_256": selected,
        "notes": [
            "accuracy is transition-weighted",
            "McNemar uses paired transitions",
            "bootstrap resamples objectives as clusters and retains all transitions",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    report = analyze(args.results_dir, args.bootstrap_samples)
    text = json.dumps(report, indent=2) + "\n"
    output = args.output or args.results_dir / "analysis.json"
    output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
