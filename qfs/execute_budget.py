"""Measure bounded single-step execution accuracy on gold task/state pairs.

This isolates Qwen's execute policy from ledger extraction, selection, and
state carry.  Runs are greedy, checksum-bound, saved one step at a time, and
safe to resume.  The fixed-span controller deliberately prevents natural
``</think>`` termination; this experiment measures correctness versus compiled
budget, not natural reasoning length.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.benchmark import CATEGORIES, file_sha256, load_suite, score_item
from qfs.ceiling import _execute_prompt
from qfs.protocol_v2 import bounded_execute


DEFAULT_BUDGETS = (16, 32, 64, 128, 256)


def _source_sha256():
    sources = [Path(__file__), Path(__file__).with_name("protocol_v2.py"),
               Path(__file__).with_name("ceiling.py")]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _wilson(successes, total, z=1.96):
    if total == 0:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) \
        / denominator
    return [round(center - radius, 6), round(center + radius, 6)]


def gold_entries(suite, max_steps_per_category=None):
    grouped = {category: [] for category in CATEGORIES}
    for item in suite["items"]:
        provenance = item.get("provenance")
        if not isinstance(provenance, dict):
            continue
        steps = provenance.get("gold_steps")
        if not isinstance(steps, list) or len(steps) != item["cycles"]:
            continue
        for step in steps:
            grouped[item["category"]].append({"item": item, "step": step,
                                               "normalization": provenance[
                                                   "normalization"]})
    selected = []
    for category in CATEGORIES:
        entries = grouped[category]
        if max_steps_per_category is not None:
            # First cover every available cycle position.  At each position use
            # the shortest compatible chain, alternating template families.
            # Fill any larger request from the remaining deterministic order.
            by_cycle = {}
            for entry in entries:
                by_cycle.setdefault(entry["step"]["cycle"], []).append(entry)
            pilot = []
            for cycle in sorted(by_cycle):
                candidates = by_cycle[cycle]
                shortest = min(entry["item"]["cycles"] for entry in candidates)
                candidates = sorted(
                    (entry for entry in candidates
                     if entry["item"]["cycles"] == shortest),
                    key=lambda entry: entry["item"]["id"])
                pilot.append(candidates[(cycle - 1) % len(candidates)])
            remaining = sorted(
                (entry for entry in entries if entry not in pilot),
                key=lambda entry: (entry["item"]["cycles"],
                                   entry["step"]["cycle"],
                                   entry["item"]["id"]))
            entries = (pilot + remaining)[:max_steps_per_category]
        selected.extend(entries)
    if not selected:
        raise ValueError("suite has no provenance.gold_steps records")
    return selected


def _step_score(item, step, normalization, result):
    probe = {
        "id": f"{item['id']}-cycle-{step['cycle']}",
        "cycles": 1,
        "rubric": {"type": "exact_state_chain", "normalization": normalization,
                   "expected_results": [step["expected"]]},
    }
    return score_item(probe, [result] if result is not None else [])


def _summarize_budget(budget, reports):
    by_category = {}
    for category in CATEGORIES:
        group = [report for report in reports if report["category"] == category]
        correct = sum(report["correct"] for report in group)
        by_category[category] = {
            "steps": len(group), "correct": correct,
            "accuracy": round(correct / len(group), 6) if group else None,
            "wilson_95": _wilson(correct, len(group)),
            "parse_fail": sum(not report["parse_ok"] for report in group),
        }
    correct = sum(report["correct"] for report in reports)
    return {
        "budget": budget,
        "steps": len(reports),
        "correct": correct,
        "accuracy": round(correct / len(reports), 6) if reports else None,
        "wilson_95": _wilson(correct, len(reports)),
        "parse_fail": sum(not report["parse_ok"] for report in reports),
        "mean_answer_tokens": round(
            sum(report["answer_token_count"] for report in reports) / len(reports), 3)
            if reports else None,
        "by_category": by_category,
    }


def run_curve(base, suite_path, output_dir, budgets=DEFAULT_BUDGETS, device=None,
              answer_tokens=256, max_steps_per_category=None):
    suite_path, output_dir = Path(suite_path), Path(output_dir)
    suite = load_suite(suite_path)
    suite_hash = file_sha256(suite_path)
    runner_hash = _source_sha256()
    entries = gold_entries(suite, max_steps_per_category)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16).to(device).eval()
    summaries = []
    for budget in budgets:
        reports = []
        budget_dir = output_dir / f"budget-{budget:04d}"
        budget_dir.mkdir(parents=True, exist_ok=True)
        for number, entry in enumerate(entries, 1):
            item, step = entry["item"], entry["step"]
            result_path = budget_dir / f"{item['id']}-cycle-{step['cycle']}.json"
            report = None
            if result_path.exists():
                candidate = json.loads(result_path.read_text())
                identity = candidate.get("identity", {})
                if identity == {
                        "suite_sha256": suite_hash, "runner_sha256": runner_hash,
                        "budget": budget, "item_id": item["id"],
                        "cycle": step["cycle"]}:
                    report = candidate
            if report is None:
                prompt = _execute_prompt(item, step["cycle"], step["incoming"],
                                         step["task"])
                execution = bounded_execute(
                    model, tok, prompt, device, think_tokens=budget,
                    answer_tokens=answer_tokens)
                score = _step_score(item, step, entry["normalization"],
                                    execution["result"])
                report = {
                    "schema_version": 1,
                    "identity": {
                        "suite_sha256": suite_hash, "runner_sha256": runner_hash,
                        "budget": budget, "item_id": item["id"],
                        "cycle": step["cycle"],
                    },
                    "category": item["category"],
                    "chain_length": item["cycles"],
                    "incoming": step["incoming"], "task": step["task"],
                    "expected": step["expected"], "result": execution["result"],
                    "parse_ok": execution["parse_ok"],
                    "correct": score["automated_pass"],
                    "think_token_count": len(execution["think_tokens"]),
                    "answer_token_count": len(execution["answer_tokens"]),
                    "think_text": execution["think_text"],
                    "answer_text": execution["answer_text"],
                    "think_token_ids": execution["think_tokens"],
                    "answer_token_ids": execution["answer_tokens"],
                    "forward_seconds": execution["forward_seconds"],
                }
                result_path.write_text(json.dumps(report, indent=2) + "\n")
            reports.append(report)
            print(json.dumps({"budget": budget, "step": number,
                              "steps": len(entries), "item": item["id"],
                              "cycle": step["cycle"], "correct": report["correct"]}),
                  flush=True)
        budget_summary = _summarize_budget(budget, reports)
        summaries.append(budget_summary)
        (budget_dir / "summary.json").write_text(
            json.dumps(budget_summary, indent=2) + "\n")

    ranked = sorted(summaries, key=lambda row: (-row["accuracy"], row["budget"]))
    summary = {
        "schema_version": 1,
        "experiment": "gold-single-step-execution-budget-curve",
        "base": str(Path(base).resolve()), "device": device,
        "suite": {"path": str(suite_path), "id": suite["benchmark_id"],
                  "sha256": suite_hash},
        "runner_sha256": runner_hash,
        "budgets": list(budgets), "answer_tokens": answer_tokens,
        "max_steps_per_category": max_steps_per_category,
        "selected_steps": len(entries),
        "summaries": summaries,
        "recommended_compile_budgets": [row["budget"] for row in ranked[:2]],
        "selection_rule": "highest step accuracy, ties resolved toward smaller budget",
        "limitations": [
            "gold tasks and incoming states isolate execute from ledger extraction",
            "fixed-span control suppresses natural think termination",
            "pilot selections are not a substitute for the complete frozen-suite run",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=list(DEFAULT_BUDGETS))
    parser.add_argument("--device")
    parser.add_argument("--answer-tokens", type=int, default=256)
    parser.add_argument("--max-steps-per-category", type=int)
    args = parser.parse_args()
    if any(budget < 1 for budget in args.budgets):
        parser.error("budgets must be positive")
    summary = run_curve(args.base, args.suite, args.output_dir, args.budgets,
                        args.device, args.answer_tokens,
                        args.max_steps_per_category)
    print(json.dumps({"output": str(args.output_dir / "summary.json"),
                      "recommended": summary["recommended_compile_budgets"]}))


if __name__ == "__main__":
    main()
