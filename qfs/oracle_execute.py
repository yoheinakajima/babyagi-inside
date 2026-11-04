"""Diagnostic execution ceiling with human-supplied task lists.

This is not an agent baseline and must never be reported as one.  It removes
create and choose entirely, then rolls model outputs forward through the
human-authored correct tasks.  The result answers only whether execute plus
state carry can solve an item when planning is perfect.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.benchmark import file_sha256, load_suite, score_item
from qfs.ceiling import add_phase_diagnostics
from qfs.protocol_v2 import bounded_execute


ORACLE_TASKS = {
    "literal-cinder-5": ["Append -north", "Append -east", "Append -south",
                         "Append -west", "Append -home"],
    "literal-moss-3": ["Wrap the current value in square brackets",
                       "Uppercase only its letters", "Append an exclamation mark"],
    "literal-small-moon-4": ["Replace the space with an underscore",
                             "Uppercase only the letters", "Prefix ID:", "Append #7"],
    "arithmetic-seven-3": ["Add 5", "Multiply the current result by 3", "Subtract 8"],
    "arithmetic-ten-4": ["Divide the current result by 2", "Square the current result",
                         "Add 11", "Subtract 6"],
    "arithmetic-three-5": ["Multiply the current result by 4", "Add 7",
                           "Multiply the current result by 2", "Subtract 5", "Add 9"],
    "structured-numbers-3": ["Sort in ascending numeric order",
                             "Remove duplicates while preserving order",
                             "Keep only odd numbers"],
    "structured-words-3": ["Sort by length with alphabetical tie-breaking",
                           "Keep only four-letter words", "Uppercase every remaining word"],
    "structured-records-3": ["Sort by descending score then ascending name",
                             "Keep records whose score is at least 4",
                             "Output only names in retained order"],
    "language-day-acrostic-3": ["Draft exactly three dawn-themed lines of three words each",
                                "Revise line initials to spell DAY",
                                "Return only the three finished lines"],
    "language-aster-summary-3": ["Draft a summary using only the supplied Aster facts",
                                  "Remove every fact not explicitly supplied",
                                  "Compress to one sentence of at most 12 words containing Aster, sunlight, and ice"],
    "language-tidetally-announcement-5": ["Draft the TideTally headline",
                                           "Draft the two-sentence body",
                                           "Draft the one-sentence CTA",
                                           "Assemble Headline, Body, and CTA into one announcement",
                                           "Polish the announcement without adding availability or accuracy claims"],
}


def execute_prompt(item, cycle, current, task):
    return (
        f"Original objective:\n{item['objective']}\n\n"
        f"Cycle: {cycle}/{item['cycles']}\n"
        f"Current result before this task: {json.dumps(current, ensure_ascii=False)}\n"
        f"Selected task: {task}\n\n"
        "Execute exactly the selected task and no other objective operation. If the current result "
        "is null, use the objective's exact initial value as input; otherwise the current result is "
        "the sole input. For a text transformation, copy every character not explicitly changed, "
        "including case, spaces, punctuation, and symbols. For arithmetic, apply only the named "
        "operation to the current number. For structured data, preserve valid JSON and change only "
        "what the task names. For drafting or revision, return the actual complete deliverable so "
        "far, not a description. Return the actual new result, not reasoning."
    )


def run_item(model, tok, item, device):
    tasks = ORACLE_TASKS[item["id"]]
    if len(tasks) != item["cycles"]:
        raise ValueError(f"oracle task-count mismatch for {item['id']}")
    current, actual, cycles = None, [], []
    for cycle, task in enumerate(tasks, 1):
        prompt = execute_prompt(item, cycle, current, task)
        execution = bounded_execute(model, tok, prompt, device)
        execution["selected_task"] = task
        report = {"cycle": cycle, "state_in": current,
                  "oracle_task": task, "execute": execution}
        cycles.append(report)
        if not execution["parse_ok"]:
            break
        current = execution["result"]
        actual.append(current)
    trace = {"mode": "oracle_tasks_bounded128", "cycles": cycles,
             "actual_results": actual,
             "parse_ok": len(actual) == item["cycles"]}
    return add_phase_diagnostics(item, trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    suite = load_suite(args.suite)
    if set(ORACLE_TASKS) != {item["id"] for item in suite["items"]}:
        raise ValueError("oracle task map does not exactly match suite IDs")
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for item in suite["items"]:
        trace = run_item(model, tok, item, device)
        score = score_item(item, trace["actual_results"])
        record = {"schema_version": 1, "diagnostic_only": True,
                  "item_id": item["id"], "objective": item["objective"],
                  "oracle_tasks": ORACLE_TASKS[item["id"]],
                  "trace": trace, "score": score}
        path = args.output_dir / f"{item['id']}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        reports.append(record)
        print(json.dumps({"item": item["id"],
                          "pass": score["automated_pass"],
                          "prefixes": sum(c["execute"]["result_prefix_correct"]
                                          for c in trace["cycles"])}), flush=True)
    summary = {"schema_version": 1, "diagnostic_only": True,
               "benchmark": {"id": suite["benchmark_id"],
                             "sha256": file_sha256(args.suite)},
               "model": str(Path(args.base).resolve()), "device": device,
               "items": len(reports),
               "automated_pass": sum(r["score"]["automated_pass"] for r in reports),
               "manual_pending": sum(r["score"]["pass"] is None for r in reports),
               "correct_prefixes": sum(c["execute"]["result_prefix_correct"]
                                       for r in reports for c in r["trace"]["cycles"]),
               "cycles": sum(len(r["trace"]["cycles"]) for r in reports)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
