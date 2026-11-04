"""Probe one-shot JSON task-list creation before changing the agent protocol.

The probe deliberately stops before execution.  It compares a zero-shot
decomposition prompt with two format-only demonstrations so that any apparent
execution gain can later be attributed to the persistent task list rather than
silently changing several phases at once.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.benchmark import file_sha256, load_suite
from qfs.ceiling import _json_values, generate_chat


MODES = ("zero", "kshot2", "ledger")


def parse_plan(text, count):
    for value in reversed(list(_json_values(text))):
        if not isinstance(value, dict):
            continue
        tasks = value.get("tasks")
        if (isinstance(tasks, list) and len(tasks) == count and
                all(isinstance(task, str) and task.strip() for task in tasks)):
            return [task.strip() for task in tasks]
    return None


def plan_prompt(item, mode):
    if mode == "ledger":
        return (
            "Separate the objective's input state from its executable work.\n\n"
            f"Original objective:\n{item['objective']}\n\n"
            "Return one JSON ledger. `initial` is the exact value supplied by Begin with, Start "
            "with, or the initial JSON data; use null if there is no single input value. `initial` "
            "is data and is never a task. `tasks` contains only executable operation clauses. For "
            "an ordered objective, transcribe the clauses introduced by First, Next, Then, and "
            f"Finally into exactly {item['cycles']} short tasks in that order. Do not add an "
            "initialization task, and do not calculate results. If the objective explicitly asks "
            f"you to decide useful work, propose exactly {item['cycles']} atomic deliverable tasks. "
            "End with `PLAN_JSON: ` followed by exactly one object shaped "
            '`{"initial":...,"tasks":["...","..."]}`. Put nothing after the object.'
        )
    examples = ""
    if mode == "kshot2":
        examples = (
            "Two unrelated format examples:\n"
            "Example objective: Begin with ember. First append -cool; then uppercase the letters.\n"
            'Example answer: {"tasks":["Append -cool to ember","Uppercase only the letters in the current result"]}\n'
            "Example objective: Start with 6. First subtract 1; then multiply by 3.\n"
            'Example answer: {"tasks":["Subtract 1 from 6","Multiply the current result by 3"]}\n\n'
            "The examples demonstrate atomicity and JSON shape only. Never copy their values or operations.\n\n"
        )
    return (
        "INITIAL TASK — Create the first task, then the remaining ordered task list.\n\n"
        f"Original objective:\n{item['objective']}\n\n"
        + examples +
        f"Create exactly {item['cycles']} atomic executable tasks in the order they should run. "
        "Each task must perform exactly one operation or one useful deliverable step. Preserve all "
        "literal values, operands, ordering words, and output constraints. The first entry must be "
        "the earliest operation, not a summary of the objective. Do not execute any task. End with "
        '`PLAN_JSON: ` followed by exactly one object shaped {"tasks":["...","..."]}. '
        "Put nothing after the object."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--device")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to(device).eval()
    modes = {}
    for mode in args.modes:
        reports = []
        for item in suite["items"]:
            prompt = plan_prompt(item, mode)
            generation = generate_chat(
                model, tok, prompt, device, thinking=False,
                max_new_tokens=args.max_new_tokens)
            plan = parse_plan(generation["text"], item["cycles"])
            ledgers = [value for value in _json_values(generation["text"])
                       if isinstance(value, dict) and "initial" in value]
            initial = ledgers[-1].get("initial") if ledgers else None
            reports.append({"item_id": item["id"], "objective": item["objective"],
                            "prompt": prompt, "generation": generation,
                            "parse_ok": plan is not None, "initial": initial,
                            "tasks": plan})
            print(json.dumps({"mode": mode, "item": item["id"],
                              "parse_ok": plan is not None, "initial": initial,
                              "tasks": plan}), flush=True)
        modes[mode] = {"parse_pass": sum(row["parse_ok"] for row in reports),
                       "reports": reports}
    result = {"schema_version": 1,
              "benchmark": {"id": suite["benchmark_id"],
                            "sha256": file_sha256(args.suite)},
              "model": str(Path(args.base).resolve()), "device": device,
              "decoding": {"mode": "greedy", "thinking": False,
                           "max_new_tokens": args.max_new_tokens},
              "modes": modes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
