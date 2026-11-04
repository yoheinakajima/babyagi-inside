"""Measure plain-Qwen and host-side BabyAGI ceilings on a frozen suite.

Four greedy baselines are supported:

* ``plain_off``: one chat turn, thinking disabled;
* ``plain_on``: one chat turn, thinking enabled;
* ``loop_off``: a Python create/choose/execute loop, thinking disabled; and
* ``loop_on``: the same loop with thinking enabled.

The host loop deliberately counterbalances candidate order.  The create call
labels one candidate as the next task and one as a distractor, the host removes
those labels and swaps positions deterministically, and a separate model call
must choose.  This gives choose-phase ground truth without using benchmark
answers or a second judge model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.benchmark import file_sha256, load_suite, score_item


MODES = ("plain_off", "plain_on", "loop_off", "loop_on")


def _json_values(text):
    """Yield JSON values embedded in prose or Markdown fences."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        yield value


def parse_plain(text, cycles):
    objects = [value for value in _json_values(text) if isinstance(value, dict)]
    for value in reversed(objects):
        results = value.get("results")
        if isinstance(results, list) and len(results) == cycles:
            return results
    arrays = [value for value in _json_values(text)
              if isinstance(value, list) and len(value) == cycles]
    return arrays[-1] if arrays else None


def parse_create(text):
    objects = [value for value in _json_values(text) if isinstance(value, dict)]
    for value in reversed(objects):
        next_task = value.get("next_task")
        distractor = value.get("distractor_task")
        if isinstance(next_task, str) and next_task.strip() \
                and isinstance(distractor, str) and distractor.strip():
            return {"next_task": next_task.strip(), "distractor_task": distractor.strip()}
    return None


def parse_choice(text):
    marked = re.findall(r"(?i)\bCHOOSE\s*:\s*([12])\b", text)
    if marked:
        return int(marked[-1])
    standalone = re.findall(r"(?<!\d)([12])(?!\d)", text)
    return int(standalone[-1]) if standalone else None


def parse_result(text):
    objects = [value for value in _json_values(text) if isinstance(value, dict)]
    for value in reversed(objects):
        if "result" in value and value["result"] is not None:
            return value["result"]
    return None


def counterbalanced_position(item_index, cycle):
    """Put the proposed next task in position 2 for half of all cycles."""
    return 2 if (item_index + cycle) % 2 == 0 else 1


@torch.inference_mode()
def generate_chat(model, tok, prompt, device, thinking, max_new_tokens):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=thinking, return_dict=False)
    begin = time.monotonic()
    output = model.generate(
        torch.tensor([ids], device=device), max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tok.eos_token_id)
    elapsed = time.monotonic() - begin
    generated = output[0, len(ids):].tolist()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    return {
        "text": tok.decode(generated, skip_special_tokens=False),
        "tokens": generated,
        "token_count": len(generated),
        "hit_limit": len(generated) >= max_new_tokens,
        "ended": bool(generated and generated[-1] in eos),
        "seconds": elapsed,
    }


def run_plain(model, tok, item, device, thinking, max_new_tokens):
    prompt = (
        item["objective"] + "\n\nSolve the entire objective in one response. Preserve the "
        "result after every successive operation, not just the final answer. Do not include the "
        "initial state as a result. Each entry must be the complete result after that operation: "
        "a JSON number for numeric work, the requested JSON value for structured work, or a JSON "
        "string for text (use `\\n` inside multiline strings). End with `FINAL_JSON: ` followed by "
        f"exactly one valid JSON object shaped `{{\"results\":[...]}}`, whose `results` array has "
        f"exactly {item['cycles']} entries in task order. Do not put text after the object."
    )
    generation = generate_chat(model, tok, prompt, device, thinking, max_new_tokens)
    actual = parse_plain(generation["text"], item["cycles"])
    return {"prompt": prompt, "thinking": thinking, "generation": generation,
            "parse_ok": actual is not None, "actual_results": actual or []}


def _create_prompt(item, cycle, current, completed):
    return (
        f"Original objective:\n{item['objective']}\n\n"
        f"Cycle: {cycle}/{item['cycles']}\n"
        f"Current result: {json.dumps(current, ensure_ascii=False)}\n"
        f"Completed selected tasks: {json.dumps(completed, ensure_ascii=False)}\n\n"
        "Create two distinct candidate tasks. `next_task` must be the single best next undone "
        "task that advances the objective from the current result. `distractor_task` must be "
        "plausible but not the next task: already completed, out of order, or verification-only. "
        "Do not execute either task. End with `CREATE_JSON: ` followed by exactly one JSON object "
        "with string fields `next_task` and `distractor_task`. Put nothing after the object."
    )


def _choose_prompt(item, cycle, current, candidates):
    return (
        f"Original objective:\n{item['objective']}\n\n"
        f"Cycle: {cycle}/{item['cycles']}\n"
        f"Current result: {json.dumps(current, ensure_ascii=False)}\n"
        f"Task 1: {candidates[0]}\nTask 2: {candidates[1]}\n\n"
        "Choose the one task that is the next undone step and advances the objective now. Reject "
        "completed, out-of-order, or verification-only work. End with exactly `CHOOSE: 1` or "
        "`CHOOSE: 2`. Put nothing after that line."
    )


def _execute_prompt(item, cycle, current, selected):
    return (
        f"Original objective:\n{item['objective']}\n\n"
        f"Cycle: {cycle}/{item['cycles']}\n"
        f"Current result before this task: {json.dumps(current, ensure_ascii=False)}\n"
        f"Selected task: {selected}\n\n"
        "Execute exactly the selected task. Return the actual new result, not a task description, "
        "promise, or explanation. Preserve the requested JSON, number, line breaks, and other "
        "formatting inside the result. End with `RESULT_JSON: ` followed by exactly one valid JSON "
        "object with one field named `result`. Put nothing after the object."
    )


def run_loop(model, tok, item, item_index, device, thinking, max_new_tokens):
    current, completed, actual, cycles = None, [], [], []
    stopped = None
    for cycle in range(1, item["cycles"] + 1):
        create_prompt = _create_prompt(item, cycle, current, completed)
        create = generate_chat(model, tok, create_prompt, device, thinking, max_new_tokens)
        created = parse_create(create["text"])
        report = {"cycle": cycle, "state_in": current,
                  "create": {"prompt": create_prompt, "generation": create,
                             "parsed": created, "parse_ok": created is not None}}
        if created is None:
            stopped = {"cycle": cycle, "phase": "create", "reason": "parse_failed"}
            cycles.append(report)
            break

        expected_choice = counterbalanced_position(item_index, cycle)
        if expected_choice == 1:
            candidates = [created["next_task"], created["distractor_task"]]
        else:
            candidates = [created["distractor_task"], created["next_task"]]
        create_distinct = candidates[0].strip().lower() != candidates[1].strip().lower()
        report["create"].update({"candidates": candidates,
                                 "candidates_distinct": create_distinct})

        choose_prompt = _choose_prompt(item, cycle, current, candidates)
        choose = generate_chat(model, tok, choose_prompt, device, thinking, max_new_tokens)
        selected_index = parse_choice(choose["text"])
        report["choose"] = {"prompt": choose_prompt, "generation": choose,
                            "selected": selected_index, "expected": expected_choice,
                            "parse_ok": selected_index in (1, 2),
                            "correct": selected_index == expected_choice}
        if selected_index not in (1, 2):
            stopped = {"cycle": cycle, "phase": "choose", "reason": "parse_failed"}
            cycles.append(report)
            break

        selected = candidates[selected_index - 1]
        execute_prompt = _execute_prompt(item, cycle, current, selected)
        execute = generate_chat(model, tok, execute_prompt, device, thinking, max_new_tokens)
        result = parse_result(execute["text"])
        report["execute"] = {"prompt": execute_prompt, "generation": execute,
                             "selected_task": selected, "result": result,
                             "parse_ok": result is not None}
        cycles.append(report)
        if result is None:
            stopped = {"cycle": cycle, "phase": "execute", "reason": "parse_failed"}
            break
        current = result
        completed.append(selected)
        actual.append(result)
    return {"thinking": thinking, "cycles": cycles, "stopped": stopped,
            "actual_results": actual}


def add_phase_diagnostics(item, trace):
    """Attach deterministic structural and result-correctness diagnostics."""
    actual = trace["actual_results"]
    for report in trace.get("cycles", []):
        cycle = report["cycle"]
        prefix = actual[:cycle]
        cycle_score = score_item({**item, "cycles": cycle,
                                  "rubric": ({**item["rubric"],
                                              "expected_results": item["rubric"][
                                                  "expected_results"][:cycle]}
                                             if item["rubric"]["type"] == "exact_state_chain"
                                             else item["rubric"])}, prefix)
        execute = report.get("execute")
        if execute is not None:
            execute["result_prefix_correct"] = cycle_score["automated_pass"]
            execute["state_carried_semantically"] = cycle_score["automated_pass"]
    return trace


def summarize_mode(mode, suite, reports):
    phase = {"create_parse_fail": 0, "create_not_distinct": 0,
             "choose_parse_fail": 0, "choose_wrong": 0,
             "execute_parse_fail": 0, "result_prefix_wrong": 0,
             "cycles_attempted": 0, "task2_expected": 0, "task2_selected": 0}
    for report in reports:
        for cycle in report["trace"].get("cycles", []):
            phase["cycles_attempted"] += 1
            create = cycle.get("create", {})
            phase["create_parse_fail"] += not create.get("parse_ok", False)
            phase["create_not_distinct"] += (create.get("parse_ok", False)
                                             and not create.get("candidates_distinct", False))
            choose = cycle.get("choose")
            if choose is not None:
                phase["choose_parse_fail"] += not choose["parse_ok"]
                phase["choose_wrong"] += choose["parse_ok"] and not choose["correct"]
                phase["task2_expected"] += choose["expected"] == 2
                phase["task2_selected"] += choose["selected"] == 2
            execute = cycle.get("execute")
            if execute is not None:
                phase["execute_parse_fail"] += not execute["parse_ok"]
                phase["result_prefix_wrong"] += (execute["parse_ok"]
                                                 and not execute["result_prefix_correct"])
    return {
        "mode": mode,
        "items": len(reports),
        "automated_pass": sum(report["score"]["automated_pass"] for report in reports),
        "manual_pending": sum(report["score"]["pass"] is None for report in reports),
        "parse_pass": sum(report["trace"].get("parse_ok", True) for report in reports),
        "phase_failures": phase if mode.startswith("loop_") else None,
    }


def run_mode(model, tok, suite, mode, output_dir, device, plain_max, loop_max):
    thinking = mode.endswith("_on")
    reports = []
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    for item_index, item in enumerate(suite["items"]):
        if mode.startswith("plain_"):
            trace = run_plain(model, tok, item, device, thinking, plain_max)
            trace["parse_ok"] = bool(trace["parse_ok"])
        else:
            trace = run_loop(model, tok, item, item_index, device, thinking, loop_max)
            trace = add_phase_diagnostics(item, trace)
            trace["parse_ok"] = trace["stopped"] is None
        score = score_item(item, trace["actual_results"])
        record = {"schema_version": 1, "mode": mode, "item_id": item["id"],
                  "category": item["category"], "cycles": item["cycles"],
                  "objective": item["objective"], "trace": trace, "score": score}
        path = mode_dir / f"{item['id']}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        reports.append({"item_id": item["id"], "category": item["category"],
                        "trace_path": str(path.relative_to(output_dir)),
                        "actual_results": trace["actual_results"], "score": score,
                        "trace": trace})
        print(json.dumps({"mode": mode, "item": item["id"],
                          "automated": score["automated_pass"],
                          "parse": trace["parse_ok"]}), flush=True)
    return {"summary": summarize_mode(mode, suite, reports), "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--device")
    parser.add_argument("--plain-max", type=int, default=2048)
    parser.add_argument("--loop-max", type=int, default=1024)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to(device).eval()
    summary_path = args.output_dir / "summary.json"
    modes = {}
    if summary_path.exists():
        prior = json.loads(summary_path.read_text())
        same_suite = prior.get("benchmark", {}).get("sha256") == file_sha256(args.suite)
        same_model = prior.get("model") == str(Path(args.base).resolve())
        if not same_suite or not same_model:
            raise ValueError("existing summary belongs to a different suite or model")
        modes.update(prior.get("modes", {}))
    for mode in args.modes:
        modes[mode] = run_mode(model, tok, suite, mode, args.output_dir, device,
                               args.plain_max, args.loop_max)
        partial = {
            "schema_version": 1,
            "benchmark": {"id": suite["benchmark_id"], "path": str(args.suite),
                          "sha256": file_sha256(args.suite)},
            "model": str(Path(args.base).resolve()), "device": device,
            "decoding": "greedy", "modes": modes,
        }
        summary_path.write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n")
    compact = {name: value["summary"] for name, value in modes.items()}
    print(json.dumps({"output": str(summary_path),
                      "summaries": compact}, indent=2))


if __name__ == "__main__":
    main()
