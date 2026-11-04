"""Host prototype of a persistent JSON-ledger BabyAGI protocol.

One model call creates ``initial`` plus the complete task queue.  Each cycle
counterbalances the next queued task against a completed or future task,
scores each candidate independently without exposing its slot, executes the
selected task with a bounded reasoning span, and carries the result forward.
The host currently parses and indexes the ledger, so this is protocol evidence,
not compiled-controller evidence.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.benchmark import file_sha256, load_suite, score_item
from qfs.ceiling import (_json_values, add_phase_diagnostics,
                         counterbalanced_position, generate_chat)
from qfs.oracle_execute import execute_prompt
from qfs.plan_probe import plan_prompt
from qfs.protocol_v2 import bounded_execute
from qfs.agent_native import NativeRunner


def parse_ledger(text, count):
    for value in reversed(list(_json_values(text))):
        if not isinstance(value, dict) or "initial" not in value:
            continue
        tasks = value.get("tasks")
        if (isinstance(tasks, list) and len(tasks) == count and
                all(isinstance(task, str) and task.strip() for task in tasks)):
            return {"initial": value["initial"],
                    "tasks": [task.strip() for task in tasks]}
    return None


def candidate_prompt(item, current, completed, candidate):
    return (
        f"Original objective:\n{item['objective']}\n\n"
        f"Current result: {json.dumps(current, ensure_ascii=False)}\n"
        f"Completed tasks: {json.dumps(completed, ensure_ascii=False)}\n"
        f"Candidate task: {candidate}\n\n"
        "Is this candidate the single next undone task that should be executed now? "
        "Treat Begin/Start/initial JSON as input, not work. Reject completed and out-of-order "
        "tasks. Answer exactly Y or N."
    )


@torch.inference_mode()
def score_candidate(model, tok, item, current, completed, candidate, device):
    prompt = candidate_prompt(item, current, completed, candidate)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    begin = time.monotonic()
    output = model(input_ids=torch.tensor([ids], device=device), use_cache=False,
                   logits_to_keep=1)
    yes, no = (tok.encode(value, add_special_tokens=False) for value in ("Y", "N"))
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("Y and N must be single tokens")
    logits = output.logits[0, -1].float()
    y, n = float(logits[yes[0]]), float(logits[no[0]])
    return {"prompt": prompt, "yes_token": yes[0], "no_token": no[0],
            "yes_logit": y, "no_logit": n, "score_yes_minus_no": y - n,
            "binary": "Y" if y >= n else "N",
            "seconds": time.monotonic() - begin}


@torch.inference_mode()
def bounded_score_prompt(model, tok, prompt, device, think_tokens=32):
    """Give an independent candidate judgment a fixed model-owned think span."""
    initial = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=False, enable_thinking=False, return_dict=False)
    prefix = list(initial) + tok.encode(
        "\n<|im_start|>assistant\n<think>\n", add_special_tokens=False)
    runner = NativeRunner(model, tok, device, temperature=0.0, top_p=0.9, seed=0)
    logits = runner.start(prefix)
    reasoning = []
    for _ in range(think_tokens):
        token = int(logits[:248044].argmax())
        reasoning.append(token)
        logits = runner.feed([token])
    suffix = tok.encode("\n</think>\n\nANSWER: ", add_special_tokens=False)
    logits = runner.feed(suffix)
    yes, no = (tok.encode(value, add_special_tokens=False) for value in ("Y", "N"))
    y, n = float(logits[yes[0]]), float(logits[no[0]])
    return {"prompt": prompt, "think_tokens": reasoning,
            "think_text": tok.decode(reasoning), "forced_suffix_tokens": suffix,
            "yes_token": yes[0], "no_token": no[0], "yes_logit": y, "no_logit": n,
            "score_yes_minus_no": y - n, "binary": "Y" if y >= n else "N",
            "forward_seconds": runner.forward_seconds}


def run_item(model, tok, item, item_index, device, plan_max):
    prompt = plan_prompt(item, "ledger")
    generation = generate_chat(model, tok, prompt, device, thinking=False,
                               max_new_tokens=plan_max)
    ledger = parse_ledger(generation["text"], item["cycles"])
    plan = {"prompt": prompt, "generation": generation,
            "parsed": ledger, "parse_ok": ledger is not None}
    if ledger is None:
        return {"mode": "ledger_jev_bounded128", "plan": plan, "cycles": [],
                "stopped": {"phase": "plan", "reason": "parse_failed"},
                "actual_results": [], "parse_ok": False}

    current = ledger["initial"]
    completed, actual, cycles = [], [], []
    for cycle, due in enumerate(ledger["tasks"], 1):
        distractor = (ledger["tasks"][cycle] if cycle < item["cycles"]
                      else ledger["tasks"][0])
        expected = counterbalanced_position(item_index, cycle)
        candidates = ([due, distractor] if expected == 1 else [distractor, due])
        scores = [score_candidate(model, tok, item, current, completed, candidate, device)
                  for candidate in candidates]
        selected = 1 if scores[0]["score_yes_minus_no"] >= scores[1]["score_yes_minus_no"] else 2
        selected_task = candidates[selected - 1]
        execution = bounded_execute(
            model, tok, execute_prompt(item, cycle, current, selected_task), device)
        execution["selected_task"] = selected_task
        report = {"cycle": cycle, "state_in": current,
                  "create": {"source": "persistent_json_ledger",
                             "candidates": candidates,
                             "candidates_distinct": candidates[0] != candidates[1],
                             "parse_ok": True},
                  "choose": {"method": "independent_yes_no_logit_score",
                             "scores": scores, "selected": selected,
                             "expected": expected, "parse_ok": True,
                             "correct": selected == expected},
                  "execute": execution}
        cycles.append(report)
        if not execution["parse_ok"]:
            return add_phase_diagnostics(item, {
                "mode": "ledger_jev_bounded128", "plan": plan, "cycles": cycles,
                "stopped": {"cycle": cycle, "phase": "execute",
                            "reason": "parse_failed"},
                "actual_results": actual, "parse_ok": False})
        current = execution["result"]
        completed.append(selected_task)
        actual.append(current)
    return add_phase_diagnostics(item, {
        "mode": "ledger_jev_bounded128", "plan": plan, "cycles": cycles,
        "stopped": None, "actual_results": actual, "parse_ok": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--plan-max", type=int, default=384)
    args = parser.parse_args()
    suite = load_suite(args.suite)
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, item in enumerate(suite["items"]):
        trace = run_item(model, tok, item, index, device, args.plan_max)
        score = score_item(item, trace["actual_results"])
        record = {"schema_version": 1, "item_id": item["id"],
                  "objective": item["objective"], "trace": trace, "score": score}
        path = args.output_dir / f"{item['id']}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        reports.append(record)
        print(json.dumps({"item": item["id"], "pass": score["automated_pass"],
                          "stopped": trace["stopped"]}), flush=True)
    cycles = [cycle for report in reports for cycle in report["trace"]["cycles"]]
    choices = [cycle["choose"] for cycle in cycles]
    summary = {"schema_version": 1,
               "benchmark": {"id": suite["benchmark_id"],
                             "sha256": file_sha256(args.suite)},
               "model": str(Path(args.base).resolve()), "device": device,
               "mode": "ledger_jev_bounded128", "items": len(reports),
               "automated_pass": sum(r["score"]["automated_pass"] for r in reports),
               "manual_pending": sum(r["score"]["pass"] is None for r in reports),
               "plan_parse_pass": sum(r["trace"]["plan"]["parse_ok"] for r in reports),
               "full_parse_pass": sum(r["trace"]["parse_ok"] for r in reports),
               "cycles": len(cycles),
               "choose_correct": sum(c["correct"] for c in choices),
               "choose_total": len(choices),
               "task2_expected": sum(c["expected"] == 2 for c in choices),
               "task2_selected": sum(c["selected"] == 2 for c in choices),
               "binary_one_yes_one_no": sum(
                   [s["binary"] for s in c["scores"]].count("Y") == 1 for c in choices),
               "correct_prefixes": sum(c["execute"]["result_prefix_correct"]
                                       for c in cycles)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
