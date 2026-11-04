"""Host-side protocol-v2 experiments before compiling new control flow.

Protocol v2 keeps thinking-off candidate creation, replaces side-by-side 1/2
selection with independent JEV-style Y/N scoring, and optionally gives execute
an exact 128-token model-owned reasoning span before forcing the result prefix.
These are diagnostic host baselines; they are not compiled-agent evidence.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.agent_native import NativeRunner
from qfs.benchmark import file_sha256, load_suite, score_item
from qfs.ceiling import (_create_prompt, _execute_prompt, add_phase_diagnostics,
                         counterbalanced_position, generate_chat, parse_create,
                         parse_result)


MODES = ("jev_off", "jev_bounded128", "indexed_jev_bounded128",
         "indexed_kshot_jev_bounded128")


def indexed_create_prompt(item, cycle, current, completed):
    """Make the cycle index authoritative for explicitly ordered objectives.

    The earlier prompt asked the 2B model to infer completion from the current
    value.  That made it skip operations whose effects were not obvious from a
    value such as ``100``.  This prompt tests the protocol repair directly:
    an initial state is input, while First/Next/Then/Finally clauses are tasks.
    """
    return (
        f"Original objective:\n{item['objective']}\n\n"
        f"Cycle: {cycle}/{item['cycles']}\n"
        f"Current result: {json.dumps(current, ensure_ascii=False)}\n"
        f"Completed selected tasks: {json.dumps(completed, ensure_ascii=False)}\n\n"
        "Create two distinct candidate tasks without executing either. If the objective names "
        "ordered operations with words such as First, Next, Then, or Finally, count only those "
        f"operation clauses: `next_task` MUST be operation {cycle} of {item['cycles']}. A phrase "
        "such as Begin with, Start with, or initial JSON supplies the input and is NEVER its own "
        "task. The cycle index is authoritative; do not guess the next operation from the current "
        "value. If the objective instead asks you to decide the useful work, create one new useful "
        "deliverable step not present in `completed`. `distractor_task` must be plausible but wrong "
        "now: an already completed operation or a later operation. Preserve exact operands, text, "
        "and output constraints. End with `CREATE_JSON: ` followed by exactly one JSON object with "
        "string fields `next_task` and `distractor_task`. Put nothing after the object."
    )


def indexed_kshot_create_prompt(item, cycle, current, completed):
    examples = (
        "Unrelated mapping examples (format and cycle mapping only):\n"
        "Example objective: Start with 4. First add 2; next multiply by 3; finally subtract 1.\n"
        'Example cycle 1/3: {"next_task":"Add 2 to 4","distractor_task":"Multiply the current result by 3"}\n'
        'Example cycle 2/3: {"next_task":"Multiply the current result by 3","distractor_task":"Add 2 to 4"}\n'
        'Example cycle 3/3: {"next_task":"Subtract 1 from the current result","distractor_task":"Multiply the current result by 3"}\n'
        "Example objective: Begin with elm. First append -a; finally uppercase only its letters.\n"
        'Example cycle 1/2: {"next_task":"Append -a to elm","distractor_task":"Uppercase only the letters"}\n'
        'Example cycle 2/2: {"next_task":"Uppercase only the letters","distractor_task":"Append -a to elm"}\n'
        "Never copy example values or operations.\n\n"
    )
    return indexed_create_prompt(item, cycle, current, completed).replace(
        "Create two distinct candidate tasks", examples + "Create two distinct candidate tasks",
        1)


@torch.inference_mode()
def independent_score(model, tok, item, cycle, current, completed, task, device,
                      indexed=False):
    ordered_rule = (
        f" For an explicitly ordered objective, Cycle {cycle} means operation clause {cycle}; "
        "Begin/Start/initial-state text is input, not an operation."
        if indexed else ""
    )
    prompt = (
        f"Original objective:\n{item['objective']}\n\n"
        f"Cycle: {cycle}/{item['cycles']}\n"
        f"Current result: {json.dumps(current, ensure_ascii=False)}\n"
        f"Completed tasks: {json.dumps(completed, ensure_ascii=False)}\n"
        f"Candidate task: {task}\n\n"
        "Is this candidate the single next undone task that should be executed now? "
        + ordered_rule + " Answer exactly Y or N."
    )
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    begin = time.monotonic()
    output = model(input_ids=torch.tensor([ids], device=device), use_cache=False,
                   logits_to_keep=1)
    elapsed = time.monotonic() - begin
    yes = tok.encode("Y", add_special_tokens=False)
    no = tok.encode("N", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("Y and N must each be one tokenizer token")
    logits = output.logits[0, -1].float()
    yes_logit, no_logit = float(logits[yes[0]]), float(logits[no[0]])
    return {"prompt": prompt, "yes_token": yes[0], "no_token": no[0],
            "yes_logit": yes_logit, "no_logit": no_logit,
            "score_yes_minus_no": yes_logit - no_logit, "seconds": elapsed}


def _answer_json(tok, tokens):
    text = tok.decode(tokens, skip_special_tokens=False)
    clean = text.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0]
    candidate = '{"result":' + clean
    try:
        value, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return text, None
    if not isinstance(value, dict) or "result" not in value:
        return text, None
    return text, value["result"]


@torch.inference_mode()
def bounded_execute(model, tok, prompt, device, think_tokens=128, answer_tokens=256):
    """Prototype the fixed-span control pattern intended for compilation."""
    initial = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=False, enable_thinking=False, return_dict=False)
    prefix = list(initial) + tok.encode(
        "\n<|im_start|>assistant\n<think>\n", add_special_tokens=False)
    runner = NativeRunner(model, tok, device, temperature=0.0, top_p=0.9, seed=0)
    logits = runner.start(prefix)
    reasoning = []
    # Native vocabulary tokens below endoftext exclude chat/tool/control symbols.
    for _ in range(think_tokens):
        token = int(logits[:248044].argmax())
        reasoning.append(token)
        logits = runner.feed([token])
    forced = tok.encode(
        '\n</think>\n\nRESULT_JSON: {"result":', add_special_tokens=False)
    logits = runner.feed(forced)
    answer = []
    for _ in range(answer_tokens):
        token = int(logits.argmax())
        answer.append(token)
        if token in (tok.eos_token_id, 248044):
            break
        logits = runner.feed([token])
    answer_text, result = _answer_json(tok, answer)
    return {"prompt": prompt, "think_tokens": reasoning,
            "think_text": tok.decode(reasoning), "forced_suffix_tokens": forced,
            "answer_tokens": answer, "answer_text": answer_text,
            "parse_ok": result is not None, "result": result,
            "forward_seconds": runner.forward_seconds}


def run_item(model, tok, item, item_index, mode, device, create_max):
    current, completed, actual, cycles = None, [], [], []
    stopped = None
    for cycle in range(1, item["cycles"] + 1):
        indexed = mode in ("indexed_jev_bounded128", "indexed_kshot_jev_bounded128")
        if mode == "indexed_kshot_jev_bounded128":
            create_prompt = indexed_kshot_create_prompt(item, cycle, current, completed)
        elif indexed:
            create_prompt = indexed_create_prompt(item, cycle, current, completed)
        else:
            create_prompt = _create_prompt(item, cycle, current, completed)
        create_generation = generate_chat(
            model, tok, create_prompt, device, thinking=False,
            max_new_tokens=create_max)
        created = parse_create(create_generation["text"])
        report = {"cycle": cycle, "state_in": current,
                  "create": {"prompt": create_prompt, "generation": create_generation,
                             "parsed": created, "parse_ok": created is not None}}
        if created is None:
            stopped = {"cycle": cycle, "phase": "create", "reason": "parse_failed"}
            cycles.append(report)
            break

        expected = counterbalanced_position(item_index, cycle)
        if expected == 1:
            candidates = [created["next_task"], created["distractor_task"]]
        else:
            candidates = [created["distractor_task"], created["next_task"]]
        report["create"].update({
            "candidates": candidates,
            "candidates_distinct": candidates[0].strip().lower() != candidates[1].strip().lower(),
        })
        scores = [independent_score(model, tok, item, cycle, current, completed, task,
                                    device, indexed=indexed)
                  for task in candidates]
        selected = 1 if scores[0]["score_yes_minus_no"] >= scores[1]["score_yes_minus_no"] else 2
        report["choose"] = {"method": "independent_yes_no_logit_score",
                            "scores": scores, "selected": selected,
                            "expected": expected, "parse_ok": True,
                            "correct": selected == expected}

        selected_task = candidates[selected - 1]
        execute_prompt = _execute_prompt(item, cycle, current, selected_task)
        if mode == "jev_off":
            generation = generate_chat(
                model, tok, execute_prompt, device, thinking=False,
                max_new_tokens=create_max)
            result = parse_result(generation["text"])
            execution = {"prompt": execute_prompt, "generation": generation,
                         "selected_task": selected_task, "result": result,
                         "parse_ok": result is not None}
        else:
            execution = bounded_execute(model, tok, execute_prompt, device)
            execution["selected_task"] = selected_task
        report["execute"] = execution
        cycles.append(report)
        if not execution["parse_ok"]:
            stopped = {"cycle": cycle, "phase": "execute", "reason": "parse_failed"}
            break
        current = execution["result"]
        completed.append(selected_task)
        actual.append(current)
    trace = {"mode": mode, "cycles": cycles, "stopped": stopped,
             "actual_results": actual, "parse_ok": stopped is None}
    return add_phase_diagnostics(item, trace)


def summarize(mode, reports):
    cycles = [cycle for report in reports for cycle in report["trace"]["cycles"]]
    chooses = [cycle["choose"] for cycle in cycles if "choose" in cycle]
    executes = [cycle["execute"] for cycle in cycles if "execute" in cycle]
    return {
        "mode": mode, "items": len(reports),
        "automated_pass": sum(report["score"]["automated_pass"] for report in reports),
        "manual_pending": sum(report["score"]["pass"] is None for report in reports),
        "parse_pass": sum(report["trace"]["parse_ok"] for report in reports),
        "phase": {
            "cycles_attempted": len(cycles),
            "create_parse_fail": sum(not cycle["create"]["parse_ok"] for cycle in cycles),
            "create_not_distinct": sum(cycle["create"].get("parse_ok", False)
                                       and not cycle["create"].get("candidates_distinct", False)
                                       for cycle in cycles),
            "choose_correct": sum(choose["correct"] for choose in chooses),
            "choose_total": len(chooses),
            "task2_expected": sum(choose["expected"] == 2 for choose in chooses),
            "task2_selected": sum(choose["selected"] == 2 for choose in chooses),
            "execute_parse_fail": sum(not execute["parse_ok"] for execute in executes),
            "result_prefix_correct": sum(execute.get("result_prefix_correct", False)
                                         for execute in executes),
            "execute_total": len(executes),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--device")
    parser.add_argument("--create-max", type=int, default=512)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to(device).eval()
    modes = {}
    for mode in args.modes:
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        for index, item in enumerate(suite["items"]):
            trace = run_item(model, tok, item, index, mode, device, args.create_max)
            score = score_item(item, trace["actual_results"])
            record = {"schema_version": 1, "item_id": item["id"],
                      "objective": item["objective"], "mode": mode,
                      "trace": trace, "score": score}
            path = mode_dir / f"{item['id']}.json"
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            reports.append({"item_id": item["id"], "trace_path": str(path),
                            "trace": trace, "score": score})
            print(json.dumps({"mode": mode, "item": item["id"],
                              "pass": score["automated_pass"],
                              "stopped": trace["stopped"]}), flush=True)
        modes[mode] = {"summary": summarize(mode, reports), "reports": reports}
        summary = {"schema_version": 1,
                   "benchmark": {"id": suite["benchmark_id"],
                                 "sha256": file_sha256(args.suite)},
                   "model": str(Path(args.base).resolve()), "modes": modes}
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({name: value["summary"] for name, value in modes.items()}, indent=2))


if __name__ == "__main__":
    main()
