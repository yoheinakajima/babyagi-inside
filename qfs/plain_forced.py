"""Budget-forced plain Qwen at token budgets matched to the compiled loop.

    python -m qfs.plain_forced run BASE SUITE COMPILED ARM OUTPUT_DIR [--device mps]
    python -m qfs.plain_forced compare COMPILED RECORDS_DIR

The one-turn plain arm (``qfs.ceiling``'s ``plain_on``) gives Qwen a single ceiling of
N*256+256 new tokens. On Qwen3.5-2B and Qwen3.5-4B no run closed its thinking within it, so
its scores come from JSON the parser found inside unfinished reasoning. Here the host closes
the thinking when its cap is reached (budget forcing: it appends "\\n</think>\\n\\n", the chat
template's close), and Qwen then writes its answer under a separate cap. The prompt, parser
and scorer are the one-turn arm's: ``qfs.ceiling.run_plain``'s prompt, ``parse_plain`` over
the whole generation, and ``qfs.benchmark.score_item``.

Arms, for an item with N cycles whose compiled run emitted T tokens:

* F1: thinking N*256, answer 256 (the one-turn ceiling, with an answer guaranteed);
* F2: thinking (N+1)*256, answer N*256 (the compiled loop's model-token allowance);
* F3: thinking T - N*256, answer N*256 (the compiled run's own total).

``COMPILED`` is a JSON file with, per model, the compiled chain verdict and token total of
each item (``results/plain-forced-v3/compiled.json``). See
``docs/plain-forced-v3-preregistration.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qfs.benchmark import file_sha256, load_suite, score_item
from qfs.ceiling import parse_plain
from qfs.end_to_end import _paired_summary

ARMS = ("F1", "F2", "F3")
THINK, ANSWER = 256, 256  # the protocol's per-stage reasoning and write allowances
THINK_END = 248069  # "</think>" in the Qwen3.5 tokenizers
CLOSE = [198, 248069, 271]  # "\n</think>\n\n"


def caps(arm, cycles, compiled_tokens):
    """(thinking cap, answer cap) in new tokens."""
    if arm == "F1":
        return cycles * THINK, ANSWER
    if arm == "F2":
        return (cycles + 1) * THINK, cycles * ANSWER
    if arm == "F3":
        return compiled_tokens - cycles * ANSWER, cycles * ANSWER
    raise ValueError(arm)


def plain_prompt(item):
    """``qfs.ceiling.run_plain``'s prompt, verbatim (tests/test_plain_forced.py)."""
    return (
        item["objective"] + "\n\nSolve the entire objective in one response. Preserve the "
        "result after every successive operation, not just the final answer. Do not include the "
        "initial state as a result. Each entry must be the complete result after that operation: "
        "a JSON number for numeric work, the requested JSON value for structured work, or a JSON "
        "string for text (use `\\n` inside multiline strings). End with `FINAL_JSON: ` followed by "
        f"exactly one valid JSON object shaped `{{\"results\":[...]}}`, whose `results` array has "
        f"exactly {item['cycles']} entries in task order. Do not put text after the object."
    )


def generate_forced(model, tok, prompt, device, think_cap, answer_cap):
    import torch

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                  add_generation_prompt=True, enable_thinking=True,
                                  return_dict=False)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    begin = time.monotonic()
    with torch.inference_mode():
        out = model.generate(torch.tensor([ids], device=device), max_new_tokens=think_cap,
                             do_sample=False, pad_token_id=tok.eos_token_id,
                             eos_token_id=[THINK_END, *eos])
        thinking = out[0, len(ids):].tolist()
        closed = bool(thinking) and thinking[-1] == THINK_END
        ended = bool(thinking) and thinking[-1] in eos
        forced = [] if closed or ended else list(CLOSE)
        answer = []
        if not ended:
            context = ids + thinking + forced
            out = model.generate(torch.tensor([context], device=device),
                                 max_new_tokens=answer_cap, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            answer = out[0, len(context):].tolist()
    generated = thinking + forced + answer
    return {"text": tok.decode(generated, skip_special_tokens=False),
            "answer_text": tok.decode(answer, skip_special_tokens=False),
            "thinking_tokens": len(thinking), "answer_tokens": len(answer),
            "model_tokens": len(thinking) + len(answer),
            "closed_naturally": closed, "forced_close": bool(forced),
            "ended_in_thinking": ended,
            "answer_hit_limit": len(answer) >= answer_cap,
            "answer_ended": bool(answer) and answer[-1] in eos,
            "seconds": time.monotonic() - begin}


def score_record(item, generation):
    actual = parse_plain(generation["text"], item["cycles"])
    return {"item_id": item["id"], "category": item["category"], "cycles": item["cycles"],
            "generation": generation, "parse_ok": actual is not None,
            "answer_parse_ok": parse_plain(generation["answer_text"], item["cycles"]) is not None,
            "actual_results": actual or [], "score": score_item(item, actual or [])}


def run(base, suite_path, compiled_path, model_name, arm, output_dir, device):
    import torch
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    compiled = json.loads(Path(compiled_path).read_text())[model_name]["records"]
    suite = load_suite(suite_path)
    tok = AutoTokenizer.from_pretrained(base)
    model = Qwen3_5ForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(device).eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in suite["items"]:
        think_cap, answer_cap = caps(arm, item["cycles"], compiled[item["id"]]["tokens"])
        generation = generate_forced(model, tok, plain_prompt(item), device, think_cap,
                                     answer_cap)
        record = {"schema_version": 1,
                  "identity": {"model": model_name, "arm": arm, "item_id": item["id"],
                               "think_cap": think_cap, "answer_cap": answer_cap,
                               "suite_sha256": file_sha256(suite_path),
                               "runner_sha256": file_sha256(Path(__file__))},
                  **score_record(item, generation)}
        (output_dir / f"{item['id']}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"item_id": item["id"], "pass": record["score"]["automated_pass"],
                          "forced_close": generation["forced_close"],
                          "model_tokens": generation["model_tokens"]}), flush=True)


def compare(compiled_path, records_dir):
    """Compiled minus plain over every model and arm found under ``records_dir``."""
    compiled = json.loads(Path(compiled_path).read_text())
    out = {}
    for model_dir in sorted(Path(records_dir).iterdir()):
        if model_dir.name not in compiled:
            continue
        verdicts = {key: {"score": {"automated_pass": row["chain_pass"]}}
                    for key, row in compiled[model_dir.name]["records"].items()}
        for arm in ARMS:
            records = {path.stem: json.loads(path.read_text())
                       for path in sorted((model_dir / arm).glob("*.json"))}
            if set(records) != set(verdicts):
                continue
            summary = _paired_summary(records, verdicts)
            out[f"{model_dir.name}/{arm}"] = {
                "compiled_chains": sum(row["score"]["automated_pass"] for row in verdicts.values()),
                "plain_chains": sum(row["score"]["automated_pass"] for row in records.values()),
                "forced_close": sum(row["generation"]["forced_close"] for row in records.values()),
                "model_tokens": sum(row["generation"]["model_tokens"] for row in records.values()),
                "compiled_minus_plain": summary}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("base", type=Path)
    runner.add_argument("suite", type=Path)
    runner.add_argument("compiled", type=Path)
    runner.add_argument("model", help="key in COMPILED, e.g. 4b")
    runner.add_argument("arm", choices=ARMS)
    runner.add_argument("output_dir", type=Path)
    runner.add_argument("--device", default="mps")
    comparer = commands.add_parser("compare")
    comparer.add_argument("compiled", type=Path)
    comparer.add_argument("records_dir", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.base, args.suite, args.compiled, args.model, args.arm, args.output_dir,
            args.device)
    else:
        print(json.dumps(compare(args.compiled, args.records_dir), indent=2))


if __name__ == "__main__":
    main()
