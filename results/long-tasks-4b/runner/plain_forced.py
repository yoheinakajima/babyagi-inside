"""E14: budget-forced plain Qwen against the compiled loop, at matched token budgets.

    python -m harness.plain_forced freeze            # writes provenance/e14-compiled.json
    python -m harness.plain_forced run MODEL ARM     # step ARM-MODEL, e.g. F2-4b
    python -m harness.plain_forced analyze           # step analysis

The frozen plain arm (2B sweep, E9/E12) ran one thinking-enabled turn with a ceiling of
N*256+256 new tokens. No run on either model closed its thinking: every plain score came
from JSON the parser found inside unfinished reasoning. Here the host closes the thinking
when its cap is reached (budget forcing: it appends "\\n</think>\\n\\n"), and Qwen then
writes its answer under a separate cap. The prompt, parser, and scorer are the frozen
plain arm's: qfs.ceiling.run_plain's prompt, qfs.ceiling.parse_plain over the whole
generation, and qfs.benchmark.score_item. Arms and settings are frozen in
experiments/E14-plain-forced/prereg.md. Records resume by identity; each command exits 0
whenever it executed, with its verdict in metrics.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from harness import port
from harness.upstream import ROOT, SUITE_V3, module
from lab.repo import sha256_file

EXPERIMENT = "E14"
FROZEN = ROOT / "provenance" / "e14-compiled.json"
RECORDS = ROOT / "build" / "e14-plain-forced"
MODELS = {"2b": "models/Qwen3.5-2B", "4b": "models/Qwen3.5-4B"}
ARMS = ("F1", "F2", "F3")
THINK, ANSWER = 256, 256  # the protocol's per-stage reasoning and write allowances
DEVICE = "mps"
THINK_END = 248069  # "</think>" in both tokenizers (prereg.md, Arms)
CLOSE = [198, 248069, 271]  # "\n</think>\n\n", the chat template's thinking close
# The validated compiled sweeps each comparison pairs with (summary.json SHA-256).
COMPILED = {
    "2b": {"dir": Path("~/Developer/Untapped/qwenfs-babyagi-m5/results/"
                       "babyagi-v3-full-live-compiled").expanduser(),
           "summary_sha256": "47eaf336a61a209483555edd961a745995644bb578ecf2a0683f3f4a7cefb9b6"},
    "4b": {"dir": ROOT / "build" / "e12-sweep-4b",
           "summary_sha256": "8b0ee42deb182cd591c5b45c79cfa1f1f0a0d14a27bdbd24f60656fb3a33904e"},
}


def caps(arm: str, cycles: int, compiled_tokens: int) -> tuple[int, int]:
    """(thinking cap, answer cap) in new tokens."""
    if arm == "F1":  # the frozen arm's ceiling, N*256 + 256, with the answer guaranteed
        return cycles * THINK, ANSWER
    if arm == "F2":  # the loop's model-token allowance: N reasoning spans and N+1 writes
        return (cycles + 1) * THINK, cycles * ANSWER
    if arm == "F3":  # the compiled run's own total emitted tokens on this item
        return compiled_tokens - cycles * ANSWER, cycles * ANSWER
    raise ValueError(arm)


def plain_prompt(item: dict) -> str:
    """qfs.ceiling.run_plain's prompt, verbatim (harness/tests/test_plain_forced.py)."""
    return (
        item["objective"] + "\n\nSolve the entire objective in one response. Preserve the "
        "result after every successive operation, not just the final answer. Do not include the "
        "initial state as a result. Each entry must be the complete result after that operation: "
        "a JSON number for numeric work, the requested JSON value for structured work, or a JSON "
        "string for text (use `\\n` inside multiline strings). End with `FINAL_JSON: ` followed by "
        f"exactly one valid JSON object shaped `{{\"results\":[...]}}`, whose `results` array has "
        f"exactly {item['cycles']} entries in task order. Do not put text after the object."
    )


# ---------------------------------------------------------------- freeze


def freeze() -> dict:
    out = {"schema_version": 1,
           "description": "E14's compiled comparisons: per item, the chain verdict "
                          "(benchmark_pass and halted) and total emitted tokens of the "
                          "validated live compiled sweeps, with each record.json SHA-256."}
    for name, source in COMPILED.items():
        summary = source["dir"] / "summary.json"
        if sha256_file(summary) != source["summary_sha256"]:
            raise RuntimeError(f"{summary} is not the validated {name} sweep summary")
        records = {}
        for path in sorted((source["dir"] / "items").glob("*/record.json")):
            row = json.loads(path.read_text())
            records[row["item_id"]] = {"cycles": row["cycles"], "tokens": row["tokens"],
                                       "chain_pass": row["benchmark_pass"] and row["halted"],
                                       "record_sha256": sha256_file(path)}
        out[name] = {"summary_sha256": source["summary_sha256"],
                     "chain_pass": sum(row["chain_pass"] for row in records.values()),
                     "tokens": sum(row["tokens"] for row in records.values()),
                     "records": records}
    return out


# ---------------------------------------------------------------- run


def runner_sha256() -> str:
    return sha256_file(Path(__file__))


def generate_forced(model, tok, prompt: str, think_cap: int, answer_cap: int) -> dict:
    import torch

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                  add_generation_prompt=True, enable_thinking=True,
                                  return_dict=False)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    begin = time.monotonic()
    with torch.inference_mode():
        out = model.generate(torch.tensor([ids], device=DEVICE), max_new_tokens=think_cap,
                             do_sample=False, pad_token_id=tok.eos_token_id,
                             eos_token_id=[THINK_END, *eos])
        thinking = out[0, len(ids):].tolist()
        closed = bool(thinking) and thinking[-1] == THINK_END
        ended = bool(thinking) and thinking[-1] in eos
        forced = [] if closed or ended else list(CLOSE)
        answer = []
        if not ended:
            context = ids + thinking + forced
            out = model.generate(torch.tensor([context], device=DEVICE),
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


def run(model_name: str, arm: str, output: Path) -> dict:
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    ceiling, benchmark = module("qfs.ceiling"), module("qfs.benchmark")
    frozen = json.loads(FROZEN.read_text())[model_name]["records"]
    suite = benchmark.load_suite(SUITE_V3)
    suite_hash = sha256_file(SUITE_V3)
    base = ROOT / MODELS[model_name]
    tok = AutoTokenizer.from_pretrained(base)
    model = Qwen3_5ForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(DEVICE).eval()
    folder = RECORDS / model_name / arm
    folder.mkdir(parents=True, exist_ok=True)
    done = reused = 0
    for index, item in enumerate(suite["items"], 1):
        think_cap, answer_cap = caps(arm, item["cycles"], frozen[item["id"]]["tokens"])
        identity = {"experiment": EXPERIMENT, "model": model_name, "arm": arm,
                    "item_id": item["id"], "think_cap": think_cap, "answer_cap": answer_cap,
                    "suite_sha256": suite_hash, "runner_sha256": runner_sha256()}
        path = folder / f"{item['id']}.json"
        if path.exists() and json.loads(path.read_text())["identity"] == identity:
            reused += 1
            continue
        generation = generate_forced(model, tok, plain_prompt(item), think_cap, answer_cap)
        actual = ceiling.parse_plain(generation["text"], item["cycles"])
        record = {"schema_version": 1, "identity": identity, "item_id": item["id"],
                  "category": item["category"], "cycles": item["cycles"],
                  "generation": generation, "parse_ok": actual is not None,
                  "answer_parse_ok": ceiling.parse_plain(generation["answer_text"],
                                                         item["cycles"]) is not None,
                  "actual_results": actual or [],
                  "score": benchmark.score_item(item, actual or [])}
        port._write(path, record)
        done += 1
        print(json.dumps({"item": index, "item_id": item["id"],
                          "pass": record["score"]["automated_pass"],
                          "parse_ok": record["parse_ok"],
                          "forced_close": generation["forced_close"],
                          "model_tokens": generation["model_tokens"],
                          "seconds": round(generation["seconds"], 1)}), flush=True)
    summary = arm_summary(model_name, arm)
    port._write(output / f"{arm}-{model_name}.json", summary)
    return {"model": model_name, "arm": arm, "ran": done, "reused": reused, **summary}


# ---------------------------------------------------------------- analysis


def load_records(model_name: str, arm: str) -> dict:
    folder = RECORDS / model_name / arm
    return {path.stem: json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))}


def arm_summary(model_name: str, arm: str) -> dict:
    records = load_records(model_name, arm)
    gens = [row["generation"] for row in records.values()]
    return {"items": len(records),
            "chain_pass": sum(row["score"]["automated_pass"] for row in records.values()),
            "parse_ok": sum(row["parse_ok"] for row in records.values()),
            "answer_parse_ok": sum(row["answer_parse_ok"] for row in records.values()),
            "closed_naturally": sum(g["closed_naturally"] for g in gens),
            "forced_close": sum(g["forced_close"] for g in gens),
            "answer_hit_limit": sum(g["answer_hit_limit"] for g in gens),
            "model_tokens": sum(g["model_tokens"] for g in gens),
            "seconds": round(sum(g["seconds"] for g in gens), 1),
            "records_sha256": hashlib.sha256(json.dumps(
                {key: sha256_file(RECORDS / model_name / arm / f"{key}.json")
                 for key in records}, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()}


def _as_scored(verdicts: dict) -> dict:
    return {key: {"score": {"automated_pass": bool(value)}} for key, value in verdicts.items()}


def paired(compiled: dict, plain: dict) -> dict:
    """Compiled minus plain over the same items: end_to_end._paired_summary (bootstrap, McNemar)."""
    summary = module("qfs.end_to_end")._paired_summary(_as_scored(plain), _as_scored(compiled))
    return {**summary, "verdict": port.verdict(summary["item_bootstrap_delta_95"])}


def analyze(output: Path) -> dict:
    frozen = json.loads(FROZEN.read_text())
    # The frozen one-turn arm, for reference: provenance/plain-baseline.json (2B) and E12.
    one_turn = {"2b": json.loads((ROOT / "provenance" / "plain-baseline.json").read_text())
                ["chain_pass"], "4b": port.load_frozen()["plain_4b"]["chain_pass"]}
    result = {}
    for model_name in MODELS:
        compiled = {key: row["chain_pass"] for key, row in frozen[model_name]["records"].items()}
        rows = {}
        for arm in ARMS:
            records = load_records(model_name, arm)
            if len(records) != len(compiled):
                rows[arm] = {"complete": False, "items": len(records)}
                continue
            plain = {key: row["score"]["automated_pass"] for key, row in records.items()}
            rows[arm] = {"complete": True, **arm_summary(model_name, arm),
                         "compiled_minus_plain": paired(compiled, plain)}
        result[model_name] = {"compiled_chain_pass": sum(compiled.values()),
                              "one_turn_chain_pass": one_turn[model_name], "arms": rows}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=os.environ.get("LAB_OUTPUT_DIR"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    runner = commands.add_parser("run")
    runner.add_argument("model", choices=sorted(MODELS))
    runner.add_argument("arm", choices=ARMS)
    commands.add_parser("analyze")
    args = parser.parse_args()
    if args.command == "freeze":
        frozen = freeze()
        port._write(FROZEN, frozen)
        print(f"wrote {FROZEN.relative_to(ROOT)}: compiled 2B {frozen['2b']['chain_pass']}/32, "
              f"compiled 4B {frozen['4b']['chain_pass']}/32")
        return
    output = port._output(args)
    metrics = run(args.model, args.arm, output) if args.command == "run" else analyze(output)
    port._write(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
