"""Paired plain-chat and host-ledger evaluation on a frozen benchmark.

Plain Qwen receives one thinking-enabled turn with a length-matched token
ceiling.  The host arm runs the exact ``babyagi_ledger_pointer`` program with a
fixed model-owned reasoning span, while Python advances stages and simulates
visible filesystem responses.  Only separate live compiled traces prove that
the controller and qwenFS execute inside the checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.agent import program
from qfs.agent_native import run_native
from qfs.agent_suite import native_results
from qfs.analyze_budget import exact_mcnemar
from qfs.benchmark import CATEGORIES, file_sha256, load_suite, score_item
from qfs.ceiling import run_plain


ARMS = ("plain", "host")


def _runner_sha256():
    names = ("end_to_end.py", "agent.py", "agent_native.py", "agent_suite.py",
             "ceiling.py", "benchmark.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _score_counts(score):
    checks = score.get("automated_checks", [])
    return sum(check["pass"] for check in checks), len(checks)


def _plan_check(item, trace):
    text = trace.get("latest_files", {}).get("ledger.json")
    if text is None:
        return {"parse_ok": False, "initial_exact": False,
                "tasks_exact": False, "exact": False}
    try:
        ledger = json.loads(text)
    except json.JSONDecodeError:
        return {"parse_ok": False, "initial_exact": False,
                "tasks_exact": False, "exact": False}
    gold = item.get("provenance", {}).get("gold_steps", [])
    initial = gold[0]["incoming"] if gold else None
    tasks = [step["task"] for step in gold]
    initial_exact = ledger.get("initial") == initial
    got_tasks = ledger.get("tasks")
    tasks_exact = got_tasks == tasks
    return {"parse_ok": isinstance(got_tasks, list),
            "initial_exact": initial_exact, "tasks_exact": tasks_exact,
            "exact": initial_exact and tasks_exact,
            "actual_initial": ledger.get("initial"), "actual_tasks": got_tasks}


def _identity(suite_hash, runner_hash, arm, item_id, think_tokens,
              plain_answer_tokens):
    return {"suite_sha256": suite_hash, "runner_sha256": runner_hash,
            "arm": arm, "item_id": item_id, "think_tokens": think_tokens,
            "plain_answer_tokens": plain_answer_tokens}


def _load_if_matching(path, identity):
    if not path.exists():
        return None
    candidate = json.loads(path.read_text())
    return candidate if candidate.get("identity") == identity else None


def _write_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def _run_plain_record(model, tok, item, device, think_tokens, answer_tokens,
                      identity):
    token_ceiling = item["cycles"] * think_tokens + answer_tokens
    trace = run_plain(model, tok, item, device, thinking=True,
                      max_new_tokens=token_ceiling)
    score = score_item(item, trace["actual_results"])
    return {"schema_version": 1, "identity": identity,
            "arm": "plain", "item_id": item["id"],
            "category": item["category"], "cycles": item["cycles"],
            "token_ceiling": token_ceiling, "trace": trace, "score": score}


def _run_host_record(model, tok, item, device, think_tokens, answer_tokens,
                     identity):
    stages = program(tok, item["cycles"], "babyagi_ledger_pointer", think_tokens)
    trace = run_native(model, tok, stages, item["objective"], device=device,
                       temperature=0.0, top_p=0.9, seed=0,
                       limit_per_write=answer_tokens)
    actual = native_results(trace)
    score = score_item(item, actual)
    return {"schema_version": 1, "identity": identity,
            "arm": "host", "item_id": item["id"],
            "category": item["category"], "cycles": item["cycles"],
            "actual_results": actual, "trace": trace, "score": score,
            "plan": _plan_check(item, trace)}


def _arm_summary(records):
    chain_pass = sum(record["score"]["automated_pass"] for record in records)
    correct_steps = total_steps = parse_fail = hit_limit = 0
    by_category = {}
    by_length = {}
    for record in records:
        correct, total = _score_counts(record["score"])
        correct_steps += correct
        total_steps += total
        if record["arm"] == "plain":
            parse_fail += not record["trace"]["parse_ok"]
            hit_limit += record["trace"]["generation"]["hit_limit"]
        else:
            parse_fail += record["trace"]["stopped"] is not None
        for key, value in (("category", record["category"]),
                           ("length", str(record["cycles"]))):
            target = by_category if key == "category" else by_length
            group = target.setdefault(value, {"items": 0, "chain_pass": 0,
                                               "correct_steps": 0,
                                               "total_steps": 0})
            group["items"] += 1
            group["chain_pass"] += record["score"]["automated_pass"]
            group["correct_steps"] += correct
            group["total_steps"] += total
    for groups in (by_category, by_length):
        for group in groups.values():
            group["chain_accuracy"] = round(group["chain_pass"] / group["items"], 6)
            group["step_accuracy"] = round(
                group["correct_steps"] / group["total_steps"], 6)
    prefix = {}
    max_cycle = max(record["cycles"] for record in records)
    for cycle in range(1, max_cycle + 1):
        eligible = [record for record in records if record["cycles"] >= cycle]
        survived = 0
        for record in eligible:
            checks = record["score"].get("automated_checks", [])
            survived += len(checks) >= cycle and all(
                check["pass"] for check in checks[:cycle])
        prefix[str(cycle)] = {"eligible": len(eligible), "survived": survived,
                              "rate": round(survived / len(eligible), 6)}
    return {"items": len(records), "chain_pass": chain_pass,
            "chain_accuracy": round(chain_pass / len(records), 6),
            "correct_steps": correct_steps, "total_steps": total_steps,
            "step_accuracy": round(correct_steps / total_steps, 6),
            "parse_fail": parse_fail, "hit_limit": hit_limit,
            "by_category": by_category, "by_length": by_length,
            "prefix_survival": prefix}


def _clustered_chain_delta_ci(plain, host, samples=10000, seed=0):
    keys = sorted(plain)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        chosen = [rng.choice(keys) for _ in keys]
        plain_rate = sum(plain[key]["score"]["automated_pass"] for key in chosen) / len(keys)
        host_rate = sum(host[key]["score"]["automated_pass"] for key in chosen) / len(keys)
        deltas.append(host_rate - plain_rate)
    deltas.sort()
    lower = deltas[math.floor((len(deltas) - 1) * 0.025)]
    upper = deltas[math.ceil((len(deltas) - 1) * 0.975)]
    return [round(lower, 6), round(upper, 6)]


def _paired_summary(plain, host):
    if set(plain) != set(host):
        raise ValueError("plain and host arms contain different items")
    improvements = sum(not plain[key]["score"]["automated_pass"]
                       and host[key]["score"]["automated_pass"] for key in plain)
    regressions = sum(plain[key]["score"]["automated_pass"]
                      and not host[key]["score"]["automated_pass"] for key in plain)
    plain_rate = sum(record["score"]["automated_pass"]
                     for record in plain.values()) / len(plain)
    host_rate = sum(record["score"]["automated_pass"]
                    for record in host.values()) / len(host)
    return {"items": len(plain), "plain_chain_accuracy": round(plain_rate, 6),
            "host_chain_accuracy": round(host_rate, 6),
            "delta": round(host_rate - plain_rate, 6),
            "host_improvements": improvements, "host_regressions": regressions,
            "discordant": improvements + regressions,
            "exact_mcnemar_p": exact_mcnemar(improvements, regressions),
            "item_bootstrap_delta_95": _clustered_chain_delta_ci(plain, host)}


def _compiled_selection(suite, host):
    selected = {}
    for length in sorted({item["cycles"] for item in suite["items"]}):
        candidates = sorted(item["id"] for item in suite["items"]
                            if item["cycles"] == length
                            and host[item["id"]]["score"]["automated_pass"])
        fallback = False
        if not candidates:
            candidates = sorted(item["id"] for item in suite["items"]
                                if item["cycles"] == length)
            fallback = True
        selected[str(length)] = {"item_id": candidates[0],
                                 "host_chain_pass": not fallback,
                                 "fallback_negative_trace": fallback}
    return selected


def run(base, suite_path, output_dir, think_tokens=256, plain_answer_tokens=256,
        device=None):
    suite_path, output_dir = Path(suite_path), Path(output_dir)
    suite = load_suite(suite_path)
    suite_hash, runner_hash = file_sha256(suite_path), _runner_sha256()
    device = device or ("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16).to(device).eval()
    records = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for index, item in enumerate(suite["items"], 1):
            identity = _identity(suite_hash, runner_hash, arm, item["id"],
                                 think_tokens, plain_answer_tokens)
            path = output_dir / arm / f"{item['id']}.json"
            record = _load_if_matching(path, identity)
            if record is None:
                if arm == "plain":
                    record = _run_plain_record(
                        model, tok, item, device, think_tokens,
                        plain_answer_tokens, identity)
                else:
                    record = _run_host_record(
                        model, tok, item, device, think_tokens,
                        plain_answer_tokens, identity)
                _write_record(path, record)
            records[arm][item["id"]] = record
            print(json.dumps({"arm": arm, "item": index, "items": len(suite["items"]),
                              "item_id": item["id"],
                              "chain_pass": record["score"]["automated_pass"]}),
                  flush=True)
    report = {
        "schema_version": 1, "experiment": "babyagi-v3-end-to-end",
        "suite": {"path": str(suite_path), "id": suite["benchmark_id"],
                  "sha256": suite_hash},
        "model": str(Path(base).resolve()), "device": device,
        "runner_sha256": runner_hash, "decoding": "greedy",
        "think_tokens": think_tokens,
        "plain_token_ceiling": "cycles * think_tokens + plain_answer_tokens",
        "plain_answer_tokens": plain_answer_tokens,
        "arms": {arm: _arm_summary(list(rows.values()))
                 for arm, rows in records.items()},
        "paired_chain_comparison": _paired_summary(records["plain"], records["host"]),
        "host_plan": {
            "exact": sum(record["plan"]["exact"] for record in records["host"].values()),
            "parse_ok": sum(record["plan"]["parse_ok"]
                            for record in records["host"].values()),
            "items": len(records["host"]),
        },
        "compiled_trace_selection": _compiled_selection(suite, records["host"]),
        "limitations": [
            "host arm advances phases and simulates visible filesystem responses",
            "only live compiled traces validate controller, qwenFS, and HALT",
            "v3 tests deterministic execution and task extraction, not open planning",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--think-tokens", type=int, default=256)
    parser.add_argument("--plain-answer-tokens", type=int, default=256)
    parser.add_argument("--device")
    args = parser.parse_args()
    report = run(args.base, args.suite, args.output_dir, args.think_tokens,
                 args.plain_answer_tokens, args.device)
    print(json.dumps({"output": str(args.output_dir / "summary.json"),
                      "plain": report["arms"]["plain"]["chain_pass"],
                      "host": report["arms"]["host"]["chain_pass"],
                      "selected": report["compiled_trace_selection"]}, indent=2))


if __name__ == "__main__":
    main()
