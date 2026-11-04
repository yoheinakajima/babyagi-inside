"""Resumable, checksum-bound live sweep of compiled BabyAGI checkpoints.

Every objective is decoded from a fresh qwenFS cache. Checkpoints are loaded
once per supported cycle count, but no tokens, files, or model state carry
between objectives. The host supplies only the initial objective prefill and
greedily samples subsequent tokens until checkpoint HALT or the external bound.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import traceback

import torch

from qfs.agent_account import account
from qfs.agent_decode import CompiledDecoder
from qfs.agent_evaluate import inspect
from qfs.benchmark import (file_sha256, load_suite, results_from_evaluation,
                           score_item)
from qfs.end_to_end import _paired_summary
from qfs.plan_fidelity import semantic_plan_score


SCHEMA_VERSION = 1
SOURCE_FILES = (
    "compiled_sweep.py", "agent_decode.py", "agent_evaluate.py",
    "agent_account.py", "benchmark.py", "end_to_end.py", "plan_fidelity.py",
)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _json_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_hashes():
    root = Path(__file__).parent
    return {name: file_sha256(root / name) for name in SOURCE_FILES}


def _checkpoint_manifest(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    files = []
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in checkpoint.rglob("*")
                       if candidate.is_file()):
        relative = str(path.relative_to(checkpoint))
        stat = path.stat()
        sha256 = file_sha256(path)
        row = {"path": relative, "bytes": stat.st_size,
               "mtime_ns": stat.st_mtime_ns, "sha256": sha256}
        files.append(row)
        digest.update(f"{relative}\0{stat.st_size}\0{sha256}\n".encode())
    if not files:
        raise ValueError(f"checkpoint has no files: {checkpoint}")
    return {"path": str(checkpoint), "sha256": digest.hexdigest(),
            "bytes": sum(row["bytes"] for row in files), "files": files}


def _checkpoint_stat_matches(manifest):
    root = Path(manifest["path"])
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file():
            return False
        stat = path.stat()
        if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            return False
    return True


def _plain_manifest(root, suite, suite_sha256):
    root = Path(root).resolve()
    records = {}
    for item in suite["items"]:
        path = root / "plain" / f"{item['id']}.json"
        record = json.loads(path.read_text())
        identity = record.get("identity", {})
        if (identity.get("item_id") != item["id"] or identity.get("arm") != "plain"
                or identity.get("suite_sha256") != suite_sha256):
            raise ValueError(f"invalid frozen plain record: {path}")
        records[item["id"]] = {"path": str(path), "sha256": file_sha256(path),
                               "chain_pass": record["score"]["automated_pass"]}
    return {"root": str(root), "records": records,
            "chain_pass": sum(row["chain_pass"] for row in records.values())}


def _experiment_config(suite_path, output_dir, checkpoint_pattern, plain_root,
                       device, seed, temperature, limit):
    return {
        "suite_path": str(Path(suite_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "checkpoint_pattern": checkpoint_pattern,
        "plain_root": str(Path(plain_root).resolve()),
        "device": device, "seed": seed, "temperature": temperature,
        "limit": limit, "decoding": "greedy" if temperature <= 0 else "sampled",
        "order": "ascending cycles, then suite order",
    }


def prepare_manifest(suite_path, output_dir, checkpoint_pattern, plain_root,
                     device="mps", seed=0, temperature=0.0, limit=7000,
                     rehash=False):
    suite_path, output_dir = Path(suite_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = load_suite(suite_path)
    suite_sha256 = file_sha256(suite_path)
    config = _experiment_config(suite_path, output_dir, checkpoint_pattern,
                                plain_root, device, seed, temperature, limit)
    manifest_path = output_dir / "manifest.json"
    source_hashes = _source_hashes()
    if manifest_path.exists() and not rehash:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("experiment") != config:
            raise ValueError("existing sweep manifest has different experiment settings")
        if manifest.get("suite", {}).get("sha256") != suite_sha256:
            raise ValueError("frozen suite differs from existing sweep manifest")
        if manifest.get("source_sha256") != source_hashes:
            raise ValueError("sweep source differs from existing sweep manifest")
        if not all(_checkpoint_stat_matches(row)
                   for row in manifest.get("checkpoints", {}).values()):
            raise ValueError("checkpoint files differ from existing sweep manifest")
        for item_id, row in manifest["plain"]["records"].items():
            if file_sha256(row["path"]) != row["sha256"]:
                raise ValueError(f"frozen plain record changed: {item_id}")
        return manifest

    cycles = sorted({item["cycles"] for item in suite["items"]})
    checkpoints = {}
    for cycle in cycles:
        checkpoint = Path(checkpoint_pattern.format(cycles=cycle))
        row = _checkpoint_manifest(checkpoint)
        meta = json.loads((checkpoint / "agent.json").read_text())
        if meta.get("cycles") != cycle or meta.get("protocol") != "babyagi_ledger_pointer":
            raise ValueError(f"checkpoint metadata mismatch for {checkpoint}")
        checkpoints[str(cycle)] = row
        print(json.dumps({"prepared_checkpoint": cycle, "sha256": row["sha256"],
                          "bytes": row["bytes"]}), flush=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": "babyagi-v3-full-live-compiled-sweep",
        "experiment": config,
        "suite": {"id": suite["benchmark_id"], "sha256": suite_sha256,
                  "items": len(suite["items"])},
        "source_sha256": source_hashes,
        "checkpoints": checkpoints,
        "plain": _plain_manifest(plain_root, suite, suite_sha256),
    }
    manifest["experiment_sha256"] = _json_digest(manifest)
    _write_json(manifest_path, manifest)
    return manifest


def _identity(manifest, item):
    return {"experiment_sha256": manifest["experiment_sha256"],
            "suite_sha256": manifest["suite"]["sha256"],
            "item_id": item["id"], "cycles": item["cycles"],
            "checkpoint_sha256": manifest["checkpoints"][str(item["cycles"])]["sha256"]}


def _artifact_hashes(item_dir):
    names = ("run.json", "evaluation.json", "account.json", "score.json", "plan.json")
    return {name: file_sha256(item_dir / name) for name in names}


def _load_complete(item_dir, identity):
    record_path = item_dir / "record.json"
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())
    if record.get("identity") != identity:
        raise ValueError(f"completed item identity mismatch: {record_path}")
    if record.get("artifact_sha256") != _artifact_hashes(item_dir):
        raise ValueError(f"completed item artifacts changed: {record_path}")
    return record


def _validate_run(run, checkpoint, item, config):
    expected = {"checkpoint": str(Path(checkpoint).resolve()),
                "objective": item["objective"], "seed": config["seed"],
                "temperature": config["temperature"], "limit": config["limit"]}
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError(f"partial run identity mismatch for {item['id']}")


def _plan_score(item, evaluation):
    text = evaluation.get("latest_files", {}).get("ledger.json", {}).get("text")
    gold = item.get("provenance", {}).get("gold_steps", [])
    expected_initial = gold[0]["incoming"] if gold else None
    expected_tasks = [step["task"] for step in gold]
    result = {"metric": "frozen_exact_string", "parse_ok": False,
              "initial_exact": False, "tasks_exact": False, "exact": False,
              "expected_initial": expected_initial,
              "expected_tasks": expected_tasks, "actual_initial": None,
              "actual_tasks": None,
              "exploratory_semantic": semantic_plan_score(expected_tasks, None)}
    if text is None:
        return result
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return result
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list):
        return result
    result.update({"parse_ok": True, "actual_initial": value.get("initial"),
                   "actual_tasks": value.get("tasks")})
    result["initial_exact"] = value.get("initial") == expected_initial
    result["tasks_exact"] = value.get("tasks") == expected_tasks
    result["exact"] = result["initial_exact"] and result["tasks_exact"]
    result["exploratory_semantic"] = semantic_plan_score(
        expected_tasks, value.get("tasks"))
    return result


def _error_report(error):
    return {"error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()}


def _postprocess(item, item_dir, checkpoint, decoder, identity):
    run_path = item_dir / "run.json"
    run = json.loads(run_path.read_text())
    try:
        evaluation = inspect(run_path, checkpoint)
    except Exception as error:  # preserve a failed trace and continue the frozen sweep
        evaluation = _error_report(error)
    _write_json(item_dir / "evaluation.json", evaluation)

    try:
        accounting = account(run_path, checkpoint, tokenizer=decoder.tok)
    except Exception as error:
        accounting = _error_report(error)
    _write_json(item_dir / "account.json", accounting)

    actual = results_from_evaluation(evaluation) if "error" not in evaluation else []
    score = score_item(item, actual)
    score["actual_results"] = actual
    _write_json(item_dir / "score.json", score)
    plan = _plan_score(item, evaluation)
    _write_json(item_dir / "plan.json", plan)

    invariants = accounting.get("invariants", {})
    system_checks = {
        "checkpoint_halt": bool(run.get("halted")),
        "controller": evaluation.get("controller_pass") is True,
        "filesystem": evaluation.get("filesystem_pass") is True,
        "complete_cycles": evaluation.get("complete_cycles") == item["cycles"],
        "token_accounting": (bool(invariants) and all(invariants.values())
                             and accounting.get("totals", {}).get("accounted_tokens")
                             == accounting.get("totals", {}).get("emitted_tokens")),
    }
    record = {
        "schema_version": SCHEMA_VERSION, "identity": identity,
        "item_id": item["id"], "category": item["category"],
        "cycles": item["cycles"], "system_checks": system_checks,
        "system_pass": all(system_checks.values()),
        "benchmark_pass": score["automated_pass"],
        "correct_steps": sum(check["pass"] for check in score["automated_checks"]),
        "total_steps": len(score["automated_checks"]),
        "plan_exact": plan["exact"], "plan_parse_ok": plan["parse_ok"],
        "plan_semantic_exact": plan["exploratory_semantic"]["exact"],
        "plan_semantic_correct_tasks": plan["exploratory_semantic"]["correct_tasks"],
        "plan_tasks": plan["exploratory_semantic"]["total_tasks"],
        "halted": bool(run.get("halted")), "tokens": len(run.get("tokens", [])),
        "seconds": run.get("seconds"),
    }
    record["artifact_sha256"] = _artifact_hashes(item_dir)
    _write_json(item_dir / "record.json", record)
    return record


def _group_summary(records, key):
    values = sorted({str(record[key]) for record in records})
    groups = {}
    for value in values:
        rows = [record for record in records if str(record[key]) == value]
        steps = sum(row["total_steps"] for row in rows)
        correct = sum(row["correct_steps"] for row in rows)
        groups[value] = {"items": len(rows),
                         "system_pass": sum(row["system_pass"] for row in rows),
                         "chain_pass": sum(row["benchmark_pass"] for row in rows),
                         "correct_steps": correct, "total_steps": steps,
                         "step_accuracy": round(correct / steps, 6) if steps else None}
    return groups


def _wilson(successes, total, z=1.96):
    if not total:
        return [None, None]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total
                           + z * z / (4 * total * total)) / denominator
    return [round(center - radius, 6), round(center + radius, 6)]


def _summary(output_dir, suite, manifest):
    records = []
    for item in suite["items"]:
        path = output_dir / "items" / item["id"] / "record.json"
        if path.exists():
            records.append(json.loads(path.read_text()))
    chain_pass = sum(record["benchmark_pass"] for record in records)
    correct_steps = sum(record["correct_steps"] for record in records)
    total_steps = sum(record["total_steps"] for record in records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": manifest["experiment_name"],
        "experiment_sha256": manifest["experiment_sha256"],
        "items_total": len(suite["items"]), "items_completed": len(records),
        "system_pass": sum(record["system_pass"] for record in records),
        "chain_pass": chain_pass,
        "chain_accuracy": round(chain_pass / len(records), 6) if records else None,
        "chain_wilson_95": _wilson(chain_pass, len(records)),
        "correct_steps": correct_steps, "total_steps": total_steps,
        "step_accuracy": round(correct_steps / total_steps, 6) if total_steps else None,
        "plan_parse_ok": sum(record["plan_parse_ok"] for record in records),
        "plan_exact": sum(record["plan_exact"] for record in records),
        "plan_semantic_exact": sum(record["plan_semantic_exact"] for record in records),
        "plan_semantic_correct_tasks": sum(
            record["plan_semantic_correct_tasks"] for record in records),
        "plan_tasks": sum(record["plan_tasks"] for record in records),
        "by_category": _group_summary(records, "category"),
        "by_length": _group_summary(records, "cycles"),
        "complete": len(records) == len(suite["items"]),
        "records": records,
    }
    if summary["complete"]:
        plain, compiled = {}, {}
        for item in suite["items"]:
            plain_record = json.loads(
                Path(manifest["plain"]["records"][item["id"]]["path"]).read_text())
            compiled_record = next(row for row in records if row["item_id"] == item["id"])
            plain[item["id"]] = plain_record
            compiled[item["id"]] = {"score": {
                "automated_pass": compiled_record["benchmark_pass"]}}
        summary["paired_vs_frozen_plain"] = _paired_summary(plain, compiled)
    _write_json(output_dir / "summary.json", summary)
    return summary


def run(suite_path, output_dir, checkpoint_pattern, plain_root, device="mps",
        seed=0, temperature=0.0, limit=7000, prepare_only=False, rehash=False):
    output_dir = Path(output_dir)
    suite = load_suite(Path(suite_path))
    manifest = prepare_manifest(suite_path, output_dir, checkpoint_pattern, plain_root,
                                device, seed, temperature, limit, rehash)
    if prepare_only:
        return _summary(output_dir, suite, manifest)
    config = manifest["experiment"]
    completed = 0
    for cycles in sorted({item["cycles"] for item in suite["items"]}):
        checkpoint = Path(checkpoint_pattern.format(cycles=cycles)).resolve()
        items = [item for item in suite["items"] if item["cycles"] == cycles]
        pending = []
        for item in items:
            item_dir = output_dir / "items" / item["id"]
            identity = _identity(manifest, item)
            record = _load_complete(item_dir, identity)
            if record is None:
                pending.append((item, item_dir, identity))
            else:
                completed += 1
                print(json.dumps({"item": item["id"], "completed": completed,
                                  "items": len(suite["items"]), "resumed": True,
                                  "system_pass": record["system_pass"],
                                  "benchmark_pass": record["benchmark_pass"]}), flush=True)
        if not pending:
            continue
        decoder = CompiledDecoder(checkpoint, device)
        try:
            for item, item_dir, identity in pending:
                run_path = item_dir / "run.json"
                if run_path.exists():
                    run_record = json.loads(run_path.read_text())
                    _validate_run(run_record, checkpoint, item, config)
                else:
                    print(json.dumps({"starting": item["id"], "cycles": cycles,
                                      "completed": completed,
                                      "items": len(suite["items"])}), flush=True)
                    decoder.decode(item["objective"], run_path, seed=seed, limit=limit,
                                   temperature=temperature, stream=False)
                record = _postprocess(item, item_dir, checkpoint, decoder, identity)
                completed += 1
                summary = _summary(output_dir, suite, manifest)
                print(json.dumps({"item": item["id"], "completed": completed,
                                  "items": len(suite["items"]),
                                  "system_pass": record["system_pass"],
                                  "benchmark_pass": record["benchmark_pass"],
                                  "chain_pass_so_far": summary["chain_pass"],
                                  "seconds": record["seconds"]}), flush=True)
        finally:
            del decoder
            gc.collect()
            if device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
    return _summary(output_dir, suite, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint-pattern",
                        default="build/babyagi-2b-ledger-pointer-b256-m{cycles}")
    parser.add_argument("--plain-root", type=Path,
                        default=Path("results/end-to-end-v3-b256"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=7000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rehash", action="store_true")
    args = parser.parse_args()
    summary = run(args.suite, args.output_dir, args.checkpoint_pattern,
                  args.plain_root, args.device, args.seed, args.temperature,
                  args.limit, args.prepare_only, args.rehash)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"),
                      "completed": summary["items_completed"],
                      "items": summary["items_total"],
                      "system_pass": summary["system_pass"],
                      "chain_pass": summary["chain_pass"]}, indent=2))


if __name__ == "__main__":
    main()
