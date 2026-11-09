"""E12 Qwen3.5-4B port: frozen comparisons, builds, parity, the live sweep, and analysis.

    python -m harness.port freeze           # writes provenance/e12-frozen-comparisons.json
    python -m harness.port build CYCLES     # step A1-mN: build, tree digest, rebuild identity
    python -m harness.port parity CYCLES    # step A3-mN: agent-inactive parity against native 4B
    python -m harness.port sweep            # step B: the frozen 32-item live compiled sweep
    python -m harness.port analyze          # step B-analysis: paired comparisons and failures

The oracle (step A2) is harness.oracle. Every step calls babyagi-inside's own code from the
pinned vendor/ checkout: `qfs.agent` builds, `tests.test_agent_parity.check` measures
parity, and `qfs.compiled_sweep.run` runs the sweep. The settings are frozen in
experiments/E12-qwen35-4b-port/prereg.md. Outputs go to LAB_OUTPUT_DIR under `lab run`;
the sweep itself writes to SWEEP_DIR so an interrupted sweep can resume. Each command exits
0 whenever it executed; its verdict is in metrics.json.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from harness import plain_baseline, scale_screen
from harness.scorer import load_suite
from harness.upstream import ROOT, SUITE_V3, module
from lab import runlog
from lab.repo import sha256_file

EXPERIMENT = "E12"
FROZEN = ROOT / "provenance" / "e12-frozen-comparisons.json"
# Frozen in prereg.md (Arms, Seeds and decoding).
BASE = "models/Qwen3.5-4B"
PATTERN = "build/babyagi-4b-ledger-pointer-b256-m{cycles}"
SWEEP_DIR = ROOT / "build" / "e12-sweep-4b"
SETTINGS = {"protocol": "babyagi_ledger_pointer", "think_tokens": 256, "device": "mps",
            "seed": 0, "temperature": 0.0, "limit": 7000}
CYCLES = (2, 3, 5, 8)
NATIVE_HIDDEN, MERGED_HIDDEN = 2560, 4096
MACHINE_MEMORY_GB = 32
PARITY_MAX_KL = 0.001  # E7's smoke target, in nats
# Frozen comparison sources (prereg.md, Arms).
E9_4B_RUN = "E9-20261107T051231Z-e18424"
E9_4B_RESULTS = ROOT / "experiments" / "E9-model-scale" / "runs" / E9_4B_RUN / "results"
# What a re-hash of the plain records must reproduce; source.checkout_commit only dates
# the lab checkout at freeze time (as in harness.sweep_validate).
PLAIN_FROZEN = ("suite_sha256", "chain_pass", "records_sha256", "records")


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _output(args) -> Path:
    if args.output is None:
        sys.exit("--output is required outside `lab run`")
    args.output.mkdir(parents=True, exist_ok=True)
    return args.output


def checkpoint(cycles: int) -> Path:
    return ROOT / PATTERN.format(cycles=cycles)


# ---------------------------------------------------------------- freeze


def freeze() -> dict:
    """Digests of the three frozen comparisons: plain 4B, E9's 4B host arm, compiled 2B."""
    suite = load_suite()
    ids = [item["id"] for item in suite["items"]]
    plain = plain_baseline.build_manifest(E9_4B_RESULTS)
    host = {}
    for item_id in ids:
        path = E9_4B_RESULTS / "host" / f"{item_id}.json"
        record = json.loads(path.read_text())
        if record["identity"]["arm"] != "host" or record["identity"]["item_id"] != item_id:
            raise ValueError(f"unexpected E9 host record identity: {path}")
        host[item_id] = {"sha256": sha256_file(path),
                         "chain_pass": record["score"]["automated_pass"],
                         "plan_exact": record["plan"]["exact"],
                         "actual_results": record["actual_results"]}
    frozen_2b = plain_baseline.load_manifest()
    sweep_dir = (Path.home() / frozen_2b["source"]["checkout"].removeprefix("~/")
                 / frozen_2b["sweep"]["dir"])
    for name, key in (("manifest.json", "manifest_sha256"), ("summary.json", "summary_sha256")):
        if sha256_file(sweep_dir / name) != frozen_2b["sweep"][key]:
            raise ValueError(f"{sweep_dir / name} differs from provenance/plain-baseline.json")
    summary_2b = json.loads((sweep_dir / "summary.json").read_text())
    manifest_2b = json.loads((sweep_dir / "manifest.json").read_text())
    compiled_2b = {row["item_id"]: {"chain_pass": row["benchmark_pass"],
                                    "system_pass": row["system_pass"]}
                   for row in summary_2b["records"]}
    return {
        "schema_version": 1,
        "description": "E12's frozen comparisons. plain_4b: E9's plain Qwen3.5-4B records "
                       "(harness.plain_baseline.build_manifest), the sweep's --plain-root. "
                       "host_4b: E9's host-simulated Qwen3.5-4B records. compiled_2b: the "
                       "2B live compiled sweep that E0 validated (provenance/"
                       "plain-baseline.json), with its runner source digests. records_sha256 "
                       "= sha256 of {item_id: sha256} as sorted compact JSON.",
        "plain_4b": plain,
        "host_4b": {"source": {"run_id": E9_4B_RUN,
                               "results_root": str(E9_4B_RESULTS.relative_to(ROOT)),
                               "records_dir": "host"},
                    "chain_pass": sum(row["chain_pass"] for row in host.values()),
                    "records_sha256": plain_baseline.records_digest(host),
                    "records": host},
        "compiled_2b": {"source": {"checkout": frozen_2b["source"]["checkout"],
                                   "dir": frozen_2b["sweep"]["dir"]},
                        "experiment_sha256": frozen_2b["sweep"]["experiment_sha256"],
                        "summary_sha256": frozen_2b["sweep"]["summary_sha256"],
                        "source_sha256": manifest_2b["source_sha256"],
                        "chain_pass": summary_2b["chain_pass"],
                        "system_pass": summary_2b["system_pass"],
                        "records": compiled_2b},
    }


def load_frozen(path: Path = FROZEN) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------- build


def parse_time(stderr: str) -> dict:
    """Wall seconds and peak RSS from macOS `/usr/bin/time -l` output."""
    real = re.search(r"([\d.]+) real", stderr)
    rss = re.search(r"(\d+)\s+maximum resident set size", stderr)
    return {"wall_seconds": float(real[1]) if real else None,
            "peak_rss_gb": round(int(rss[1]) / 1e9, 2) if rss else None}


def _timed_build(cycles: int, dst: Path) -> dict:
    command = ["/usr/bin/time", "-l", sys.executable, "-m", "qfs.agent", BASE,
               str(dst.relative_to(ROOT)), "--cycles", str(cycles),
               "--protocol", SETTINGS["protocol"],
               "--think-tokens", str(SETTINGS["think_tokens"]), "--device", SETTINGS["device"]]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    return {"returncode": done.returncode, **parse_time(done.stderr)}


def build(cycles: int, output: Path) -> dict:
    """Build one checkpoint, digest it, rebuild it into a scratch directory, and compare."""
    target = checkpoint(cycles)
    if target.exists():
        raise FileExistsError(f"{target} exists; step A1 builds from nothing")
    compiled_sweep = module("qfs.compiled_sweep")
    first = _timed_build(cycles, target)
    if first["returncode"]:
        return {"cycles": cycles, "build_pass": False, "first": first}
    manifest = compiled_sweep._checkpoint_manifest(target)
    _write(output / "checkpoint-manifest.json", manifest)
    meta = json.loads((target / "agent.json").read_text())

    repeat_dir = ROOT / "build" / f"e12-repeat-m{cycles}"
    if repeat_dir.exists():  # this step's own scratch, left by an interrupted run
        shutil.rmtree(repeat_dir)
    second = _timed_build(cycles, repeat_dir)
    repeat = compiled_sweep._checkpoint_manifest(repeat_dir) if not second["returncode"] else None
    if repeat_dir.exists():
        shutil.rmtree(repeat_dir)
    differing = []
    if repeat is not None:
        a = {row["path"]: row["sha256"] for row in manifest["files"]}
        b = {row["path"]: row["sha256"] for row in repeat["files"]}
        differing = sorted(path for path in a.keys() | b.keys() if a.get(path) != b.get(path))
    checks = {
        "builds_completed": not second["returncode"],
        "repeat_identical": repeat is not None and repeat["sha256"] == manifest["sha256"],
        "metadata": (meta["cycles"] == cycles and meta["protocol"] == SETTINGS["protocol"]
                     and meta["native_hidden"] == NATIVE_HIDDEN
                     and meta["merged_hidden"] == MERGED_HIDDEN),
        "fits_in_memory": (first["peak_rss_gb"] is not None
                           and second["peak_rss_gb"] is not None
                           and max(first["peak_rss_gb"], second["peak_rss_gb"])
                           < MACHINE_MEMORY_GB),
    }
    return {"cycles": cycles, "build_pass": all(checks.values()), "checks": checks,
            "tree_sha256": manifest["sha256"], "bytes": manifest["bytes"],
            "files": len(manifest["files"]),
            "repeat_tree_sha256": repeat["sha256"] if repeat else None,
            "differing_files": differing, "scratch": meta["scratch"],
            "wall_seconds": first["wall_seconds"], "peak_rss_gb": first["peak_rss_gb"],
            "repeat_wall_seconds": second["wall_seconds"],
            "repeat_peak_rss_gb": second["peak_rss_gb"]}


# ---------------------------------------------------------------- parity


def printed_result(text: str) -> dict:
    """The result JSON tests.test_agent_parity.check prints before its pass assertion."""
    start = text.index("\n{", text.index("loading native source")) + 1
    return json.JSONDecoder().raw_decode(text[start:])[0]


def parity_gate(result: dict) -> dict:
    kl = result["kl_native_to_compiled"]
    max_kl = max(kl)  # a NaN anywhere fails the finite check below, wherever max() skips it
    checks = {"max_kl": all(math.isfinite(value) for value in kl) and max_kl <= PARITY_MAX_KL,
              "top1": result["top1_agreement"] == result["positions"]}
    return {"parity_pass": all(checks.values()), "checks": checks, "max_kl": max_kl,
            "kl_native_to_compiled": result["kl_native_to_compiled"],
            "top1_agreement": f"{result['top1_agreement']}/{result['positions']}",
            "upstream_pass": result["pass"]}


def parity(cycles: int, output: Path) -> dict:
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            module("tests.test_agent_parity").check(str(checkpoint(cycles)), str(ROOT / BASE),
                                                    SETTINGS["device"])
    except AssertionError:
        pass  # upstream's own 0.03 assertion; the verdict below uses the E7 threshold
    print(captured.getvalue(), end="")
    result = printed_result(captured.getvalue())
    _write(output / "parity.json", result)
    return {"cycles": cycles, **parity_gate(result)}


# ---------------------------------------------------------------- sweep


def a1_digests(records: list[dict]) -> dict:
    """Tree digest of each checkpoint from its latest ok A1 run (A1-m2, A1-m3, ...)."""
    digests = {}
    for record in records:
        if (record.get("experiment_id") == EXPERIMENT and record.get("step") == "A1"
                and record.get("status") == "ok" and record["metrics"].get("build_pass")):
            digests[str(record["metrics"]["cycles"])] = record["metrics"]["tree_sha256"]
    return digests


def _bindings(manifest: dict, frozen: dict, digests: dict) -> dict:
    plain = {item_id: row["sha256"] for item_id, row in manifest["plain"]["records"].items()}
    return {
        "checkpoints_match_a1": {cycles: row["sha256"] == digests.get(cycles)
                                 for cycles, row in manifest["checkpoints"].items()},
        "plain_matches_frozen": plain == {item_id: row["sha256"] for item_id, row
                                          in frozen["plain_4b"]["records"].items()},
        "sources_match_2b_sweep": manifest["source_sha256"]
                                  == frozen["compiled_2b"]["source_sha256"],
    }


def _bound(bindings: dict) -> bool:
    return (all(bindings["checkpoints_match_a1"].values()) and bindings["plain_matches_frozen"]
            and bindings["sources_match_2b_sweep"])


def sweep(output: Path) -> dict:
    frozen = load_frozen()
    fresh = plain_baseline.build_manifest(E9_4B_RESULTS)
    if any(fresh[key] != frozen["plain_4b"][key] for key in PLAIN_FROZEN):
        raise ValueError("E9's plain 4B records differ from provenance/e12-frozen-comparisons.json")
    compiled_sweep = module("qfs.compiled_sweep")
    plain_root = ROOT / frozen["plain_4b"]["source"]["results_root"]
    arguments = (SUITE_V3, SWEEP_DIR, PATTERN, plain_root, SETTINGS["device"], SETTINGS["seed"],
                 SETTINGS["temperature"], SETTINGS["limit"])
    manifest = compiled_sweep.prepare_manifest(*arguments)
    bindings = _bindings(manifest, frozen, a1_digests(runlog.read(ROOT)))
    if not _bound(bindings):  # stop before any decode
        return {"lab_status": "invalid", "invalid_reason": "sweep inputs are not the frozen ones",
                "bindings": bindings}
    summary = compiled_sweep.run(*arguments)
    records = summary["records"]
    tokens = sum(record["tokens"] for record in records)
    seconds = sum(record["seconds"] or 0 for record in records)
    return {
        "experiment_sha256": manifest["experiment_sha256"],
        "manifest_sha256": sha256_file(SWEEP_DIR / "manifest.json"),
        "summary_sha256": sha256_file(SWEEP_DIR / "summary.json"),
        "bindings": bindings, "complete": summary["complete"],
        "items_completed": summary["items_completed"],
        "system_pass": summary["system_pass"], "chain_pass": summary["chain_pass"],
        "chain_wilson_95": summary["chain_wilson_95"],
        "step_accuracy": summary["step_accuracy"],
        "correct_steps": summary["correct_steps"], "total_steps": summary["total_steps"],
        "plan_parse_ok": summary["plan_parse_ok"], "plan_exact": summary["plan_exact"],
        "plan_semantic_exact": summary["plan_semantic_exact"],
        "by_category": summary["by_category"], "by_length": summary["by_length"],
        "paired_vs_frozen_plain": summary.get("paired_vs_frozen_plain"),
        "halted": sum(record["halted"] for record in records),
        "tokens": tokens, "decode_seconds": round(seconds, 1),
        "tokens_per_sec": round(tokens / seconds, 4) if seconds else None,
    }


# ---------------------------------------------------------------- analyze


def verdict(ci: list[float]) -> str:
    """Preregistered reading of a paired difference's item-bootstrap 95% CI."""
    if ci[0] > 0:
        return "SUPPORTED"
    if ci[1] < 0:
        return "CONTRADICTED"
    return "INCONCLUSIVE"


def _verdicts(rows: dict, key: str) -> dict:
    return {item_id: {"score": {"automated_pass": bool(row[key])}}
            for item_id, row in rows.items()}


def chain_verdict(record: dict) -> bool:
    """Prereg (Analysis plan): a chain passes only if it is scored correct and reached HALT."""
    return bool(record["benchmark_pass"] and record["halted"])


def _failure(record: dict, score: dict) -> dict:
    failed_checks = [name for name, ok in record["system_checks"].items() if not ok]
    steps = [check["pass"] for check in score["automated_checks"]]
    return {"item_id": record["item_id"], "category": record["category"],
            "cycles": record["cycles"], "system_failures": failed_checks,
            "chain_pass": chain_verdict(record),
            "first_failed_step": next((i + 1 for i, ok in enumerate(steps) if not ok), None),
            "correct_steps": f"{record['correct_steps']}/{record['total_steps']}",
            "plan_semantic_exact": record["plan_semantic_exact"]}


def analyze(output: Path) -> dict:
    frozen = load_frozen()
    runs = [record for record in runlog.read(ROOT) if record.get("experiment_id") == EXPERIMENT
            and record.get("step") == "B" and record.get("status") == "ok"]
    if not runs:
        raise ValueError("no ok E12 step B run in runs.jsonl")
    logged = runs[-1]["metrics"]
    for name, key in (("manifest.json", "manifest_sha256"), ("summary.json", "summary_sha256")):
        if sha256_file(SWEEP_DIR / name) != logged[key]:
            raise ValueError(f"{SWEEP_DIR / name} changed after step B {runs[-1]['run_id']}")
    summary = json.loads((SWEEP_DIR / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("the sweep is not complete")
    compiled = {row["item_id"]: row for row in summary["records"]}
    scores = {item_id: json.loads((SWEEP_DIR / "items" / item_id / "score.json").read_text())
              for item_id in compiled}
    host = frozen["host_4b"]["records"]
    chain = scale_screen._chain
    compiled_v = {i: {"score": {"automated_pass": chain_verdict(row)}}
                  for i, row in compiled.items()}
    paired = {
        "compiled_4b_minus_plain_4b": scale_screen._paired(
            _verdicts(frozen["plain_4b"]["records"], "chain_pass"), compiled_v, chain),
        "compiled_4b_minus_compiled_2b": scale_screen._paired(
            _verdicts(frozen["compiled_2b"]["records"], "chain_pass"), compiled_v, chain),
        "compiled_4b_minus_host_4b": scale_screen._paired(
            _verdicts(host, "chain_pass"), compiled_v, chain),
    }
    agreement = {
        "chain_verdict_same": sum(chain_verdict(compiled[i]) == host[i]["chain_pass"]
                                  for i in compiled),
        "actual_results_identical": sum(scores[i]["actual_results"] == host[i]["actual_results"]
                                        for i in compiled),
        "differing_items": sorted(i for i in compiled
                                  if scores[i]["actual_results"] != host[i]["actual_results"]),
    }
    failures = [_failure(compiled[i], scores[i]) for i in compiled
                if not compiled[i]["system_pass"] or not chain_verdict(compiled[i])]
    primary = paired["compiled_4b_minus_plain_4b"]
    return {
        "items": len(compiled),
        "chain_pass": sum(chain_verdict(row) for row in compiled.values()),
        "scorer_pass_without_halt": sorted(i for i, row in compiled.items()
                                           if row["benchmark_pass"] and not row["halted"]),
        "sweep_chain_pass": summary["chain_pass"],
        "system_pass": summary["system_pass"],
        "hypothesis": verdict(primary["item_bootstrap_delta_95"]),
        "paired": paired, "host_4b_agreement": agreement,
        "system_failures": sum(bool(row["system_failures"]) for row in failures),
        "failures": failures,
        "summary_matches_sweep_paired": summary["paired_vs_frozen_plain"]["delta"]
                                        == primary["delta"],
    }


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=os.environ.get("LAB_OUTPUT_DIR"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    for name in ("build", "parity"):
        commands.add_parser(name).add_argument("cycles", type=int, choices=CYCLES)
    commands.add_parser("sweep")
    commands.add_parser("analyze")
    args = parser.parse_args()
    if args.command == "freeze":
        frozen = freeze()
        _write(FROZEN, frozen)
        print(f"wrote {FROZEN.relative_to(ROOT)}: plain 4B {frozen['plain_4b']['chain_pass']}/32, "
              f"host 4B {frozen['host_4b']['chain_pass']}/32, compiled 2B "
              f"{frozen['compiled_2b']['chain_pass']}/32")
        return
    output = _output(args)
    if args.command == "build":
        metrics = build(args.cycles, output)
    elif args.command == "parity":
        metrics = parity(args.cycles, output)
    elif args.command == "sweep":
        metrics = sweep(output)
    else:
        metrics = analyze(output)
    _write(output / "metrics.json", metrics)
    print(json.dumps({key: value for key, value in metrics.items()
                      if key not in ("failures", "by_category", "by_length")}, indent=2))


if __name__ == "__main__":
    main()
