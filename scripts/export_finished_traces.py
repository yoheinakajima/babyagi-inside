"""Export the ten frozen live traces into the public compact schema.

This is a CPU-only transformation over completed JSON artifacts. It does not
load a checkpoint, import a model framework, decode tokens, or rerun scoring.
With --sweep, the same ten items are read from a qfs.compiled_sweep output
directory (items/<id>/) instead of the archived result layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TRACE_LOCATIONS = {
    "v3-arithmetic-add-2": "compiled",
    "v3-arithmetic-add-3": "compiled",
    "v3-arithmetic-add-5": "compiled",
    "v3-arithmetic-add-8": "compiled",
    "v3-arithmetic-multiply-2": "compiled-host-gains",
    "v3-arithmetic-multiply-3": "compiled-control-checks",
    "v3-arithmetic-multiply-5": "compiled-control-checks",
    "v3-arithmetic-multiply-8": "compiled-host-gains",
    "v3-structured-even-8": "compiled-host-gains",
    "v3-structured-odd-8": "compiled-host-gains",
}


def load(path):
    return json.loads(Path(path).read_text())


def result_values(evaluation):
    values = []
    for phase in evaluation["phases"]:
        if phase["kind"] != "execute":
            continue
        calls = [call for call in phase["calls"]
                 if call["function"] == "write_file"
                 and call["parameters"].get("name") == "result.json"]
        if not calls:
            continue
        payload = json.loads(calls[-1]["parameters"]["text"])
        values.append(payload["result"])
    return values


def emitter(stage):
    if stage["model_generated"]:
        return "mixed:controller+qwen+qwenFS"
    if stage["filesystem_forced"]:
        return "controller+qwenFS"
    return "controller"


def artifact_paths(trace_id, source_root, sweep=False):
    if sweep:
        directory = Path(source_root) / "items" / trace_id
        return (directory / "run.json", directory / "evaluation.json",
                directory / "account.json")
    prefix = Path(source_root) / TRACE_LOCATIONS[trace_id] / trace_id
    return (Path(f"{prefix}-run.json"), Path(f"{prefix}-evaluation.json"),
            Path(f"{prefix}-account.json"))


def export_trace(trace_id, source_root, benchmark_items, sweep=False):
    paths = artifact_paths(trace_id, source_root, sweep)
    run, evaluation, accounting = map(load, paths)
    item = benchmark_items[trace_id]
    for path, value in zip(paths[1:], (evaluation, accounting)):
        if "error" in value:
            raise SystemExit(f"{trace_id}: {path} records an error: {value['error']}")

    ledger_text = evaluation["latest_files"]["ledger.json"]["text"]
    ledger = json.loads(ledger_text)
    expected = item["rubric"]["expected_results"]
    if sweep:
        # The sweep's own scored results and verdict (qfs.benchmark normalizes values).
        score = load(paths[0].parent / "score.json")
        results, passed = score["actual_results"], score["automated_pass"]
    else:
        results = result_values(evaluation)
        passed = len(results) == len(expected) and results == expected
    totals = accounting["totals"]
    stages = [{
        "cycle": stage["cycle"],
        "phase": stage["kind"],
        "emitter": emitter(stage),
        "start": None,
        "end": None,
    } for stage in accounting["stages"]]

    return {
        "trace_id": trace_id,
        "cycles": item["cycles"],
        "objective": run["objective"],
        "ledger": ledger,
        "results": results,
        "expected": expected,
        "pass": passed,
        "tokens": {
            "total": totals["emitted_tokens"],
            "forced": totals["forced_total"],
            "qwenfs": totals["filesystem_forced"],
            "qwen_reasoning": totals["model_reasoning"],
            "qwen_payload": totals["model_payload"] + totals["model_tool_closure"],
        },
        "stages": stages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path,
                        help="finished end-to-end-v3-b256 result root, or a sweep "
                             "directory with --sweep")
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sweep", action="store_true",
                        help="source_root is a qfs.compiled_sweep output directory")
    args = parser.parse_args()

    benchmark = load(args.benchmark)
    items = {item["id"]: item for item in benchmark["items"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for trace_id in TRACE_LOCATIONS:
        value = export_trace(trace_id, args.source_root, items, args.sweep)
        output = args.output_dir / f"{trace_id}.json"
        output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        print(f"{trace_id}: {'pass' if value['pass'] else 'fail'}")


if __name__ == "__main__":
    main()
