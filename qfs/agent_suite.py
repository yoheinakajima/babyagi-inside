"""Run a frozen benchmark through the stock-weight semantic harness.

This is a prompt-screening experiment, not evidence that the compiled
controller ran.  It loads Qwen once, executes every objective with the same
compiled-agent program, saves each full native trace, and produces an exact
machine-readable score summary.  Final claims about qwenFS and the controller
must come from ``qfs.agent_decode`` plus ``qfs.agent_evaluate``.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.agent import PROTOCOLS, program
from qfs.agent_native import run_native
from qfs.benchmark import extract_output, file_sha256, load_suite, score_item


def native_results(trace):
    """Return one semantic result per completed execute write."""
    values = []
    for phase in trace.get("phases", []):
        if phase.get("kind") != "execute" or phase.get("op") != "write":
            continue
        parameters = phase.get("parameters", {})
        if parameters.get("name") not in ("result.txt", "result.json") or "text" not in parameters:
            continue
        text = parameters["text"]
        if parameters.get("name") == "result.json":
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "result" in value:
                values.append(value["result"])
        else:
            values.append(extract_output(text))
    return values


def run_suite(base, suite_path, output_dir, protocol="babyagi_typed", device=None,
              temperature=0.0, top_p=0.9, seed=0, limit_per_write=240):
    suite_path, output_dir = Path(suite_path), Path(output_dir)
    suite = load_suite(suite_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16).to(device).eval()
    reports = []
    for item in suite["items"]:
        trace = run_native(
            model, tok, program(tok, item["cycles"], protocol), item["objective"],
            device=device, temperature=temperature, top_p=top_p, seed=seed,
            limit_per_write=limit_per_write)
        trace.update({"base": str(Path(base).resolve()), "cycles": item["cycles"],
                      "protocol": protocol, "device": device,
                      "benchmark_id": suite["benchmark_id"], "item_id": item["id"]})
        trace_path = output_dir / f"{item['id']}.json"
        trace_path.write_text(json.dumps(trace, indent=2) + "\n")
        actual = native_results(trace)
        score = score_item(item, actual)
        reports.append({"item_id": item["id"], "category": item["category"],
                        "cycles": item["cycles"], "trace": trace_path.name,
                        "actual_results": actual, "score": score,
                        "stopped": trace["stopped"],
                        "stages_completed": trace["stages_completed"],
                        "stages_expected": trace["stages_expected"],
                        "forward_seconds": trace["forward_seconds"]})
        print(json.dumps({"item": item["id"], "automated": score["automated_pass"],
                          "pass": score["pass"], "results": len(actual)}), flush=True)

    automated_pass = sum(report["score"]["automated_pass"] for report in reports)
    summary = {
        "schema_version": 1,
        "harness": "native-semantic-simulator",
        "compiled_controller_executed": False,
        "benchmark": {"path": str(suite_path), "id": suite["benchmark_id"],
                      "status": suite["status"], "sha256": file_sha256(suite_path)},
        "candidate": {"base": str(Path(base).resolve()), "protocol": protocol,
                      "program_sha256": hashlib.sha256(
                          Path(__file__).with_name("agent.py").read_bytes()).hexdigest()},
        "decoding": {"temperature": temperature, "top_p": top_p, "seed": seed,
                     "mode": "greedy" if temperature <= 0 else "sampled"},
        "counts": {"items": len(reports), "automated_pass": automated_pass,
                   "automated_fail": len(reports) - automated_pass,
                   "manual_pending": sum(report["score"]["pass"] is None
                                         for report in reports)},
        "reports": reports,
        "limitations": ["host advances phases and simulates visible filesystem responses",
                        "only compiled traces validate controller, qwenFS, and HALT"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="babyagi_typed")
    parser.add_argument("--device")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-per-write", type=int, default=240)
    args = parser.parse_args()
    summary = run_suite(args.base, args.suite, args.output_dir, args.protocol,
                        args.device, args.temperature, args.top_p, args.seed,
                        args.limit_per_write)
    print(json.dumps({"output": str(args.output_dir / "summary.json"),
                      **summary["counts"]}))


if __name__ == "__main__":
    main()
