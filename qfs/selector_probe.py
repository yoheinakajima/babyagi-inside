"""Counterbalanced semantic probe for the typed file selector.

The circuit oracle proves that both legal tokens can pass the compiled gate.
This native-weight probe asks a different question: does Qwen choose the work
task when the work/verification contents swap IDs?  It deliberately reports a
failure when a model has positional preference; constrained decoding guarantees
validity, not semantic correctness.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.agent import program
from qfs.agent_native import run_native


CASES = {
    "work_first": {
        "objective": ("Original objective: append -cyan to ash. Task 1: Append -cyan to ash. "
                      "Task 2: Verify Task 1. Choose the task that performs the unfinished "
                      "objective."),
        "expected": "1",
    },
    "work_second": {
        "objective": ("Original objective: append -cyan to ash. Task 1: Verify Task 2. "
                      "Task 2: Append -cyan to ash. Choose the task that performs the unfinished "
                      "objective."),
        "expected": "2",
    },
}


def probe(base, output, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16).to(device).eval()
    priority = dict(program(tok, 1, "babyagi_typed")[2])
    # The regular stage follows two prior tool turns and begins by closing the
    # empty assistant turn. A standalone chat prefill is already closed.
    interstage = tok.encode("<|im_end|>", add_special_tokens=False)
    if priority["rom"][:len(interstage)] != interstage:
        raise ValueError("priority stage no longer has the expected interstage prefix")
    priority["rom"] = priority["rom"][len(interstage):]

    reports = []
    for name, case in CASES.items():
        trace = run_native(model, tok, [priority], case["objective"], device=device,
                           temperature=0.0)
        phase = trace["phases"][0]
        decision = phase["decision"]
        reports.append({"case": name, "expected": case["expected"],
                        "selected": decision["selected_text"],
                        "selected_file": phase["parameters"]["name"],
                        "allowed_logits": decision["native_allowed_logits"],
                        "pass": decision["selected_text"] == case["expected"]})
    result = {
        "schema_version": 1,
        "probe": "counterbalanced-typed-file-choice",
        "base": str(Path(base).resolve()),
        "protocol": "babyagi_typed",
        "compiled_controller_executed": False,
        "reports": reports,
        "pass": all(report["pass"] for report in reports),
        "interpretation": ("The typed gate guarantees a legal filename token. This probe tests "
                           "whether stock Qwen chooses that token by candidate content."),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("--output", default="results/selector-counterbalance.json")
    parser.add_argument("--device")
    args = parser.parse_args()
    result = probe(args.base, args.output, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
