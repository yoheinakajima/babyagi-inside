"""Counterbalanced semantic and calibration probes for the typed selector.

The circuit oracle proves that both legal tokens can pass the compiled gate.
This native-weight probe asks a different question: does Qwen choose the work
task when work and distractor contents swap IDs? Twenty-five scenarios are
tested in both orders. Neutral prompts estimate the constant logit preference
for token 1, and the report also shows post-hoc constant-offset accuracy.
"""
import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.agent import program
from qfs.agent_native import run_native


SCENARIOS = (
    ("Append -cyan to ash; current value is ash.", "Append -cyan to ash", "Verify ash"),
    ("Uppercase moss; current value is moss.", "Uppercase moss", "Count letters in moss"),
    ("Add 5 to 7; current value is 7.", "Add 5 to 7", "Verify that 7 is numeric"),
    ("Multiply 12 by 3; current value is 12.", "Multiply 12 by 3", "Recompute completed addition"),
    ("Sort [3,1,2] ascending.", "Sort [3,1,2] ascending", "Check that the list has three items"),
    ("Remove duplicates from [1,1,3].", "Remove duplicate 1", "Sort the already sorted list"),
    ("Keep odd values from [1,2,3].", "Filter out even values", "Verify the JSON brackets"),
    ("Replace the space in small moon with an underscore.",
     "Replace the space with an underscore", "Uppercase the text before replacing the space"),
    ("Prefix ID: to CODE.", "Prefix ID: to CODE", "Append an unrelated suffix"),
    ("Draft a headline for Luma.", "Draft the Luma headline", "Verify a headline not yet drafted"),
    ("Write the body after the headline is complete.", "Write the body", "Rewrite the headline"),
    ("Add a CTA after headline and body are complete.", "Write the CTA", "Verify the headline"),
    ("Compress the supplied facts to one sentence.",
     "Compress the facts to one sentence", "Add an unsupported fact"),
    ("Remove unsupported claims from the draft.",
     "Remove unsupported claims", "Publish the unverified draft"),
    ("Make three lines start with D, A, Y.",
     "Revise line initials to D, A, Y", "Add a fourth line"),
    ("Divide 10 by 2; current value is 10.", "Divide 10 by 2", "Square 10 first"),
    ("Square 5; current value is 5.", "Square 5", "Subtract the later operand"),
    ("Subtract 8 from 36; current value is 36.", "Subtract 8 from 36", "Add 5 again"),
    ("Map each remaining word to uppercase.",
     "Uppercase each remaining word", "Restore filtered-out words"),
    ("Output only record names after filtering.",
     "Project each retained record to its name", "Sort the discarded records"),
    ("Append -east after -north; current value is cinder-north.",
     "Append -east to cinder-north", "Append -north again"),
    ("Append -home after -west; current value ends in -west.",
     "Append -home", "Verify the earlier -east step"),
    ("Turn the selected outline into actual copy.",
     "Write the actual copy", "Describe how to write the copy"),
    ("Correct the numeric result 19 by multiplying by 2.",
     "Multiply 19 by 2", "Verify the completed addition"),
    ("Return valid JSON for the transformed array.",
     "Serialize the transformed array as JSON", "Explain what JSON means"),
)

NEUTRAL_PROMPTS = (
    "Calibration only. Task 1 is option alpha. Task 2 is option beta. Both are equally valid.",
    "Calibration only. Task 1 is left. Task 2 is right. Neither has semantic priority.",
    "Calibration only. Task 1 is red. Task 2 is blue. Treat them as equally appropriate.",
    "Calibration only. Task 1 says A. Task 2 says B. There is no correct content choice.",
    "Calibration only. Task 1 is candidate X. Task 2 is candidate Y. Both are acceptable.",
    "Calibration only. Task 1 is north. Task 2 is south. They have equal status.",
    "Calibration only. Task 1 is circle. Task 2 is square. Do not prefer either meaning.",
    "Calibration only. Task 1 is first label. Task 2 is second label. Content is neutral.",
    "Calibration only. Task 1 is foo. Task 2 is bar. Both choices are equally useful.",
    "Calibration only. Task 1 is item P. Task 2 is item Q. No task is better.",
)


def _standalone_priority(tok):
    priority = dict(program(tok, 1, "babyagi_typed")[2])
    interstage = tok.encode("<|im_end|>", add_special_tokens=False)
    if priority["rom"][:len(interstage)] != interstage:
        raise ValueError("priority stage no longer has the expected interstage prefix")
    priority["rom"] = priority["rom"][len(interstage):]
    return priority


def _decision(model, tok, stage, objective, device):
    trace = run_native(model, tok, [stage], objective, device=device, temperature=0.0)
    phase = trace["phases"][0]
    decision = phase["decision"]
    logits = decision["native_allowed_logits"]
    allowed = decision["allowed_tokens"]
    one, two = float(logits[str(allowed[0])]), float(logits[str(allowed[1])])
    return {"selected": decision["selected_text"],
            "selected_file": phase["parameters"]["name"],
            "logit_1": one, "logit_2": two, "delta_1_minus_2": one - two}


def probe(base, output, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16).to(device).eval()
    priority = _standalone_priority(tok)

    neutral = [{"probe": index, **_decision(model, tok, priority, objective, device)}
               for index, objective in enumerate(NEUTRAL_PROMPTS)]
    bias = mean(report["delta_1_minus_2"] for report in neutral)

    reports = []
    for scenario, (objective, work, distractor) in enumerate(SCENARIOS):
        for order in ("work_first", "work_second"):
            if order == "work_first":
                task1, task2, expected = work, distractor, "1"
            else:
                task1, task2, expected = distractor, work, "2"
            prompt = (f"Original objective and current state: {objective}\n"
                      f"Task 1: {task1}\nTask 2: {task2}\n"
                      "Choose the task that directly performs the next undone objective step.")
            decision = _decision(model, tok, priority, prompt, device)
            calibrated_delta = decision["delta_1_minus_2"] - bias
            calibrated = "1" if calibrated_delta >= 0 else "2"
            reports.append({"scenario": scenario, "order": order,
                            "objective": objective, "task1": task1, "task2": task2,
                            "expected": expected, **decision,
                            "raw_pass": decision["selected"] == expected,
                            "calibrated_delta": calibrated_delta,
                            "calibrated_selected": calibrated,
                            "calibrated_pass": calibrated == expected})

    result = {
        "schema_version": 2,
        "probe": "counterbalanced-typed-file-choice",
        "base": str(Path(base).resolve()),
        "protocol": "babyagi_typed",
        "compiled_controller_executed": False,
        "counts": {"neutral": len(neutral), "counterbalanced": len(reports)},
        "neutral": {"mean_delta_1_minus_2": bias, "reports": neutral},
        "summary": {
            "raw_accuracy": sum(report["raw_pass"] for report in reports) / len(reports),
            "calibrated_accuracy": (sum(report["calibrated_pass"] for report in reports)
                                    / len(reports)),
            "raw_task2_selection_rate": (sum(report["selected"] == "2" for report in reports)
                                         / len(reports)),
            "calibrated_task2_selection_rate": (
                sum(report["calibrated_selected"] == "2" for report in reports) / len(reports)),
            "work_first_accuracy": (sum(report["raw_pass"] for report in reports
                                        if report["order"] == "work_first")
                                    / (len(reports) / 2)),
            "work_second_accuracy": (sum(report["raw_pass"] for report in reports
                                         if report["order"] == "work_second")
                                     / (len(reports) / 2)),
        },
        "reports": reports,
        "interpretation": ("The typed gate guarantees a legal filename token. Raw accuracy tests "
                           "semantic choice; calibrated accuracy subtracts only the neutral "
                           "constant logit bias and does not change model weights."),
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
    print(json.dumps({"output": args.output, "counts": result["counts"],
                      "neutral_bias": result["neutral"]["mean_delta_1_minus_2"],
                      **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
