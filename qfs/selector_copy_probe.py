"""Probe a compile-friendly selector that copies the chosen ledger task."""
import argparse
import json
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.ceiling import _json_values, generate_chat


def parse_selected(text):
    for value in reversed(list(_json_values(text))):
        if isinstance(value, dict) and isinstance(value.get("selected_task"), str):
            return value["selected_task"].strip()
    return None


def normalized(text):
    return re.sub(r"\s+", " ", text.strip()).casefold()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to(device).eval()
    reports = []
    for path in sorted(args.trace_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        record = json.loads(path.read_text())
        completed = []
        for cycle in record["trace"]["cycles"]:
            candidates = cycle["create"]["candidates"]
            prompt = (
                f"Original objective:\n{record['objective']}\n\n"
                f"Current result: {json.dumps(cycle['state_in'], ensure_ascii=False)}\n"
                f"Completed tasks: {json.dumps(completed, ensure_ascii=False)}\n"
                f"Candidate 1: {candidates[0]}\nCandidate 2: {candidates[1]}\n\n"
                "Select the single next undone task. Reject completed and out-of-order work. "
                "Do not execute or paraphrase it. End with `SELECT_JSON: ` followed by one object "
                'shaped {"selected_task":"exact candidate text"}. Put nothing after the object.'
            )
            generation = generate_chat(model, tok, prompt, device, thinking=False,
                                       max_new_tokens=96)
            selected_text = parse_selected(generation["text"])
            matched = None
            if selected_text is not None:
                hits = [index for index, candidate in enumerate(candidates, 1)
                        if normalized(selected_text) == normalized(candidate)]
                matched = hits[0] if len(hits) == 1 else None
            expected = cycle["choose"]["expected"]
            reports.append({"item_id": record["item_id"], "cycle": cycle["cycle"],
                            "candidates": candidates, "expected": expected,
                            "prompt": prompt, "generation": generation,
                            "selected_text": selected_text, "matched": matched,
                            "parse_ok": selected_text is not None,
                            "exact_candidate": matched is not None,
                            "correct": matched == expected})
            completed.append(cycle["execute"]["selected_task"])
            print(json.dumps({"item": record["item_id"], "cycle": cycle["cycle"],
                              "expected": expected, "matched": matched,
                              "correct": matched == expected}), flush=True)
    result = {"schema_version": 1, "probe": "ledger-selector-copy",
              "source": str(args.trace_dir.resolve()),
              "model": str(Path(args.base).resolve()), "device": device,
              "total": len(reports),
              "parse_pass": sum(r["parse_ok"] for r in reports),
              "exact_candidate": sum(r["exact_candidate"] for r in reports),
              "correct": sum(r["correct"] for r in reports),
              "task2_expected": sum(r["expected"] == 2 for r in reports),
              "task2_selected": sum(r["matched"] == 2 for r in reports),
              "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("total", "parse_pass", "exact_candidate", "correct",
                       "task2_expected", "task2_selected")}, indent=2))


if __name__ == "__main__":
    main()
