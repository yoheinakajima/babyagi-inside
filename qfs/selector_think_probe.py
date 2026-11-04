"""Replay saved counterbalanced ledger decisions with bounded selector thought."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.ledger_protocol import bounded_score_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--think-tokens", type=int, default=32)
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
        for cycle in record["trace"]["cycles"]:
            prior = cycle["choose"]
            scores = [bounded_score_prompt(model, tok, score["prompt"], device,
                                           args.think_tokens)
                      for score in prior["scores"]]
            selected = 1 if scores[0]["score_yes_minus_no"] >= scores[1]["score_yes_minus_no"] else 2
            row = {"item_id": record["item_id"], "cycle": cycle["cycle"],
                   "expected": prior["expected"], "prior_selected": prior["selected"],
                   "selected": selected, "correct": selected == prior["expected"],
                   "scores": scores}
            reports.append(row)
            print(json.dumps({key: row[key] for key in
                              ("item_id", "cycle", "expected", "selected", "correct")}),
                  flush=True)
    result = {"schema_version": 1, "probe": "ledger-selector-bounded-think",
              "source": str(args.trace_dir.resolve()),
              "model": str(Path(args.base).resolve()), "device": device,
              "think_tokens": args.think_tokens, "total": len(reports),
              "correct": sum(r["correct"] for r in reports),
              "task2_expected": sum(r["expected"] == 2 for r in reports),
              "task2_selected": sum(r["selected"] == 2 for r in reports),
              "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("total", "correct", "task2_expected", "task2_selected")}, indent=2))


if __name__ == "__main__":
    main()
