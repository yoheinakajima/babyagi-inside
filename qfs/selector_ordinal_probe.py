"""Replay ledger choices as independent ordinal readiness judgments."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


@torch.inference_mode()
def ordinal_score(model, tok, prompt, device):
    context = prompt.rsplit("\n\n", 1)[0]
    ordinal_prompt = context + (
        "\n\nRate whether this candidate is the single next undone task: "
        "0 = completed or out of order; 1 = probably not next; 2 = probably next; "
        "3 = definitely the next task. Treat initial-state text as input, not work. "
        "Answer exactly one digit: 0, 1, 2, or 3."
    )
    ids = tok.apply_chat_template(
        [{"role": "user", "content": ordinal_prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    output = model(input_ids=torch.tensor([ids], device=device), use_cache=False,
                   logits_to_keep=1)
    token_ids = [tok.encode(str(value), add_special_tokens=False)[0] for value in range(4)]
    logits = output.logits[0, -1].float()[token_ids]
    probabilities = logits.softmax(-1)
    predicted = int(logits.argmax())
    expectation = float((probabilities * torch.arange(4, device=device)).sum())
    return {"prompt": ordinal_prompt, "token_ids": token_ids,
            "logits": [float(value) for value in logits],
            "probabilities": [float(value) for value in probabilities],
            "predicted": predicted, "expectation": expectation}


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
        for cycle in record["trace"]["cycles"]:
            prior = cycle["choose"]
            scores = [ordinal_score(model, tok, score["prompt"], device)
                      for score in prior["scores"]]
            by_digit = (1 if scores[0]["predicted"] >= scores[1]["predicted"] else 2)
            by_expectation = (1 if scores[0]["expectation"] >= scores[1]["expectation"] else 2)
            reports.append({"item_id": record["item_id"], "cycle": cycle["cycle"],
                            "expected": prior["expected"], "scores": scores,
                            "selected_digit": by_digit,
                            "digit_correct": by_digit == prior["expected"],
                            "selected_expectation": by_expectation,
                            "expectation_correct": by_expectation == prior["expected"]})
    result = {"schema_version": 1, "probe": "ledger-selector-ordinal",
              "source": str(args.trace_dir.resolve()),
              "model": str(Path(args.base).resolve()), "device": device,
              "total": len(reports),
              "digit_correct": sum(r["digit_correct"] for r in reports),
              "expectation_correct": sum(r["expectation_correct"] for r in reports),
              "digit_task2_selected": sum(r["selected_digit"] == 2 for r in reports),
              "expectation_task2_selected": sum(r["selected_expectation"] == 2
                                                for r in reports),
              "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("reports", "model", "source")}, indent=2))


if __name__ == "__main__":
    main()
