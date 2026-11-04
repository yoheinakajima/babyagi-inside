"""Isolate indexed task creation from selection, execution, and state errors."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs.benchmark import file_sha256, load_suite
from qfs.ceiling import generate_chat, parse_create
from qfs.protocol_v2 import indexed_create_prompt, indexed_kshot_create_prompt


PROMPTS = {"indexed": indexed_create_prompt,
           "indexed_kshot": indexed_kshot_create_prompt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--modes", nargs="+", choices=PROMPTS,
                        default=list(PROMPTS))
    parser.add_argument("--device")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to(device).eval()
    modes = {}
    for mode in args.modes:
        reports = []
        for item in suite["items"]:
            completed = []
            expected = item.get("rubric", {}).get("expected_results", [])
            for offset in range(item["cycles"]):
                cycle = offset + 1
                current = expected[offset - 1] if offset and expected else None
                prompt = PROMPTS[mode](item, cycle, current, completed)
                generation = generate_chat(model, tok, prompt, device, thinking=False,
                                           max_new_tokens=args.max_new_tokens)
                created = parse_create(generation["text"])
                reports.append({"item_id": item["id"], "cycle": cycle,
                                "current": current, "prompt": prompt,
                                "generation": generation, "parsed": created,
                                "parse_ok": created is not None})
                if created:
                    completed.append(created["next_task"])
                print(json.dumps({"mode": mode, "item": item["id"],
                                  "cycle": cycle, "parsed": created}), flush=True)
        modes[mode] = {"parse_pass": sum(row["parse_ok"] for row in reports),
                       "total": len(reports), "reports": reports}
    result = {"schema_version": 1,
              "benchmark": {"id": suite["benchmark_id"],
                            "sha256": file_sha256(args.suite)},
              "model": str(Path(args.base).resolve()), "device": device,
              "modes": modes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
