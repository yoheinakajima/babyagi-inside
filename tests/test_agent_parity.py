"""Sequential, memory-bounded smoke check of the 0.8B widening.

Agent-inactive next-token distributions are compared with the native source
model. Models are loaded one at a time on a 16 GB machine. This is a small
numerical check, not a claim of full behavioral equivalence.
"""
import argparse
import gc
import json

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs import disk, merge as M


QUERIES = [
    "What is the capital of France?",
    "Start with oak and append -red. Show only the result.",
    "Give two tasks to write a short guide for a note-taking app.",
    "Compute 3 + 2. Show only the answer.",
]


@torch.inference_mode()
def capture(model, inputs, device):
    outputs = []
    for ids in inputs:
        out = model(input_ids=torch.tensor([ids], device=device), use_cache=False,
                    logits_to_keep=1)
        outputs.append(out.logits[0, -1, :248320].float().cpu())
    return torch.stack(outputs)


def check(checkpoint, source, device="mps"):
    tok = AutoTokenizer.from_pretrained(source)
    inputs = [tok.apply_chat_template([{"role": "user", "content": q}],
              tokenize=True, add_generation_prompt=True, enable_thinking=False,
              return_dict=False) for q in QUERIES]
    model = Qwen3_5ForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).to(device).eval()
    disk.attach(model)
    compiled = capture(model, inputs, device)
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    print("Compiled logits captured; loading native source", flush=True)
    model = M.base_model(source).to(device).eval()
    native = capture(model, inputs, device)
    p, q = native.log_softmax(-1), compiled.log_softmax(-1)
    kl = (p.exp() * (p - q)).sum(-1)
    result = {"queries": QUERIES, "kl_native_to_compiled": kl.tolist(),
              "top1_native": native.argmax(-1).tolist(),
              "top1_compiled": compiled.argmax(-1).tolist(),
              "top1_agreement": int((native.argmax(-1) == compiled.argmax(-1)).sum()),
              "positions": len(QUERIES), "pass": float(kl.max()) < 0.03}
    print(json.dumps(result, indent=2), flush=True)
    assert result["pass"], "The widening changed native predictions materially"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("source")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    check(args.checkpoint, args.source, args.device)


if __name__ == "__main__":
    main()
