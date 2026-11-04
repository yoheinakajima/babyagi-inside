"""Inference-only host: one objective in, sampled tokens out.

There is no phase parser, task scheduler, prompt injection after prefill, tool
executor, logits processor, or recovery callback here. qwenFS computes tool
responses; the compiled agent computes its transitions. START/HALT are analogous
to BOS/EOS. A maximum token budget is an external experimental bound.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs import disk
from qfs.chat import TOOLS


class CompiledDecoder:
    """Load one compiled checkpoint and decode multiple independent objectives."""

    def __init__(self, checkpoint, device="mps"):
        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(self.checkpoint)
        print(f"Loading compiled model {self.checkpoint}", flush=True)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            self.checkpoint, dtype=torch.bfloat16).to(device).eval()
        disk.attach(self.model)  # qwenFS attention-cache masks; no host-side filesystem.

    @torch.inference_mode()
    def decode(self, objective, output, seed=0, limit=6000, temperature=0.7,
               stream=True):
        """Decode one clean-cache run without reloading checkpoint weights."""
        cache = disk.FSCache(self.model.config)
        prefix = self.tok.apply_chat_template(
            [{"role": "user", "content": objective}], tools=TOOLS,
            tokenize=True, add_generation_prompt=False, enable_thinking=False,
            return_dict=False)
        prefix = list(prefix) + [self.model.config.bos_token_id]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        emitted, current = [], prefix
        begin = time.monotonic()
        if stream:
            print("Decoding (all subsequent inputs are sampled model tokens)", flush=True)
        for _ in range(limit):
            out = self.model(input_ids=torch.tensor([current], device=self.device),
                             past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            logits = out.logits[0, -1].float()
            if temperature <= 0:
                token = int(logits.argmax())
            else:
                probs = (logits / temperature).softmax(-1).cpu()
                p, indices = probs.sort(descending=True)
                p *= (p.cumsum(0) - p < 0.9)
                token = int(indices[torch.multinomial(
                    p / p.sum(), 1, generator=generator)])
            emitted.append(token)
            if stream:
                print(self.tok.decode([token]), end="", flush=True)
            if token == self.model.config.eos_token_id:
                break
            current = [token]
        elapsed = time.monotonic() - begin
        record = {
            "checkpoint": str(self.checkpoint), "objective": objective,
            "seed": seed, "temperature": temperature, "limit": limit,
            "input_codec": "one objective chat prefill plus BOS; all later inputs sampled tokens",
            "halted": bool(emitted and emitted[-1] == self.model.config.eos_token_id),
            "tokens": emitted, "prefix": prefix, "seconds": elapsed,
            "text": self.tok.decode(emitted),
        }
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n")
        if stream:
            print(flush=True)
        print(f"Saved {len(emitted)} tokens in {elapsed:.1f}s to {output}", flush=True)
        return record


def decode(checkpoint, objective, output, device="mps", seed=0, limit=6000,
           temperature=0.7):
    decoder = CompiledDecoder(checkpoint, device)
    return decoder.decode(objective, output, seed, limit, temperature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("objective")
    parser.add_argument("--output", default="results/agent-run.json")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    decode(args.checkpoint, args.objective, args.output, args.device,
           args.seed, args.limit, args.temperature)


if __name__ == "__main__":
    main()
