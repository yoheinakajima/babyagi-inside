"""Check actual checkpoint logits against an independent scripted protocol.

This checks ROM tokens, file responses, microstate transitions, and HALT.
Semantic phase bodies are scripted ONLY in this circuit test. End-to-end agent
tests separately use unmodified sampling and actual Qwen-produced phase bodies.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs import disk
from qfs.agent import START
from qfs.bridge import parse_call
from qfs.chat import TOOLS
from qfs import circuit as F
from tests import test_circuit as reference


@torch.inference_mode()
def check(checkpoint, device="mps", chunk=64):
    meta = json.loads((Path(checkpoint) / "agent.json").read_text())
    tok = AutoTokenizer.from_pretrained(checkpoint)
    # Include the actual chat/tool-schema prefill. Its literal example calls
    # exercise the parser-state reset at START, which an empty-prefix test misses.
    objective = "Perform a sequence of small text tasks."
    tokens = list(tok.apply_chat_template(
        [{"role": "user", "content": objective}],
        tools=TOOLS, tokenize=True,
        add_generation_prompt=meta["stages"][0]["kind"] == "bootstrap",
        enable_thinking=False, return_dict=False)) + [START]
    if meta["stages"][0]["kind"] == "bootstrap":
        tokens += tok.encode("<tool_call>\n<function=write_file>\n<parameter=name>\ngoal.txt"
                             "\n</parameter>\n<parameter=text>\n" + objective +
                             "\n</parameter>\n</function>\n</tool_call>", add_special_tokens=False)
    expected = {}
    choice_positions = {}
    free_positions = {}
    files = {}
    for i, (rom, stage) in enumerate(zip(meta["rom"], meta["stages"])):
        for token in rom:
            expected[len(tokens) - 1] = token
            tokens.append(token)
        decision_body = []
        if "decision" in stage:
            allowed = stage["decision"]["allowed_token_ids"]
            decision_body = [allowed[(stage["cycle"] - 1) % len(allowed)]]
            choice_positions[len(tokens) - 1] = allowed
            tokens += decision_body
            for token in stage.get("rom_suffix", []):
                expected[len(tokens) - 1] = token
                tokens.append(token)
        elif stage.get("free_span"):
            free_token_ids = tok.encode(" test", add_special_tokens=False)
            assert free_token_ids and free_token_ids[0] < stage["free_vocab_stop"]
            for _ in range(stage["free_span"]):
                free_positions[len(tokens) - 1] = stage["free_vocab_stop"]
                tokens.append(free_token_ids[0])
            for token in stage.get("rom_suffix", []):
                expected[len(tokens) - 1] = token
                tokens.append(token)
        filename = stage["file"]
        if stage["op"] == "read" and decision_body:
            _, parameters = parse_call(tok.decode(rom + decision_body + stage["rom_suffix"]))
            filename = parameters["name"]
        name = tok.encode(filename, add_special_tokens=False)
        if stage["op"] == "write":
            file_index, version, _ = files.get(filename, (len(files), 0, None))
            version += 1
            semantic_body = decision_body or tok.encode(
                ('"sample"}' if stage.get("free_span") else
                 objective if stage["kind"] == "bootstrap" else
                 f"Sample content for cycle {stage['cycle']} {stage['kind']}."),
                add_special_tokens=False)
            # A protocol may compile a static field prefix into the open write call.
            # It is already present in the ROM token stream, but it is also part of
            # the stored file body used by later reference reads.
            body = tok.encode(stage.get("text_prefix", ""), add_special_tokens=False) + semantic_body
            files[filename] = file_index, version, body
            # Qwen supplies only this semantic value and its normal XML close.
            if stage["kind"] != "bootstrap" and not decision_body:
                tokens += semantic_body + [F.NL] + tok.encode(
                    "</parameter>\n</function>\n</tool_call>", add_special_tokens=False)
            response = (reference.HEAD + [F.ackw(file_index, version), F.SAVED, F.NL,
                         F.MARK0 + file_index, *name, F.NL, F.TRESPEND] + reference.TURN)
        else:
            file_index, version, body = files[filename]
            response = (reference.HEAD + [F.ackr(file_index, version), *body, F.NL,
                                          F.TRESPEND] + reference.TURN)
        for token in response:
            expected[len(tokens) - 1] = token
            tokens.append(token)
        control = START + i + 1 if i + 1 < len(meta["rom"]) else meta["halt"]
        expected[len(tokens) - 1] = control
        tokens.append(control)
    # The terminal HALT does not need a forward pass.
    tokens = tokens[:-1]
    model = Qwen3_5ForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).to(device).eval()
    disk.attach(model)
    cache = disk.FSCache(model.config)
    good, bad, choices, free_checks = 0, [], [], []
    for start in range(0, len(tokens), chunk):
        out = model(input_ids=torch.tensor([tokens[start:start + chunk]], device=device),
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        predictions = out.logits[0].argmax(-1).tolist()
        for offset, pred in enumerate(predictions):
            pos = start + offset
            if pos in expected:
                if pred == expected[pos]:
                    good += 1
                else:
                    bad.append({"position": pos, "wanted": expected[pos], "got": pred})
            if pos in choice_positions:
                allowed = choice_positions[pos]
                logits = out.logits[0, offset].float()
                allowed_values = logits[allowed]
                masked = logits.clone()
                masked[allowed] = -torch.inf
                margin = float(allowed_values.min() - masked.max())
                top = logits.topk(len(allowed)).indices.tolist()
                choices.append({"position": pos, "allowed": allowed, "top": top,
                                "margin": margin, "pass": set(top) == set(allowed) and margin > 0})
            if pos in free_positions:
                stop = free_positions[pos]
                logits = out.logits[0, offset].float()
                allowed_top = float(logits[:stop].max())
                illegal_top = float(logits[stop:].max())
                margin = allowed_top - illegal_top
                free_checks.append({"position": pos, "vocab_stop": stop,
                                    "margin": margin, "pass": margin > 0})
        print(f"checked {min(start + chunk, len(tokens))}/{len(tokens)} tokens", flush=True)
    result = {"forced_tokens_correct": good, "forced_tokens_total": len(expected),
              "failures": bad[:20], "typed_choices": choices,
              "free_span": {"positions": len(free_checks),
                            "minimum_margin": min((check["margin"] for check in free_checks),
                                                  default=None),
                            "failures": [check for check in free_checks
                                         if not check["pass"]][:20]},
              "pass": (not bad and all(choice["pass"] for choice in choices)
                       and all(check["pass"] for check in free_checks))}
    print(json.dumps(result), flush=True)
    assert result["pass"], result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    check(args.checkpoint, args.device)


if __name__ == "__main__":
    main()
