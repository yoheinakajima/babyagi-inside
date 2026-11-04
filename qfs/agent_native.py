"""Fast semantic harness for compiled-agent prompts on stock Qwen weights.

This is an experiment harness, not the compiled agent.  It feeds the same ROM
instructions and Qwen-visible filesystem responses to an unmodified base model,
so prompt candidates can be rejected before a costly merge and live decode.
The host advances phases and holds files here; only a compiled trace can prove
that the controller and qwenFS did those jobs inside the checkpoint.

Greedy decoding is the primary reproducible mode.  Sampled runs are useful as
robustness checks, but are not seed-identical to compiled decoding: the compiled
decoder calls multinomial even when a circuit-forced token has probability near
one, whereas this harness feeds forced tokens without consuming RNG draws.
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs import circuit as F
from qfs.agent import PROTOCOLS, program
from qfs.bridge import parse_call
from qfs.chat import TOOLS


TURN = [F.NL, F.IMSTART, F.ASSISTANT, F.NL, F.THINK, F.NL2, F.THINKEND, F.NL2]


def _ids(value):
    """Normalize token metadata without assuming the typed-stage schema."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return list(value)


@dataclass(frozen=True)
class StageView:
    prefix: list[int]
    suffix: list[int]
    allowed: list[int]
    allowed_text: list[str]
    decision_name: str | None
    decision_mode: str | None
    decision_owner: str | None
    free_span: int
    free_vocab_stop: int | None


def stage_view(stage):
    """Read current ROM stages and prospective prefix/decision/suffix stages.

    The compiler may expose a typed decision either as top-level fields or under
    ``decision``.  Keeping this adapter permissive lets the native harness remain
    useful while the in-weight selector metadata is finalized.
    """
    decision = stage.get("decision") or {}
    prefix = (stage.get("rom_prefix") if "rom_prefix" in stage else
              stage.get("prefix_rom") if "prefix_rom" in stage else stage.get("rom", []))
    suffix = (stage.get("rom_suffix") if "rom_suffix" in stage else
              stage.get("suffix_rom") if "suffix_rom" in stage else
              decision.get("suffix", []))
    allowed = (stage.get("allowed_token_ids") or stage.get("allowed_tokens") or
               stage.get("choice_tokens") or decision.get("allowed_token_ids") or
               decision.get("allowed_tokens") or decision.get("token_ids") or [])
    allowed_text = stage.get("allowed_text") or decision.get("allowed_text") or []
    name = stage.get("decision_name") or decision.get("name")
    return StageView(_ids(prefix), _ids(suffix), _ids(allowed), list(allowed_text), name,
                     decision.get("mode"), decision.get("owner"),
                     int(stage.get("free_span", 0)), stage.get("free_vocab_stop"))


def visible_response_tokens(tok, function, name, body=None):
    """Qwen-visible surrogate for qwenFS's response tokens.

    Acks are embedded as newlines and file markers as a dash in compiled qwenFS.
    Feeding those ordinary tokens to stock Qwen preserves the semantic context
    without requiring the extended filesystem vocabulary.
    """
    head = [F.NL, F.IMSTART, F.USER, F.NL, F.TRESP, F.NL]
    if function == "write_file":
        payload = [F.NL, F.SAVED, F.NL, F.DASH]
        payload += tok.encode(name, add_special_tokens=False)
    elif body is not None:
        payload = [F.NL] + list(body)
    else:
        payload = [F.NL, F.NOT, F.FOUND_]
    return head + payload + [F.NL, F.TRESPEND] + TURN


def choose_token(logits, temperature, top_p, generator, allowed=None, vocab_stop=None):
    """Match the compiled decoder's greedy/nucleus policy, optionally typed."""
    allowed = list(allowed or [])
    if allowed:
        ids = torch.tensor(allowed, device=logits.device)
        choice_logits = logits.index_select(0, ids)
        if temperature <= 0:
            return int(ids[int(choice_logits.argmax())])
        probs = (choice_logits / temperature).softmax(-1).cpu()
        p, order = probs.sort(descending=True)
        p *= p.cumsum(0) - p < top_p
        picked = int(order[torch.multinomial(p / p.sum(), 1, generator=generator)])
        return allowed[picked]
    if vocab_stop is not None:
        if not 0 < vocab_stop <= logits.shape[0]:
            raise ValueError(f"invalid vocab_stop {vocab_stop}")
        logits = logits[:vocab_stop]
    if temperature <= 0:
        return int(logits.argmax())
    probs = (logits / temperature).softmax(-1).cpu()
    p, indices = probs.sort(descending=True)
    p *= p.cumsum(0) - p < top_p
    return int(indices[torch.multinomial(p / p.sum(), 1, generator=generator)])


class NativeRunner:
    def __init__(self, model, tok, device, temperature=0.0, top_p=0.9, seed=0):
        self.model, self.tok, self.device = model, tok, device
        self.temperature, self.top_p = temperature, top_p
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.cache = None
        self.pending = []
        self.context = []
        self.forward_seconds = 0.0

    @torch.inference_mode()
    def feed(self, tokens):
        tokens = list(tokens)
        if not tokens:
            raise ValueError("cannot feed an empty token sequence")
        begin = time.monotonic()
        out = self.model(input_ids=torch.tensor([tokens], device=self.device),
                         past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.forward_seconds += time.monotonic() - begin
        self.cache = out.past_key_values
        self.context.extend(tokens)
        return out.logits[0, -1].float()

    def queue(self, tokens):
        self.pending.extend(tokens)

    def start(self, forced_prefix):
        tokens = self.pending + list(forced_prefix)
        self.pending = []
        return self.feed(tokens)

    def pick(self, logits, allowed=None, vocab_stop=None):
        return choose_token(logits, self.temperature, self.top_p,
                            self.generator, allowed=allowed, vocab_stop=vocab_stop)


def _call(stage, tok, call_tokens):
    function, parameters = parse_call(tok.decode(call_tokens))
    expected = stage.get("file")
    if function == "?":
        function = "read_file" if stage.get("op") == "read" else "write_file"
    parameters.setdefault("name", expected)
    return function, parameters


def run_native(model, tok, stages, objective, device="mps", temperature=0.0,
               top_p=0.9, seed=0, limit_per_write=240):
    """Run an objective through static stages using the stock semantic model."""
    wall_begin = time.monotonic()
    runner = NativeRunner(model, tok, device, temperature, top_p, seed)
    initial = tok.apply_chat_template(
        [{"role": "user", "content": objective}], tools=TOOLS, tokenize=True,
        add_generation_prompt=False, enable_thinking=False, return_dict=False)
    runner.queue(list(initial))
    files, reports = {}, []
    forced_rom = forced_response = model_free = model_choices = 0
    stopped = None

    for number, stage in enumerate(stages):
        view = stage_view(stage)
        phase_begin = time.monotonic()
        report = {"stage": number, "cycle": stage.get("cycle"),
                  "kind": stage.get("kind"), "op": stage.get("op"),
                  "file": stage.get("file"), "forced_prefix_tokens": len(view.prefix),
                  "forced_suffix_tokens": len(view.suffix), "model_tokens": 0,
                  "bounded_reasoning_tokens": view.free_span,
                  "decision": None, "completed": False}
        forced_rom += len(view.prefix) + len(view.suffix)

        generated = []
        if view.allowed:
            logits = runner.start(view.prefix)
            native_allowed = {str(token): float(logits[token]) for token in view.allowed}
            unrestricted = int(logits.argmax())
            token = runner.pick(logits, view.allowed)
            generated.append(token)
            model_choices += 1
            logits = runner.feed([token])
            report["decision"] = {"name": view.decision_name or "typed_choice",
                                  "mode": view.decision_mode or "typed_choice",
                                  "owner": view.decision_owner or "model",
                                  "allowed_tokens": view.allowed, "selected_token": token,
                                  "allowed_text": view.allowed_text,
                                  "selected_text": tok.decode([token]),
                                  "unrestricted_top_token": unrestricted,
                                  "native_allowed_logits": native_allowed}
            if view.suffix:
                logits = runner.feed(view.suffix)
            call_tokens = view.prefix + generated + view.suffix
        elif view.free_span:
            if not view.suffix:
                raise ValueError("a bounded free span requires a forced suffix")
            logits = runner.start(view.prefix)
            for _ in range(view.free_span):
                token = runner.pick(logits, vocab_stop=view.free_vocab_stop)
                generated.append(token)
                logits = runner.feed([token])
            logits = runner.feed(view.suffix)
            call_tokens = view.prefix + generated + view.suffix
        else:
            logits = None
            call_tokens = view.prefix

        if stage.get("op") == "read":
            function, parameters = _call(stage, tok, call_tokens)
            name = parameters.get("name")
            body = files.get(name)
            response = visible_response_tokens(tok, function, name, body)
            # Static reads have not been fed yet. A typed read was fed through its
            # decision and forced suffix so the model itself supplied the filename.
            runner.queue((view.prefix if not (view.allowed or view.free_span) else []) + response)
            forced_response += len(response)
            report.update({"function": function, "parameters": parameters,
                           "read_found": body is not None, "read_text":
                           tok.decode(body) if body is not None else None,
                           "model_tokens": len(generated),
                           "model_token_ids": generated,
                           "model_text": tok.decode(generated),
                           "completed": bool(call_tokens and F.TCEND in call_tokens),
                           "seconds": time.monotonic() - phase_begin})
            reports.append(report)
            continue

        if view.allowed:
            if not view.suffix or view.suffix[-1] != F.TCEND:
                while len(generated) < limit_per_write:
                    token = runner.pick(logits)
                    generated.append(token)
                    logits = runner.feed([token])
                    if token == F.TCEND:
                        break
        elif view.free_span:
            answer_tokens = 0
            while answer_tokens < limit_per_write:
                token = runner.pick(logits)
                generated.append(token)
                answer_tokens += 1
                logits = runner.feed([token])
                if token == F.TCEND:
                    break
        else:
            logits = runner.start(view.prefix)
            while len(generated) < limit_per_write:
                token = runner.pick(logits)
                generated.append(token)
                logits = runner.feed([token])
                if token == F.TCEND:
                    break
        model_free += len(generated) - (1 if view.allowed else 0)
        if view.allowed:
            decision_count = int((stage.get("decision") or {}).get("token_count", 1))
            call_tokens = (view.prefix + generated[:decision_count] + view.suffix
                           + generated[decision_count:])
        elif view.free_span:
            call_tokens = (view.prefix + generated[:view.free_span] + view.suffix
                           + generated[view.free_span:])
        else:
            call_tokens = view.prefix + generated + view.suffix
        function, parameters = _call(stage, tok, call_tokens)
        completed = bool(call_tokens and F.TCEND in call_tokens)
        name, text = parameters.get("name"), parameters.get("text")
        if completed and function == "write_file" and name and text is not None:
            files[name] = tok.encode(text, add_special_tokens=False)
            response = visible_response_tokens(tok, function, name)
            runner.queue(response)
            forced_response += len(response)
        else:
            stopped = {"stage": number, "reason": "write_call_incomplete_or_invalid"}
        report.update({"function": function, "parameters": parameters,
                       "model_tokens": len(generated), "model_token_ids": generated,
                       "model_text": tok.decode(generated),
                       "completed": completed and stopped is None,
                       "seconds": time.monotonic() - phase_begin})
        reports.append(report)
        if stopped:
            break

    latest = {name: tok.decode(body) for name, body in files.items()}
    return {
        "schema_version": 1, "harness": "native-semantic-simulator",
        "compiled_controller_executed": False, "objective": objective,
        "decoding": {"temperature": temperature, "top_p": top_p, "seed": seed,
                     "mode": "greedy" if temperature <= 0 else "sampled",
                     "sampled_rng_equivalence": ("exact" if temperature <= 0 else
                         "not seed-identical: forced-token multinomial draws are not consumed")},
        "stages_expected": len(stages), "stages_attempted": len(reports),
        "stages_completed": sum(bool(phase["completed"]) for phase in reports),
        "stopped": stopped, "phases": reports, "latest_files": latest,
        "token_provenance": {"objective_prefill": len(initial), "forced_rom": forced_rom,
                             "forced_visible_responses": forced_response,
                             "model_free": model_free, "model_typed_choices": model_choices,
                             "model_bounded_reasoning": sum(
                                 phase.get("bounded_reasoning_tokens", 0) for phase in reports)},
        "forward_seconds": runner.forward_seconds,
        "wall_seconds": time.monotonic() - wall_begin,
        "limitations": ["host advances phases and simulates visible filesystem responses",
                        "only compiled traces validate controller, qwenFS, and HALT"],
    }


def load_and_run(base, objective, cycles=3, protocol="babyagi", device=None,
                 temperature=0.0, top_p=0.9, seed=0, limit_per_write=240,
                 think_tokens=None):
    device = device or ("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    stages = program(tok, cycles, protocol, think_tokens)
    model = Qwen3_5ForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(device).eval()
    result = run_native(model, tok, stages, objective, device, temperature,
                        top_p, seed, limit_per_write)
    result.update({"base": str(Path(base).resolve()), "cycles": cycles,
                   "protocol": protocol, "think_tokens": think_tokens,
                   "device": device})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("objective")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="babyagi")
    parser.add_argument("--output", default="results/native-agent.json")
    parser.add_argument("--device")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-per-write", type=int, default=240)
    parser.add_argument("--think-tokens", type=int,
                        help="bounded reasoning tokens for pointer protocols")
    args = parser.parse_args()
    result = load_and_run(args.base, args.objective, args.cycles, args.protocol,
                          args.device, args.temperature, args.top_p, args.seed,
                          args.limit_per_write, args.think_tokens)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "stages": result["stages_completed"],
                      "stopped": result["stopped"], "seconds": result["forward_seconds"]}))


if __name__ == "__main__":
    main()
