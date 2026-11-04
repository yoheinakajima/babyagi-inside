"""Compare model-owned tokens in a host trace and a compiled live trace.

Forced controller and qwenFS tokens intentionally differ from the stock-model
surrogates used by :mod:`qfs.agent_native`.  This comparison removes those
forced intervals and asks the stronger semantic question: given the same
visible protocol, did pretrained Qwen choose the same tokens at every position
it owned?
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from qfs import circuit as F


class EquivalenceError(ValueError):
    """A trace cannot be aligned to its checkpoint program."""


def _compiled_stage_tokens(record, meta):
    tokens = list(record["tokens"])
    phase_tokens = set(meta["phase_tokens"])
    controls = [i for i, token in enumerate(tokens)
                if token in phase_tokens or token == meta["halt"]]
    if len(controls) != len(meta["stages"]):
        raise EquivalenceError("compiled trace does not have one control token per stage")
    starts = [0] + [position + 1 for position in controls[:-1]]
    owned = []
    for index, (stage, prefix, start, stop) in enumerate(
            zip(meta["stages"], meta["rom"], starts, controls)):
        body = tokens[start:stop]
        prefix = list(prefix)
        if body[:len(prefix)] != prefix:
            raise EquivalenceError(f"stage {index} ROM prefix mismatch")
        suffix = list(stage.get("rom_suffix", []))
        if stage.get("decision"):
            count = int(stage["decision"].get("token_count", 1))
            first, last = len(prefix), len(prefix) + count
            reasoning, answer = [], body[first:last]
            if body[last:last + len(suffix)] != suffix:
                raise EquivalenceError(f"stage {index} typed suffix mismatch")
        elif stage.get("free_span"):
            count = int(stage["free_span"])
            first, last = len(prefix), len(prefix) + count
            reasoning = body[first:last]
            if len(reasoning) != count or body[last:last + len(suffix)] != suffix:
                raise EquivalenceError(f"stage {index} bounded span mismatch")
            answer_start = last + len(suffix)
            try:
                close = body.index(F.TCEND, answer_start)
            except ValueError as exc:
                raise EquivalenceError(f"stage {index} missing tool-call close") from exc
            answer = body[answer_start:close + 1]
        elif stage["op"] == "write":
            try:
                close = body.index(F.TCEND, len(prefix))
            except ValueError as exc:
                raise EquivalenceError(f"stage {index} missing tool-call close") from exc
            reasoning, answer = [], body[len(prefix):close + 1]
        else:
            reasoning, answer = [], []
        owned.append({"reasoning": reasoning, "answer": answer,
                      "all": reasoning + answer})
    return owned


def compare(compiled_path, native_path, checkpoint=None):
    compiled_path, native_path = Path(compiled_path), Path(native_path)
    compiled = json.loads(compiled_path.read_text())
    native = json.loads(native_path.read_text())
    checkpoint = Path(checkpoint or compiled["checkpoint"])
    meta = json.loads((checkpoint / "agent.json").read_text())
    tok = AutoTokenizer.from_pretrained(checkpoint)
    compiled_owned = _compiled_stage_tokens(compiled, meta)
    native_phases = native.get("phases", [])
    if len(native_phases) != len(meta["stages"]):
        raise EquivalenceError("native stage count differs from checkpoint program")

    stages = []
    exact_tokens = total_tokens = 0
    exact_stages = 0
    answer_exact = reasoning_exact = 0
    answer_stages = reasoning_stages = 0
    for index, (stage, got_parts, phase) in enumerate(
            zip(meta["stages"], compiled_owned, native_phases)):
        expected = phase.get("model_token_ids")
        if expected is None:
            expected = tok.encode(phase.get("model_text", ""), add_special_tokens=False)
        count = int(stage.get("free_span", 0))
        expected_reasoning = expected[:count]
        expected_answer = expected[count:]
        got = got_parts["all"]
        limit = min(len(expected), len(got))
        first_mismatch = next((i for i in range(limit) if expected[i] != got[i]), None)
        if first_mismatch is None and len(expected) != len(got):
            first_mismatch = limit
        prefix_exact = first_mismatch if first_mismatch is not None else len(got)
        exact_tokens += prefix_exact
        total_tokens += max(len(expected), len(got))
        equal = expected == got
        exact_stages += int(equal)
        has_answer = bool(expected_answer or got_parts["answer"])
        has_reasoning = bool(count)
        this_answer_exact = expected_answer == got_parts["answer"]
        this_reasoning_exact = expected_reasoning == got_parts["reasoning"]
        answer_stages += int(has_answer)
        answer_exact += int(has_answer and this_answer_exact)
        reasoning_stages += int(has_reasoning)
        reasoning_exact += int(has_reasoning and this_reasoning_exact)
        stages.append({
            "stage": index, "cycle": stage["cycle"], "kind": stage["kind"],
            "op": stage["op"], "file": stage["file"], "exact": equal,
            "reasoning_exact": this_reasoning_exact if has_reasoning else None,
            "answer_exact": this_answer_exact if has_answer else None,
            "native_reasoning_tokens": len(expected_reasoning),
            "compiled_reasoning_tokens": len(got_parts["reasoning"]),
            "native_answer_tokens": len(expected_answer),
            "compiled_answer_tokens": len(got_parts["answer"]),
            "native_tokens": len(expected), "compiled_tokens": len(got),
            "exact_prefix_tokens": prefix_exact, "first_mismatch": first_mismatch,
            "native_at_mismatch": (expected[first_mismatch]
                                   if first_mismatch is not None
                                   and first_mismatch < len(expected) else None),
            "compiled_at_mismatch": (got[first_mismatch]
                                     if first_mismatch is not None
                                     and first_mismatch < len(got) else None),
        })
    result = {
        "schema_version": 1,
        "compiled_run": str(compiled_path), "native_run": str(native_path),
        "checkpoint": str(checkpoint), "protocol": meta.get("protocol"),
        "stages_exact": exact_stages, "stages_total": len(stages),
        "answer_stages_exact": answer_exact, "answer_stages_total": answer_stages,
        "reasoning_stages_exact": reasoning_exact,
        "reasoning_stages_total": reasoning_stages,
        "all_model_answers_exact": answer_exact == answer_stages,
        "all_bounded_reasoning_exact": reasoning_exact == reasoning_stages,
        "model_tokens_exact_prefix": exact_tokens,
        "model_tokens_total": total_tokens,
        "all_model_tokens_exact": all(stage["exact"] for stage in stages),
        "stages": stages,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiled")
    parser.add_argument("native")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(args.compiled, args.native, args.checkpoint)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
