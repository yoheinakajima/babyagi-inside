"""Exact provenance accounting for a compiled-agent trace.

The decoder records one flat sequence after the initial prefill.  This module
partitions every emitted token into four disjoint classes:

* compiled ROM (including a typed stage's forced suffix),
* qwenFS response tokens,
* controller transition/HALT tokens, and
* model-generated tool-call content.

The last class is further split into semantic payload and ordinary tool-call
closing syntax.  Token provenance and semantic ownership are deliberately
reported separately: a long model-written JSON object is not an additional
agent decision, while a one-token typed choice is.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from qfs import circuit as F


SCHEMA_VERSION = 1
OWNERS = {
    "model_free_text",
    "typed_model_choice",
    "deterministic_circuit_choice",
    "deterministic_circuit_content",
}


class TraceAccountingError(ValueError):
    """The trace cannot be partitioned exactly under its checkpoint metadata."""


def _endswith(values, suffix):
    return bool(suffix) and len(values) >= len(suffix) and values[-len(suffix):] == suffix


def _closure_length(tokens, tok):
    """Return the canonical model-written closing suffix length, or zero."""
    forms = (
        "\n</parameter>\n</function>\n</tool_call>",
        "</parameter>\n</function>\n</tool_call>",
    )
    matches = [tok.encode(text, add_special_tokens=False) for text in forms]
    lengths = [len(ids) for ids in matches if _endswith(tokens, ids)]
    return max(lengths, default=0)


def _suffix_for(meta, stage, index):
    """Accept the typed-stage metadata convention and harmless aliases."""
    if "rom_suffix" in stage:
        return list(stage["rom_suffix"])
    suffixes = meta.get("rom_suffixes")
    if suffixes is not None:
        return list(suffixes[index])
    return []


def _decision_for(stage):
    decision = stage.get("decision")
    if isinstance(decision, dict):
        return decision
    # Forward-compatible alias for early experimental metadata.
    decision = stage.get("typed_decision")
    return decision if isinstance(decision, dict) else None


def _ownership(stage, decision, payload_count):
    explicit = stage.get("semantic_owner")
    if explicit:
        if explicit not in OWNERS:
            raise TraceAccountingError(f"unknown semantic_owner {explicit!r}")
        return explicit
    if decision:
        return "typed_model_choice"
    if stage.get("op") == "write" and stage.get("kind") == "prioritize" \
            and payload_count == 0 and stage.get("text_prefix", "").strip():
        return "deterministic_circuit_choice"
    if stage.get("op") == "write" and payload_count == 0:
        return "deterministic_circuit_content"
    return "model_free_text"


def account_data(record, meta, tok, run_label=None, checkpoint_label=None):
    """Account for an already loaded record/agent metadata pair.

    A future typed stage is represented by its usual forced prefix in
    ``meta["rom"][i]``, a stage ``decision`` object with ``token_count`` and
    ``allowed_tokens``, and a forced ``rom_suffix`` in the stage (or the
    parallel top-level ``rom_suffixes`` list).  The decision token remains
    model-owned; prefix and suffix remain ROM-owned.
    """
    tokens = list(record["tokens"])
    stages = meta["stages"]
    roms = meta["rom"]
    if len(stages) != len(roms):
        raise TraceAccountingError("stage/ROM length mismatch")

    phase_tokens = set(meta["phase_tokens"])
    control_positions = [
        i for i, token in enumerate(tokens)
        if token in phase_tokens or token == meta["halt"]
    ]
    expected_controls = list(meta["phase_tokens"][1:]) + [meta["halt"]]
    actual_controls = [tokens[i] for i in control_positions]
    if actual_controls != expected_controls:
        raise TraceAccountingError(
            f"control sequence mismatch: expected {expected_controls}, got {actual_controls}")
    if len(control_positions) != len(stages):
        raise TraceAccountingError("one transition/HALT token is required per stage")

    starts = [0] + [position + 1 for position in control_positions[:-1]]
    stage_reports = []
    totals = Counter()
    by_kind = defaultdict(Counter)
    decisions = []

    for index, (stage, prefix, start, stop) in enumerate(
            zip(stages, roms, starts, control_positions)):
        body = tokens[start:stop]
        prefix = list(prefix)
        if body[:len(prefix)] != prefix:
            raise TraceAccountingError(f"stage {index} ROM prefix differs from checkpoint")

        suffix = _suffix_for(meta, stage, index)
        decision = _decision_for(stage)
        rom_count = len(prefix)
        model_tokens = []
        payload_count = 0
        closure_count = 0

        if decision:
            decision_count = int(decision.get("token_count", 1))
            if decision_count < 1:
                raise TraceAccountingError(f"stage {index} has invalid decision token_count")
            decision_start = len(prefix)
            decision_end = decision_start + decision_count
            model_tokens = body[decision_start:decision_end]
            if len(model_tokens) != decision_count:
                raise TraceAccountingError(f"stage {index} is missing its typed decision")
            allowed = decision.get("allowed_token_ids", decision.get("allowed_tokens"))
            if allowed is not None and any(token not in allowed for token in model_tokens):
                raise TraceAccountingError(f"stage {index} emitted an illegal typed decision")
            if body[decision_end:decision_end + len(suffix)] != suffix:
                raise TraceAccountingError(f"stage {index} forced suffix differs from checkpoint")
            rom_count += len(suffix)
            response_start = decision_end + len(suffix)
            if not suffix or suffix[-1] != F.TCEND:
                raise TraceAccountingError(f"stage {index} typed suffix must close the tool call")
            payload_count = decision_count
            chosen = tok.decode(model_tokens)
        elif stage["op"] == "write":
            try:
                close = body.index(F.TCEND, len(prefix))
            except ValueError as exc:
                raise TraceAccountingError(f"stage {index} has no completed model tool call") from exc
            model_tokens = body[len(prefix):close + 1]
            closure_count = _closure_length(model_tokens, tok)
            payload_count = len(model_tokens) - closure_count
            response_start = close + 1
            chosen = None
        else:
            if not prefix or prefix[-1] != F.TCEND:
                raise TraceAccountingError(f"stage {index} read ROM must close the tool call")
            response_start = len(prefix)
            chosen = None

        fs_count = len(body) - response_start
        if fs_count < 0:
            raise TraceAccountingError(f"stage {index} has a negative response length")
        owner = None
        if stage["op"] == "write" or decision:
            owner = _ownership(stage, decision, payload_count)
            decision_report = {
                "stage": index,
                "cycle": stage["cycle"],
                "kind": stage["kind"],
                "owner": owner,
                "payload_tokens": payload_count,
            }
            if decision:
                decision_report.update({
                    "chosen": chosen,
                    "chosen_token_ids": model_tokens,
                    "allowed_token_ids": allowed,
                    "allowed_text": decision.get("allowed_text"),
                })
            decisions.append(decision_report)

        counts = {
            "rom_forced": rom_count,
            "filesystem_forced": fs_count,
            "controller_control": 1,
            "model_generated": len(model_tokens),
            "model_payload": payload_count,
            "model_tool_closure": closure_count,
        }
        for key, value in counts.items():
            totals[key] += value
            by_kind[stage["kind"]][key] += value
        stage_reports.append({
            "stage": index,
            "cycle": stage["cycle"],
            "kind": stage["kind"],
            "op": stage["op"],
            "file": stage["file"],
            "semantic_owner": owner,
            **counts,
        })

    totals["forced_total"] = (
        totals["rom_forced"] + totals["filesystem_forced"]
        + totals["controller_control"])
    totals["accounted_tokens"] = totals["forced_total"] + totals["model_generated"]
    emitted = len(tokens)
    exact_partition = totals["accounted_tokens"] == emitted
    model_split_exact = (
        totals["model_payload"] + totals["model_tool_closure"]
        == totals["model_generated"])
    if not exact_partition or not model_split_exact:
        raise TraceAccountingError("token partition invariant failed")

    summary = Counter(decision["owner"] for decision in decisions)
    result = {
        "schema_version": SCHEMA_VERSION,
        "run": run_label,
        "checkpoint": checkpoint_label,
        "protocol": meta.get("protocol"),
        "cycles": meta.get("cycles"),
        "totals": {
            "emitted_tokens": emitted,
            **dict(totals),
            "forced_percent": round(100 * totals["forced_total"] / emitted, 3) if emitted else 0.0,
            "model_generated_percent": (round(
                100 * totals["model_generated"] / emitted, 3) if emitted else 0.0),
        },
        "by_kind": {kind: dict(counts) for kind, counts in sorted(by_kind.items())},
        "decision_summary": dict(summary),
        "decisions": decisions,
        "stages": stage_reports,
        "invariants": {
            "control_sequence_exact": True,
            "stage_rom_exact": True,
            "token_partition_exact": exact_partition,
            "model_payload_plus_closure_exact": model_split_exact,
        },
    }
    return result


def account(run_path, checkpoint_override=None, tokenizer=None):
    run_path = Path(run_path)
    record = json.loads(run_path.read_text())
    checkpoint = Path(checkpoint_override or record["checkpoint"])
    meta = json.loads((checkpoint / "agent.json").read_text())
    tok = tokenizer or AutoTokenizer.from_pretrained(checkpoint)
    return account_data(record, meta, tok, str(run_path), str(checkpoint))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint",
                        help="Override the checkpoint path (only valid with one run)")
    args = parser.parse_args()
    if args.checkpoint and len(args.runs) != 1:
        parser.error("--checkpoint requires exactly one run")
    reports = [account(path, args.checkpoint) for path in args.runs]
    value = reports[0] if len(reports) == 1 else reports
    text = json.dumps(value, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
