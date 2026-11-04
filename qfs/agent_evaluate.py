"""Post-hoc trace inspection. Never imported by the decoder/model."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from qfs import circuit as F, disk
from qfs.agent import START, KINDS
from qfs.bridge import parse_call


def inspect(run_path, checkpoint_override=None):
    record = json.loads(Path(run_path).read_text())
    checkpoint = Path(checkpoint_override or record["checkpoint"])
    meta = json.loads((checkpoint / "agent.json").read_text())
    tok = AutoTokenizer.from_pretrained(checkpoint)
    tokens = record["tokens"]
    # START is supplied only in the one initial prefill, all other controls must
    # have been generated. Every static ROM token is checked byte-for-byte.
    controls = [(0, START)] + [(i + 1, t) for i, t in enumerate(tokens)
                              if t in meta["phase_tokens"]]
    stages = []
    for index, (start, control) in enumerate(controls):
        phase_index = control - START
        rom = meta["rom"][phase_index]
        end = controls[index + 1][0] - 1 if index + 1 < len(controls) else len(tokens)
        stage_meta = meta["stages"][phase_index]
        body_tokens = tokens[start:end]
        exact = body_tokens[:len(rom)] == rom
        decision_token = None
        if "decision" in stage_meta:
            decision_at = len(rom)
            decision_token = (body_tokens[decision_at]
                              if decision_at < len(body_tokens) else None)
            suffix = stage_meta.get("rom_suffix", [])
            suffix_at = decision_at + 1
            exact = (exact
                     and decision_token in stage_meta["decision"]["allowed_token_ids"]
                     and body_tokens[suffix_at:suffix_at + len(suffix)] == suffix)
        calls, pos = [], 0
        while F.TCSTART in body_tokens[pos:]:
            a = body_tokens.index(F.TCSTART, pos)
            try:
                b = body_tokens.index(F.TCEND, a)
            except ValueError:
                break
            fn, parameters = parse_call(tok.decode(body_tokens[a:b + 1]))
            calls.append({"function": fn, "parameters": parameters})
            pos = b + 1
        stages.append({
            "stage": phase_index, **stage_meta, "rom_exact": exact,
            "decision_token": decision_token,
            "response_completed": F.TRESPEND in body_tokens,
            "calls": calls,
            "body": tok.decode(body_tokens),
        })
    phases = []
    for cycle in range(1, meta["cycles"] + 1):
        for kind in KINDS:
            group = [s for s in stages if s["cycle"] == cycle and s["kind"] == kind]
            expected_files = {s["file"] for s in group if s["op"] == "write"}
            expected = (next(iter(expected_files)) if len(expected_files) == 1 else
                        sorted(expected_files) if expected_files else None)
            calls = [c for s in group for c in s["calls"]]
            writes = [c for c in calls if c["function"] == "write_file"
                      and c["parameters"].get("name") in expected_files
                      and c["parameters"].get("text", "").strip()]
            written_files = {call["parameters"].get("name") for call in writes}
            writes_complete = expected_files <= written_files
            ended = bool(group) and all(s["response_completed"] for s in group)
            phases.append({"cycle": cycle, "kind": kind,
                           "expected_file": expected,
                           "ended": ended,
                           "expected_file_written": bool(expected_files) and writes_complete,
                           "phase_complete": ended and writes_complete, "calls": calls,
                           "body": "".join(s["body"] for s in group)})
    all_tokens = record["prefix"] + tokens
    files = disk.compact(all_tokens)
    # Independently compare each generated read response with the most recent
    # prior write of that (file, version). These are token-exact checks, not a
    # language-model assessment of whether a tool "seemed" to work.
    writes = disk.writes(all_tokens)
    reads = []
    for pos, token in enumerate(all_tokens):
        if not F.ACKR0 <= token < F.NONE:
            continue
        value = token - F.ACKR0
        file_index, version = value >> F.VB, value & ((1 << F.VB) - 1)
        prior = [w for w in writes if w[0] == file_index and w[1] == version and w[4] < pos]
        try:
            stop = all_tokens.index(F.TRESPEND, pos)
        except ValueError:
            stop = len(all_tokens)
        expected = all_tokens[prior[-1][3]:prior[-1][4]] + [F.NL] if prior else None
        reads.append({"file_index": file_index, "version": version,
                      "exact": expected == all_tokens[pos + 1:stop]})
    result = {
        "objective": record["objective"], "seed": record["seed"],
        "protocol": meta["protocol"],
        "halted": bool(tokens and tokens[-1] == meta["halt"]), "token_count": len(tokens),
        "seconds": record["seconds"], "expected_phases": meta["cycles"] * 3,
        "expected_reads": sum(s["op"] == "read" for s in meta["stages"]),
        "write_count": len(writes),
        "expected_writes": sum(s["op"] == "write" for s in meta["stages"]),
        "phases": phases,
        "stages": stages,
        "reads": reads,
        "filesystem_pass": bool(reads) and all(r["exact"] for r in reads),
        "complete_cycles": sum(all(p["phase_complete"]
                                   for p in phases if p["cycle"] == c)
                               and sum(p["cycle"] == c for p in phases) == 3
                               for c in range(1, meta["cycles"] + 1)),
        "controller_pass": (record["halted"] and bool(tokens and tokens[-1] == meta["halt"])
                            and len(stages) == len(meta["rom"])
                            and [s["stage"] for s in stages] == list(range(len(meta["rom"])))
                            and all(s["rom_exact"] and s["response_completed"] for s in stages)),
        "latest_files": {tok.decode(list(name)): {"version": version, "text": tok.decode(body)}
                         for name, version, body in files.values()},
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint", help="Use a rebuilt checkpoint when moving an archived trace")
    args = parser.parse_args()
    result = inspect(args.run, args.checkpoint)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
