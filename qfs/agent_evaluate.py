"""Post-hoc trace inspection. Never imported by the decoder/model."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from qfs import circuit as F, disk
from qfs.agent import START, KINDS
from qfs.bridge import parse_call


def inspect(run_path):
    record = json.loads(Path(run_path).read_text())
    checkpoint = Path(record["checkpoint"])
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
        exact = tokens[start:start + len(rom)] == rom
        body_tokens = tokens[start:end]
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
            "stage": phase_index, **meta["stages"][phase_index], "rom_exact": exact,
            "response_completed": F.TRESPEND in body_tokens,
            "calls": calls,
            "body": tok.decode(body_tokens),
        })
    phases = []
    for cycle in range(1, meta["cycles"] + 1):
        for kind, expected in zip(KINDS, ("tasks.txt", "current.txt", "result.txt")):
            group = [s for s in stages if s["cycle"] == cycle and s["kind"] == kind]
            calls = [c for s in group for c in s["calls"]]
            writes = [c for c in calls if c["function"] == "write_file"
                      and c["parameters"].get("name") == expected
                      and c["parameters"].get("text", "").strip()]
            phases.append({"cycle": cycle, "kind": kind,
                           "ended": bool(group) and all(s["response_completed"] for s in group),
                           "expected_file_written": bool(writes), "calls": calls,
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
        "halted": record["halted"], "token_count": len(tokens),
        "seconds": record["seconds"], "expected_phases": meta["cycles"] * 3,
        "expected_reads": sum(s["op"] == "read" for s in meta["stages"]),
        "phases": phases,
        "stages": stages,
        "reads": reads,
        "filesystem_pass": bool(reads) and all(r["exact"] for r in reads),
        "complete_cycles": sum(all(p["expected_file_written"] and p["ended"]
                                   for p in phases if p["cycle"] == c)
                               and sum(p["cycle"] == c for p in phases) == 3
                               for c in range(1, meta["cycles"] + 1)),
        "controller_pass": (record["halted"] and len(stages) == len(meta["rom"])
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
    args = parser.parse_args()
    result = inspect(args.run)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
