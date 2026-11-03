"""Independent structural checks for a generated end-to-end agent trace.

Run after decoding: uv run python -m tests.test_agent results/word-seed0.json
These checks do not run/influence the agent or substitute scripted LM answers.
"""
import argparse
import json
from pathlib import Path

from qfs.agent_evaluate import inspect


def check(run_path):
    result = inspect(run_path)
    assert result["controller_pass"], "ROM framing, phase order, or bounded HALT failed"
    assert result["complete_cycles"] == result["expected_phases"] // 3, "Missing phase writes"
    assert result["filesystem_pass"], "A read differed from the prior stored token sequence"
    assert len(result["reads"]) == result["expected_reads"], "Missing compiled reads"
    for phase in result["phases"]:
        if phase["kind"] == "create" and phase["cycle"] > 1:
            assert any(c["function"] == "read_file" and c["parameters"].get("name") == "result.txt"
                       for c in phase["calls"]), "Later creation did not read previous result"
        if phase["kind"] in ("prioritize", "execute"):
            name = "tasks.txt" if phase["kind"] == "prioritize" else "current.txt"
            assert any(c["function"] == "read_file" and c["parameters"].get("name") == name
                       for c in phase["calls"]), f"Missing read of {name}"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    for path in args.runs:
        result = check(path)
        print(json.dumps({"run": str(Path(path)), "controller": "PASS",
                          "cycles": result["complete_cycles"],
                          "exact_reads": len(result["reads"]),
                          "tokens": result["token_count"]}))


if __name__ == "__main__":
    main()
