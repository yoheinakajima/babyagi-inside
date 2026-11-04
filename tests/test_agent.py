"""Independent structural checks for a generated end-to-end agent trace.

Run after decoding: uv run python -m tests.test_agent results/word-seed0.json
These checks do not run/influence the agent or substitute scripted LM answers.
"""
import argparse
import json
from pathlib import Path

from qfs.agent_evaluate import inspect


def check(run_path, checkpoint=None):
    result = inspect(run_path, checkpoint)
    assert result["controller_pass"], "ROM framing, phase order, or bounded HALT failed"
    assert result["complete_cycles"] == result["expected_phases"] // 3, "Missing phase writes"
    assert result["filesystem_pass"], "A read differed from the prior stored token sequence"
    assert len(result["reads"]) == result["expected_reads"], "Missing compiled reads"
    assert result["write_count"] == result["expected_writes"], "Missing model-generated write acks"
    for phase in result["phases"]:
        expected_reads = {
            stage["file"] for stage in result["stages"]
            if stage["cycle"] == phase["cycle"] and stage["kind"] == phase["kind"]
            and stage["op"] == "read"
        }
        actual_reads = {
            call["parameters"].get("name") for call in phase["calls"]
            if call["function"] == "read_file"
        }
        assert expected_reads <= actual_reads, f"Missing reads: {expected_reads - actual_reads}"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    for path in args.runs:
        result = check(path, args.checkpoint)
        print(json.dumps({"run": str(Path(path)), "controller": "PASS",
                          "cycles": result["complete_cycles"],
                          "exact_reads": len(result["reads"]),
                          "tokens": result["token_count"]}))


if __name__ == "__main__":
    main()
