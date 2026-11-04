"""Generic exact state-chain check, separate from the compiled model/controller.

Example:
  uv run python -m tests.test_agent_chain_goal results/run.json \
    pine pine-red pine-red-blue
"""
import argparse
import json
import re

from qfs.agent_evaluate import inspect


def check(path, states, checkpoint=None):
    report = inspect(path, checkpoint)
    execute = [phase for phase in report["phases"] if phase["kind"] == "execute"]
    checks = []
    for phase, before, after in zip(execute, states, states[1:]):
        results = [call["parameters"].get("text", "") for call in phase["calls"]
                   if call["function"] == "write_file"
                   and call["parameters"].get("name") == "result.txt"]
        text = results[-1] if results else ""
        plain = re.sub(r"[*`\"']", "", text)
        input_ok = bool(re.search(r"\bInput\s*:\s*" + re.escape(before) +
                                  r"(?![\w-])", plain, re.I))
        output_ok = bool(re.search(r"\bOutput\s*:\s*" + re.escape(after) +
                                   r"(?![\w-])", plain, re.I))
        checks.append({"cycle": phase["cycle"], "expected_input": before,
                       "expected_output": after, "input_correct": input_ok,
                       "output_correct": output_ok, "actual": text})
    expected = len(states) - 1
    return {"run": path, "cycles": checks,
            "pass": len(execute) == expected and len(checks) == expected
                    and all(c["input_correct"] and c["output_correct"] for c in checks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("states", nargs="+", help="Expected states, including the initial state")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    if len(args.states) < 2:
        parser.error("provide at least an initial and final state")
    result = check(args.run, args.states, args.checkpoint)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
