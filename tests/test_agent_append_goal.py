"""Objective-specific correctness check, separate from circuit/IO checks.

The model program never imports this module and contains none of these labels.
"""
import argparse
import json
import re

from qfs.agent_evaluate import inspect


def check(path, checkpoint=None):
    report = inspect(path, checkpoint)
    chain = ("oak", "oak-red", "oak-red-blue", "oak-red-blue-gold")
    checks = []
    for phase in report["phases"]:
        if phase["kind"] != "execute":
            continue
        cycle = phase["cycle"]
        results = [c["parameters"].get("text", "") for c in phase["calls"]
                   if c["function"] == "write_file" and c["parameters"].get("name") == "result.txt"]
        text = results[-1] if results else ""
        plain = re.sub(r"[*`\"']", "", text)
        before, after = chain[cycle - 1:cycle + 1]
        input_ok = bool(re.search(r"\bInput\s*:\s*" + re.escape(before) + r"(?![\w-])", plain, re.I))
        output_ok = bool(re.search(r"\bOutput\s*:\s*" + re.escape(after) + r"(?![\w-])", plain, re.I))
        checks.append({"cycle": cycle, "expected_input": before, "expected_output": after,
                       "input_correct": input_ok, "output_correct": output_ok, "actual": text})
    return {"run": path, "cycles": checks,
            "pass": len(checks) == 3 and all(c["input_correct"] and c["output_correct"] for c in checks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    result = check(args.run, args.checkpoint)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
