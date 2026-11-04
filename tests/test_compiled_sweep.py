"""Cheap checks for the frozen full live-compiled sweep."""
import json
from pathlib import Path

from qfs.benchmark import load_suite
from qfs.compiled_sweep import _group_summary, _plan_score, _wilson
from qfs.plan_fidelity import semantic_plan_score, task_signature


def check():
    suite = load_suite(Path("benchmarks/babyagi-v3.json"))
    gold_tasks = [step["task"] for item in suite["items"]
                  for step in item["provenance"]["gold_steps"]]
    assert all(task_signature(task)[0] != "unparsed" for task in gold_tasks)

    item = next(item for item in suite["items"]
                if item["id"] == "v3-arithmetic-multiply-2")
    ledger = {"initial": 7,
              "tasks": ["multiply 7 by 3", "add 4 to the result"]}
    evaluation = {"latest_files": {
        "ledger.json": {"version": 1, "text": json.dumps(ledger)}}}
    plan = _plan_score(item, evaluation)
    assert plan["parse_ok"] and plan["initial_exact"]
    assert not plan["tasks_exact"] and not plan["exact"]
    assert plan["exploratory_semantic"]["exact"]

    bad = semantic_plan_score(
        ["sort it in ascending numeric order", "remove duplicates while preserving order"],
        ["add 18", "add 1"])
    assert not bad["exact"] and bad["correct_tasks"] == 0

    rows = [
        {"category": "a", "cycles": 2, "system_pass": True,
         "benchmark_pass": True, "correct_steps": 2, "total_steps": 2},
        {"category": "a", "cycles": 3, "system_pass": True,
         "benchmark_pass": False, "correct_steps": 1, "total_steps": 3},
    ]
    grouped = _group_summary(rows, "category")
    assert grouped["a"]["system_pass"] == 2
    assert grouped["a"]["chain_pass"] == 1
    assert grouped["a"]["step_accuracy"] == 0.6
    assert _wilson(0, 0) == [None, None]
    print({"gold_signatures": len(gold_tasks), "plan_metrics": "PASS",
           "sweep_summary": "PASS"})


if __name__ == "__main__":
    check()
