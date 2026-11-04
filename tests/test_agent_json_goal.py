"""Check a JSON-ledger BabyAGI trace against an expected result chain."""
import argparse
import json
from pathlib import Path

from qfs.agent_evaluate import inspect


def check(run_path, expected_results):
    inspected = inspect(run_path)
    assert inspected["protocol"] == "babyagi_json"
    execute_phases = [phase for phase in inspected["phases"] if phase["kind"] == "execute"]
    assert len(execute_phases) == len(expected_results)
    reports = []
    prior_completed = []
    passed = True
    for phase, expected in zip(execute_phases, expected_results):
        writes = [call for call in phase["calls"] if call["function"] == "write_file"
                  and call["parameters"].get("name") == "tasks.json"]
        raw = writes[-1]["parameters"].get("text", "") if writes else ""
        try:
            ledger = json.loads(raw)
        except json.JSONDecodeError:
            ledger = None
        schema_ok = bool(ledger) and set(ledger) == {
            "pending", "completed", "result",
        }
        pending_ok = bool(schema_ok and isinstance(ledger["pending"], list)
                          and all(isinstance(item, dict) and set(item) == {"id", "task"}
                                  and isinstance(item["id"], str)
                                  and isinstance(item["task"], str)
                                  for item in ledger["pending"]))
        completed_ok = bool(schema_ok and isinstance(ledger["completed"], list)
                            and all(isinstance(task, str) for task in ledger["completed"]))
        newly_completed = (ledger["completed"][-1]
                           if completed_ok and ledger["completed"] else None)
        pending_tasks = [item["task"] for item in ledger["pending"]] if pending_ok else []
        lifecycle_ok = bool(pending_ok and completed_ok
                            and ledger["completed"][:len(prior_completed)] == prior_completed
                            and len(ledger["completed"]) == len(prior_completed) + 1
                            and newly_completed not in pending_tasks)
        result_ok = bool(schema_ok and ledger["result"] == expected)
        if completed_ok:
            prior_completed = ledger["completed"]
        passed &= schema_ok and lifecycle_ok and result_ok
        reports.append({
            "cycle": phase["cycle"], "expected_result": expected,
            "schema_ok": schema_ok, "lifecycle_ok": lifecycle_ok,
            "result_ok": result_ok, "actual": ledger if ledger is not None else raw,
        })
    report = {"run": str(Path(run_path)), "cycles": reports, "pass": bool(passed)}
    print(json.dumps(report, indent=2))
    assert passed, report
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("expected_results", nargs="+")
    args = parser.parse_args()
    check(args.run, args.expected_results)


if __name__ == "__main__":
    main()
