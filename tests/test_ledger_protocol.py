"""Cheap invariants for the selected host-side ledger protocol."""
from pathlib import Path

from qfs.benchmark import load_suite
from qfs.ledger_protocol import candidate_prompt, parse_ledger
from qfs.oracle_execute import ORACLE_TASKS


ROOT = Path(__file__).parents[1]


def check():
    text = ('PLAN_JSON: {"initial":7,"tasks":["Add 5",'
            '"Multiply by 3","Subtract 8"]}')
    ledger = parse_ledger(text, 3)
    assert ledger == {"initial": 7,
                      "tasks": ["Add 5", "Multiply by 3", "Subtract 8"]}
    assert parse_ledger('{"initial":7,"tasks":["Add 5"]}', 3) is None
    assert parse_ledger('{"tasks":["a","b","c"]}', 3) is None

    v1 = load_suite(ROOT / "benchmarks" / "babyagi-v1.json")
    assert set(ORACLE_TASKS) == {item["id"] for item in v1["items"]}
    for item in v1["items"]:
        assert len(ORACLE_TASKS[item["id"]]) == item["cycles"]

    prompt = candidate_prompt(v1["items"][0], "cinder-north", ["Append -north"],
                              "Append -east")
    assert "Candidate task: Append -east" in prompt
    assert "Completed tasks" in prompt
    assert "Candidate 1" not in prompt and "Candidate 2" not in prompt
    print({"ledger_parse": "PASS", "oracle_task_maps": len(ORACLE_TASKS),
           "selector": "independent"})


if __name__ == "__main__":
    check()
