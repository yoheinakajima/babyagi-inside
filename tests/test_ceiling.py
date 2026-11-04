"""Cheap parsing and counterbalance tests for ceiling baselines."""
from qfs.ceiling import (add_phase_diagnostics, counterbalanced_position,
                         parse_choice, parse_create, parse_plain, parse_result)


def check():
    assert parse_plain('reason\nFINAL_JSON: {"results":[12,36,28]}', 3) == [12, 36, 28]
    assert parse_plain("[1, 2] then [3, 4, 5]", 3) == [3, 4, 5]
    assert parse_plain("no array", 3) is None
    created = parse_create(
        'CREATE_JSON: {"next_task":"Add 5 to 7","distractor_task":"Verify 7"}')
    assert created == {"next_task": "Add 5 to 7", "distractor_task": "Verify 7"}
    assert parse_choice("consider both\nCHOOSE: 2") == 2
    assert parse_result('RESULT_JSON: {"result":[1,3,5]}') == [1, 3, 5]
    positions = [counterbalanced_position(0, cycle) for cycle in range(1, 6)]
    assert positions == [1, 2, 1, 2, 1]

    item = {"id": "test", "category": "sequential_arithmetic", "cycles": 2,
            "objective": "", "rubric": {"type": "exact_state_chain",
                                           "normalization": "number",
                                           "expected_results": [12, 36]}}
    trace = {"actual_results": [12, 35], "cycles": [
        {"cycle": 1, "execute": {"parse_ok": True}},
        {"cycle": 2, "execute": {"parse_ok": True}},
    ]}
    add_phase_diagnostics(item, trace)
    assert trace["cycles"][0]["execute"]["result_prefix_correct"] is True
    assert trace["cycles"][1]["execute"]["result_prefix_correct"] is False
    print({"parsers": "PASS", "counterbalance": positions, "diagnostics": "PASS"})


if __name__ == "__main__":
    check()
