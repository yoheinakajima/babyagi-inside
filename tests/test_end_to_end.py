"""Cheap checks for end-to-end summaries and deterministic trace selection."""
from qfs.end_to_end import _arm_summary, _compiled_selection, _paired_summary


def _record(arm, item, category, cycles, checks):
    score = {"automated_pass": all(checks),
             "automated_checks": [{"pass": value} for value in checks]}
    trace = ({"parse_ok": True, "generation": {"hit_limit": False}}
             if arm == "plain" else {"stopped": None})
    return {"arm": arm, "item_id": item, "category": category,
            "cycles": cycles, "score": score, "trace": trace,
            "plan": {"exact": True, "parse_ok": True}}


def check():
    plain_rows = [
        _record("plain", "a", "x", 2, [True, True]),
        _record("plain", "b", "x", 3, [True, False, False]),
    ]
    host_rows = [
        _record("host", "a", "x", 2, [True, True]),
        _record("host", "b", "x", 3, [True, True, True]),
    ]
    plain = _arm_summary(plain_rows)
    host = _arm_summary(host_rows)
    assert plain["chain_pass"] == 1 and plain["correct_steps"] == 3
    assert host["chain_pass"] == 2 and host["prefix_survival"]["3"]["survived"] == 1
    paired = _paired_summary({row["item_id"]: row for row in plain_rows},
                             {row["item_id"]: row for row in host_rows})
    assert paired["host_improvements"] == 1 and paired["host_regressions"] == 0
    suite = {"items": [{"id": "a", "cycles": 2}, {"id": "c", "cycles": 2},
                       {"id": "b", "cycles": 3}]}
    selection_rows = {row["item_id"]: row for row in host_rows}
    selection_rows["c"] = _record("host", "c", "x", 2, [False, False])
    selected = _compiled_selection(suite, selection_rows)
    assert selected["2"]["item_id"] == "a"
    assert selected["3"]["item_id"] == "b"
    print({"summary": "PASS", "paired": "PASS", "selection": "PASS"})


if __name__ == "__main__":
    check()
