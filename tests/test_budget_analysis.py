"""Cheap checks for paired budget-curve statistics."""
from qfs.analyze_budget import exact_mcnemar, grouped_accuracy, paired_comparison


def _record(item, cycle, correct, category="c", length=2):
    return {"identity": {"item_id": item, "cycle": cycle},
            "correct": correct, "category": category,
            "chain_length": length, "parse_ok": True}


def check():
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(6, 17) < 0.05
    low = {
        ("a", 1): _record("a", 1, False, "x", 2),
        ("a", 2): _record("a", 2, True, "x", 2),
        ("b", 1): _record("b", 1, False, "y", 3),
    }
    high = {
        ("a", 1): _record("a", 1, True, "x", 2),
        ("a", 2): _record("a", 2, False, "x", 2),
        ("b", 1): _record("b", 1, True, "y", 3),
    }
    comparison = paired_comparison(1, 2, low, high, bootstrap_samples=100)
    assert comparison["improvements"] == 2 and comparison["regressions"] == 1
    assert comparison["delta"] == 0.333333
    grouped = grouped_accuracy(high, "chain_length")
    assert grouped["2"]["steps"] == 2 and grouped["3"]["correct"] == 1
    print({"paired": "PASS", "grouped": "PASS"})


if __name__ == "__main__":
    check()
