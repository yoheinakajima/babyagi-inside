"""Longer-task screen on Qwen3.5-4B: the published records, scores, caps, and comparisons.

    uv run python -m tests.test_long_tasks
"""
import hashlib
import json
from pathlib import Path

from qfs.benchmark import load_suite, score_item
from qfs.end_to_end import _paired_summary

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
RESULTS = ROOT / "results" / "long-tasks-4b"
V4_ARMS = ("H-adapt", "H-oracle", "P-F2")
# Chains per arm, and each paired difference's delta and bootstrap 95% CI (analysis.json).
CHAINS = {"H-adapt": 20, "H-oracle": 25, "P-F2": 38, "H-adapt-v3": 17}
PAIRED = {"adapt_minus_plain": ("H-adapt", "P-F2", -0.375, [-0.520833, -0.229167]),
          "oracle_minus_plain": ("H-oracle", "P-F2", -0.270833, [-0.395833, -0.145833]),
          "adapt_minus_oracle": ("H-adapt", "H-oracle", -0.104167, [-0.1875, -0.020833])}


def v4_items():
    """v4 lengths exceed load_suite's bound of 8, so the frozen digest is checked here."""
    path = BENCHMARKS / "babyagi-v4-horizon.json"
    sums = dict(reversed(line.split("  ", 1))
                for line in (BENCHMARKS / "babyagi-v4-horizon.sha256").read_text().splitlines())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sums[path.name]
    return {item["id"]: item for item in json.loads(path.read_text())["items"]}


def check_records():
    rows = [line.split("  ", 1) for line in (RESULTS / "SHA256SUMS").read_text().splitlines()]
    assert len(rows) == 3 * 48 + 32 + 1
    for digest, name in rows:
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest, name
    v4 = v4_items()
    v3 = {item["id"]: item for item in load_suite(BENCHMARKS / "babyagi-v3.json")["items"]}
    verdicts = {}
    for arm, items in [(arm, v4) for arm in V4_ARMS] + [("H-adapt-v3", v3)]:
        records = {path.stem: json.loads(path.read_text())
                   for path in sorted((RESULTS / arm).glob("*.json"))}
        assert set(records) == set(items), arm
        for item_id, record in records.items():
            item = items[item_id]
            assert record["identity"]["arm"] == arm
            assert record["score"] == score_item(item, record["actual_results"]), item_id
            if arm == "P-F2":  # the loop's maximum model-token allowance at the gold length
                identity, generation = record["identity"], record["generation"]
                assert identity["think_cap"] == 600 + 256 * item["cycles"]
                assert identity["answer_cap"] == 241 * item["cycles"]
                assert generation["thinking_tokens"] <= identity["think_cap"]
                assert generation["answer_tokens"] <= identity["answer_cap"]
            else:
                assert record["trace"]["cycles"] <= 40
        verdicts[arm] = {key: record["score"]["automated_pass"] for key, record in records.items()}
        assert sum(verdicts[arm].values()) == CHAINS[arm], arm
    return verdicts


def scored(verdicts):
    return {key: {"score": {"automated_pass": value}} for key, value in verdicts.items()}


def check_comparisons(verdicts):
    analysis = json.loads((RESULTS / "analysis.json").read_text())
    assert analysis["decision"]["decision"] == "NO-GO" and analysis["complete"]
    for name, (after, before, delta, interval) in PAIRED.items():
        stats = _paired_summary(scored(verdicts[before]), scored(verdicts[after]))
        assert stats["delta"] == delta and stats["item_bootstrap_delta_95"] == interval, name
        assert stats["host_improvements"] == 0
        assert analysis["comparisons"][name]["delta"] == delta
        assert analysis["comparisons"][name]["item_bootstrap_delta_95"] == interval
    compiled = json.loads((ROOT / "results/plain-forced-v3/compiled.json").read_text())
    reference = {key: row["chain_pass"] for key, row in compiled["4b"]["records"].items()}
    stats = _paired_summary(scored(reference), scored(verdicts["H-adapt-v3"]))
    assert stats["delta"] == -0.34375 and stats["host_improvements"] == 0
    return {name: PAIRED[name][2] for name in PAIRED}


def check():
    verdicts = check_records()
    deltas = check_comparisons(verdicts)
    print({"records": "PASS", "chains": {arm: sum(v.values()) for arm, v in verdicts.items()},
           "deltas": deltas, "decision": "NO-GO"})


if __name__ == "__main__":
    check()
