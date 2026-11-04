"""Cheap invariants for the frozen BabyAGI publication benchmark."""
import json
from pathlib import Path
import shutil
import tempfile

from qfs.benchmark import (BenchmarkError, extract_output, item_by_id, load_suite,
                           results_from_evaluation, score_item)
from qfs.execute_budget import gold_entries


ROOT = Path(__file__).parents[1]
FROZEN = ROOT / "benchmarks" / "babyagi-v1.json"
CONFIRMATION = ROOT / "benchmarks" / "babyagi-v2.json"
DEVELOPMENT = ROOT / "benchmarks" / "babyagi-v1-dev.json"
GRADIENT = ROOT / "benchmarks" / "babyagi-v3.json"


def check():
    heldout = load_suite(FROZEN)
    confirmation = load_suite(CONFIRMATION)
    development = load_suite(DEVELOPMENT)
    gradient = load_suite(GRADIENT)
    assert heldout["benchmark_id"] == "babyagi-heldout-v1"
    assert heldout["status"] == "frozen"
    assert heldout["frozen_on"] == "2026-11-03"
    assert len(heldout["items"]) == 12
    assert set(heldout["category_counts"].values()) == {3}
    assert confirmation["benchmark_id"] == "babyagi-confirmation-v2"
    assert confirmation["status"] == "frozen"
    assert len(confirmation["items"]) == 12
    assert set(confirmation["category_counts"].values()) == {3}
    assert {item["id"] for item in heldout["items"]}.isdisjoint(
        item["id"] for item in confirmation["items"])
    assert development["status"] == "mutable"
    assert {item["id"] for item in heldout["items"]}.isdisjoint(
        item["id"] for item in development["items"])
    assert gradient["benchmark_id"] == "babyagi-execution-gradient-v3"
    assert gradient["status"] == "frozen" and len(gradient["items"]) == 32
    assert set(gradient["category_counts"].values()) == {8}
    assert {item["cycles"] for item in gradient["items"]} == {2, 3, 5, 8}
    assert len(gold_entries(gradient)) == 144
    pilot = gold_entries(gradient, max_steps_per_category=8)
    assert len(pilot) == 32
    for category in gradient["category_counts"]:
        cycles = [entry["step"]["cycle"] for entry in pilot
                  if entry["item"]["category"] == category]
        assert cycles == list(range(1, 9))

    heldout_text = json.dumps(heldout).lower()
    for development_only in ("oak", "pine", "elm-hot", "luma"):
        assert development_only not in heldout_text
    compiled_prompts = (ROOT / "qfs" / "agent.py").read_text().lower()
    for sentinel in heldout["leak_sentinels"]:
        assert sentinel.lower() not in compiled_prompts
    for sentinel in confirmation["leak_sentinels"]:
        assert sentinel.lower() not in compiled_prompts
    for sentinel in gradient["leak_sentinels"]:
        assert sentinel.lower() not in compiled_prompts

    literal = item_by_id(heldout, "literal-moss-3")
    report = score_item(literal, ["[moss]", "[MOSS]", "[MOSS]!"])
    assert report["pass"] is True
    assert score_item(literal, ["[moss]", "[MOSS]", "[MOSS]"])["pass"] is False
    assert extract_output("Output:  pine-red\ncontinued") == "pine-red\ncontinued"
    evaluation = {"phases": [
        {"kind": "create", "calls": []},
        {"kind": "execute", "calls": [{"function": "write_file", "parameters": {
            "name": "result.txt", "text": "Output:  pine-red"}}]},
        {"kind": "execute", "calls": [{"function": "write_file", "parameters": {
            "name": "result.txt", "text": "Output:  pine-red-blue"}}]},
        {"kind": "execute", "calls": [{"function": "write_file", "parameters": {
            "name": "result.json", "text": '{"result":[1,2,3]}'}}]},
    ]}
    assert results_from_evaluation(evaluation) == ["pine-red", "pine-red-blue", [1, 2, 3]]

    arithmetic = item_by_id(heldout, "arithmetic-seven-3")
    assert score_item(arithmetic, ["12.0", "36", "28"])["pass"] is True

    structured = item_by_id(heldout, "structured-numbers-3")
    structured_results = ["[1, 3, 3, 5, 8]", "[1,3,5,8]", "[1,3,5]"]
    assert score_item(structured, structured_results)["pass"] is True

    acrostic = item_by_id(heldout, "language-day-acrostic-3")
    poem = "Dawn paints clouds\nAmber light rises\nYellow birds awaken"
    report = score_item(acrostic, ["draft", "revision", poem])
    assert report["automated_pass"] is True
    assert report["manual_required"] is True and report["pass"] is None
    manual = {criterion: True for criterion in acrostic["rubric"]["manual_criteria"]}
    assert score_item(acrostic, ["draft", "revision", poem], manual)["pass"] is True

    announcement = item_by_id(heldout, "language-tidetally-announcement-5")
    final = ("Headline: Meet TideTally\n"
             "Body: TideTally turns imaginary tide patterns into a family curiosity prompt. "
             "It invites questions without claiming scientific accuracy.\n"
             "CTA: Imagine your next shoreline question together.")
    report = score_item(announcement, ["a", "b", "c", "d", final])
    assert report["automated_pass"] is True and report["pass"] is None

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / FROZEN.name
        shutil.copyfile(FROZEN, target)
        shutil.copyfile(FROZEN.with_suffix(".sha256"), target.with_suffix(".sha256"))
        target.write_text(target.read_text() + "\n")
        try:
            load_suite(target)
        except BenchmarkError as exc:
            assert "checksum mismatch" in str(exc)
        else:
            raise AssertionError("modified frozen benchmark passed checksum validation")

    print({"benchmark": heldout["benchmark_id"], "heldout": len(heldout["items"]),
           "development": len(development["items"]), "categories": heldout["category_counts"]})


if __name__ == "__main__":
    check()
