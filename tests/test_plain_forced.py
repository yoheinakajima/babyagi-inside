"""Budget-forced plain arms: prompt, caps, close tokens, and the published records.

    uv run python -m tests.test_plain_forced
"""
import hashlib
import json
from pathlib import Path

from qfs import ceiling, plain_forced
from qfs.benchmark import load_suite, score_item

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks" / "babyagi-v3.json"
RESULTS = ROOT / "results" / "plain-forced-v3"
# Plain chains, compiled minus plain delta, and its bootstrap 95% CI, per model and arm.
EXPECTED = {
    "4b/F1": (29, -0.03125, [-0.09375, 0.0]), "4b/F2": (29, -0.03125, [-0.09375, 0.0]),
    "4b/F3": (29, -0.03125, [-0.09375, 0.0]), "2b/F1": (9, -0.03125, [-0.09375, 0.0]),
    "2b/F2": (12, -0.125, [-0.25, -0.03125]), "2b/F3": (15, -0.21875, [-0.375, -0.09375]),
}


def check_prompt(items):
    seen = []
    original = ceiling.generate_chat
    ceiling.generate_chat = lambda model, tok, prompt, *rest: seen.append(prompt) or {"text": ""}
    try:
        for item in items:
            ceiling.run_plain(None, None, item, "cpu", True, 1)
    finally:
        ceiling.generate_chat = original
    assert seen == [plain_forced.plain_prompt(item) for item in items]


def check_caps():
    assert plain_forced.caps("F1", 8, 0) == (2048, 256)  # the one-turn ceiling, 2,304 in all
    assert plain_forced.caps("F2", 8, 0) == (2304, 2048)  # the loop's allowance, 4,352
    assert plain_forced.caps("F3", 5, 3376) == (3376 - 1280, 1280)
    try:
        plain_forced.caps("F4", 2, 0)
    except ValueError:
        return
    raise AssertionError("unknown arm accepted")


def check_close_tokens():
    if not (ROOT / "models" / "Qwen3.5-4B" / "tokenizer.json").exists():
        return "skipped (no local tokenizers)"
    from transformers import AutoTokenizer

    for name in ("Qwen3.5-2B", "Qwen3.5-4B"):
        tok = AutoTokenizer.from_pretrained(ROOT / "models" / name)
        assert tok.convert_tokens_to_ids("</think>") == plain_forced.THINK_END
        assert tok.encode("\n</think>\n\n", add_special_tokens=False) == plain_forced.CLOSE
    return "PASS"


def check_records(items):
    rows = [line.split("  ", 1) for line in (RESULTS / "SHA256SUMS").read_text().splitlines()]
    assert len(rows) == 1 + 2 * 3 * 32
    for digest, name in rows:
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest, name
    compiled = json.loads((RESULTS / "compiled.json").read_text())
    by_id = {item["id"]: item for item in items}
    for model in ("2b", "4b"):
        for arm in plain_forced.ARMS:
            for path in sorted((RESULTS / model / arm).glob("*.json")):
                record = json.loads(path.read_text())
                item = by_id[record["item_id"]]
                tokens = compiled[model]["records"][item["id"]]["tokens"]
                identity = record["identity"]
                assert (identity["think_cap"], identity["answer_cap"]) == \
                    plain_forced.caps(arm, item["cycles"], tokens)
                generation = record["generation"]
                assert generation["thinking_tokens"] <= identity["think_cap"]
                assert generation["answer_tokens"] <= identity["answer_cap"]
                rescored = plain_forced.score_record(item, generation)
                assert rescored["actual_results"] == record["actual_results"]
                assert rescored["score"] == record["score"]
                assert record["score"] == score_item(item, record["actual_results"])
    return len(rows) - 1


def check_comparisons():
    result = plain_forced.compare(RESULTS / "compiled.json", RESULTS)
    assert set(result) == set(EXPECTED)
    for key, (plain, delta, interval) in EXPECTED.items():
        row, paired = result[key], result[key]["compiled_minus_plain"]
        assert row["plain_chains"] == plain and row["forced_close"] == 32
        assert paired["delta"] == delta and paired["item_bootstrap_delta_95"] == interval
        assert paired["host_improvements"] == 0
    assert result["4b/F2"]["compiled_chains"] == 28 and result["2b/F2"]["compiled_chains"] == 8
    return {key: result[key]["plain_chains"] for key in sorted(result)}


def check():
    items = load_suite(SUITE)["items"]
    check_prompt(items)
    check_caps()
    close = check_close_tokens()
    records = check_records(items)
    chains = check_comparisons()
    print({"prompt": "PASS", "caps": "PASS", "close_tokens": close, "records": records,
           "plain_chains": chains})


if __name__ == "__main__":
    check()
