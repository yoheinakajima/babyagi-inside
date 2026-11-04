"""Frozen BabyAGI benchmark loading, validation, and rubric scoring.

The benchmark is evaluation data. Nothing in this module is imported by the
compiled agent or made available to its prompts.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re


CATEGORIES = (
    "ordered_literal",
    "sequential_arithmetic",
    "structured_operation",
    "checkable_language",
)
RUBRIC_TYPES = ("exact_state_chain", "final_constraints")
CONSTRAINTS = {
    "case_sensitive",
    "forbidden_literals",
    "line_count",
    "line_initials",
    "max_words",
    "min_words",
    "ordered_literals",
    "required_literals",
    "section_sentence_counts",
    "sentence_count",
    "words_per_line",
}
ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


class BenchmarkError(ValueError):
    """A suite or result violates the benchmark contract."""


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checksum_path(path):
    return Path(path).with_suffix(".sha256")


def verify_checksum(path):
    path = Path(path)
    sidecar = checksum_path(path)
    try:
        parts = sidecar.read_text().strip().split()
    except OSError as exc:
        raise BenchmarkError(f"missing checksum sidecar: {sidecar}") from exc
    if len(parts) != 2 or parts[1] != path.name or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
        raise BenchmarkError(f"invalid checksum sidecar: {sidecar}")
    actual = file_sha256(path)
    if actual != parts[0]:
        raise BenchmarkError(f"frozen benchmark checksum mismatch: wanted {parts[0]}, got {actual}")
    return actual


def _require(condition, message):
    if not condition:
        raise BenchmarkError(message)


def _validate_rubric(item):
    rubric = item.get("rubric")
    _require(isinstance(rubric, dict), f"{item['id']}: rubric must be an object")
    kind = rubric.get("type")
    _require(kind in RUBRIC_TYPES, f"{item['id']}: unsupported rubric type {kind!r}")
    if kind == "exact_state_chain":
        _require(set(rubric) == {"type", "normalization", "expected_results"},
                 f"{item['id']}: invalid exact-state rubric fields")
        _require(rubric["normalization"] in {"text", "number", "json"},
                 f"{item['id']}: invalid normalization")
        expected = rubric["expected_results"]
        _require(isinstance(expected, list) and len(expected) == item["cycles"],
                 f"{item['id']}: expected_results must have one value per cycle")
    else:
        _require(set(rubric) <= {"type", "automated", "manual_criteria"},
                 f"{item['id']}: invalid constraint-rubric fields")
        automated = rubric.get("automated", {})
        manual = rubric.get("manual_criteria", [])
        _require(isinstance(automated, dict) and set(automated) <= CONSTRAINTS,
                 f"{item['id']}: unsupported automated constraint")
        _require(isinstance(manual, list) and all(isinstance(x, str) and x for x in manual),
                 f"{item['id']}: manual criteria must be nonempty strings")
        _require(automated or manual, f"{item['id']}: empty rubric")


def validate_suite(suite):
    _require(isinstance(suite, dict), "suite must be an object")
    required = {
        "schema_version", "benchmark_id", "status", "description", "objective_count",
        "category_counts", "leak_sentinels", "items",
    }
    allowed = required | {"frozen_on"}
    _require(required <= set(suite) and set(suite) <= allowed, "invalid suite fields")
    _require(suite["schema_version"] == 1, "unsupported schema version")
    _require(ID_RE.fullmatch(suite["benchmark_id"] or ""), "invalid benchmark_id")
    _require(suite["status"] in {"frozen", "mutable"}, "invalid benchmark status")
    _require(isinstance(suite["description"], str) and suite["description"],
             "description is required")
    if suite["status"] == "frozen":
        _require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", suite.get("frozen_on", "")),
                 "frozen suite requires frozen_on")
    counts = suite["category_counts"]
    _require(isinstance(counts, dict) and set(counts) == set(CATEGORIES),
             "category_counts must name every category")
    _require(all(isinstance(v, int) and v >= 0 for v in counts.values()),
             "category counts must be nonnegative integers")
    sentinels = suite["leak_sentinels"]
    _require(isinstance(sentinels, list)
             and all(isinstance(x, str) and x for x in sentinels),
             "leak_sentinels must be strings")
    items = suite["items"]
    _require(isinstance(items, list) and len(items) == suite["objective_count"],
             "objective_count does not match items")
    ids = set()
    observed = {category: 0 for category in CATEGORIES}
    for item in items:
        _require(isinstance(item, dict), "each item must be an object")
        item_required = {"id", "category", "cycles", "objective", "rubric"}
        _require(item_required <= set(item) <= item_required | {"provenance"},
                 "invalid item fields")
        item_id = item.get("id", "")
        _require(ID_RE.fullmatch(item_id), f"invalid item id {item_id!r}")
        _require(item_id not in ids, f"duplicate item id {item_id}")
        ids.add(item_id)
        category = item.get("category")
        _require(category in CATEGORIES, f"{item_id}: invalid category")
        observed[category] += 1
        _require(isinstance(item.get("cycles"), int) and 1 <= item["cycles"] <= 5,
                 f"{item_id}: cycles must be 1..5")
        _require(isinstance(item.get("objective"), str) and item["objective"].strip(),
                 f"{item_id}: objective is required")
        _validate_rubric(item)
    _require(observed == counts, f"category counts mismatch: {observed}")
    _require(sum(counts.values()) == suite["objective_count"], "category total mismatch")
    return suite


def load_suite(path, verify_frozen=True):
    path = Path(path)
    try:
        suite = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read benchmark: {path}") from exc
    validate_suite(suite)
    if verify_frozen and suite["status"] == "frozen":
        verify_checksum(path)
    return suite


def item_by_id(suite, item_id):
    try:
        return next(item for item in suite["items"] if item["id"] == item_id)
    except StopIteration as exc:
        raise BenchmarkError(f"unknown benchmark item: {item_id}") from exc


def extract_output(text):
    """Remove the result-file envelope while preserving the semantic payload."""
    text = str(text).strip()
    lines = text.splitlines()
    if lines and lines[0].lstrip().lower().startswith("output:"):
        first = lines[0].split(":", 1)[1].strip()
        return "\n".join([first, *lines[1:]]).strip()
    return text


def results_from_evaluation(report):
    """Extract one completed result-file value per cycle from an inspected trace."""
    results = []
    for phase in report.get("phases", []):
        if phase.get("kind") != "execute":
            continue
        writes = [call for call in phase.get("calls", [])
                  if call.get("function") == "write_file"
                  and call.get("parameters", {}).get("name") == "result.txt"]
        if writes:
            results.append(extract_output(writes[-1]["parameters"].get("text", "")))
    return results


def _normalize(value, mode):
    if mode == "text":
        return str(value).strip()
    if mode == "number":
        try:
            return Decimal(str(value).strip())
        except InvalidOperation as exc:
            raise BenchmarkError(f"not a number: {value!r}") from exc
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"not valid JSON: {value!r}") from exc


def _sentence_count(text):
    text = text.strip()
    if not text:
        return 0
    terminated = re.findall(r"[^.!?]+[.!?](?=\s|$)", text)
    remainder = re.sub(r"[^.!?]+[.!?](?=\s|$)", "", text).strip()
    return len(terminated) + bool(remainder)


def _sections(text, names):
    names = list(names)
    pattern = re.compile(r"(?im)^(" + "|".join(map(re.escape, names)) + r")\s*:\s*")
    matches = list(pattern.finditer(text))
    out = {}
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        out[match.group(1).lower()] = text[match.end():stop].strip()
    return out


def _constraint_checks(text, constraints):
    case_sensitive = constraints.get("case_sensitive", False)
    haystack = text if case_sensitive else text.lower()
    convert = (lambda x: x) if case_sensitive else (lambda x: x.lower())
    checks = []

    def add(name, passed, expected, actual):
        checks.append({"name": name, "pass": bool(passed),
                       "expected": expected, "actual": actual})

    required = constraints.get("required_literals", [])
    for literal in required:
        add(f"required:{literal}", convert(literal) in haystack, True,
            convert(literal) in haystack)
    forbidden = constraints.get("forbidden_literals", [])
    for literal in forbidden:
        add(f"forbidden:{literal}", convert(literal) not in haystack, False,
            convert(literal) in haystack)
    ordered = constraints.get("ordered_literals", [])
    positions, start = [], 0
    for literal in ordered:
        pos = haystack.find(convert(literal), start)
        positions.append(pos)
        if pos >= 0:
            start = pos + len(literal)
    if ordered:
        add("ordered_literals", all(pos >= 0 for pos in positions), ordered, positions)
    words = WORD_RE.findall(text)
    if "max_words" in constraints:
        add("max_words", len(words) <= constraints["max_words"],
            constraints["max_words"], len(words))
    if "min_words" in constraints:
        add("min_words", len(words) >= constraints["min_words"],
            constraints["min_words"], len(words))
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if "line_count" in constraints:
        add("line_count", len(lines) == constraints["line_count"],
            constraints["line_count"], len(lines))
    if "words_per_line" in constraints:
        actual = [len(WORD_RE.findall(line)) for line in lines]
        add("words_per_line", bool(lines) and all(n == constraints["words_per_line"]
                                                   for n in actual),
            constraints["words_per_line"], actual)
    if "line_initials" in constraints:
        actual = "".join(line[0] for line in lines if line).upper()
        add("line_initials", actual == constraints["line_initials"].upper(),
            constraints["line_initials"], actual)
    if "sentence_count" in constraints:
        actual = _sentence_count(text)
        add("sentence_count", actual == constraints["sentence_count"],
            constraints["sentence_count"], actual)
    section_counts = constraints.get("section_sentence_counts", {})
    if section_counts:
        sections = _sections(text, section_counts)
        for name, expected in section_counts.items():
            actual = _sentence_count(sections.get(name.lower(), ""))
            add(f"section_sentences:{name}", actual == expected, expected, actual)
    return checks


def score_item(item, actual_results, manual_scores=None):
    _require(isinstance(actual_results, list), "actual_results must be a list")
    cycle_count_ok = len(actual_results) == item["cycles"]
    rubric = item["rubric"]
    checks = []
    if rubric["type"] == "exact_state_chain":
        mode = rubric["normalization"]
        for cycle, expected in enumerate(rubric["expected_results"], 1):
            actual = actual_results[cycle - 1] if cycle <= len(actual_results) else None
            try:
                passed = cycle <= len(actual_results) and _normalize(actual, mode) == _normalize(
                    expected, mode)
            except BenchmarkError:
                passed = False
            checks.append({"name": f"cycle:{cycle}", "pass": passed,
                           "expected": expected, "actual": actual})
        automated_pass = cycle_count_ok and all(check["pass"] for check in checks)
        return {"item_id": item["id"], "cycle_count_ok": cycle_count_ok,
                "automated_checks": checks, "automated_pass": automated_pass,
                "manual_required": False, "manual_pass": None, "pass": automated_pass}

    final = str(actual_results[-1]) if actual_results else ""
    checks = _constraint_checks(final, rubric.get("automated", {}))
    automated_pass = cycle_count_ok and all(check["pass"] for check in checks)
    criteria = rubric.get("manual_criteria", [])
    manual_required = bool(criteria)
    manual_pass = None
    if manual_scores is not None:
        _require(isinstance(manual_scores, dict), "manual_scores must be an object")
        _require(set(manual_scores) == set(criteria), "manual score criteria mismatch")
        _require(all(isinstance(v, bool) for v in manual_scores.values()),
                 "manual scores must be booleans")
        manual_pass = all(manual_scores.values())
    if not manual_required:
        passed = automated_pass
    elif not automated_pass:
        passed = False
    elif manual_pass is None:
        passed = None
    else:
        passed = manual_pass
    return {"item_id": item["id"], "cycle_count_ok": cycle_count_ok,
            "automated_checks": checks, "automated_pass": automated_pass,
            "manual_required": manual_required, "manual_pass": manual_pass, "pass": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("suite", type=Path)
    score = sub.add_parser("score")
    score.add_argument("suite", type=Path)
    score.add_argument("item_id")
    score.add_argument("results", type=Path, help="JSON array with one result per cycle")
    score.add_argument("--manual", type=Path, help="JSON object mapping manual criteria to booleans")
    trace = sub.add_parser("score-evaluation")
    trace.add_argument("suite", type=Path)
    trace.add_argument("item_id")
    trace.add_argument("evaluation", type=Path,
                       help="JSON produced by qfs.agent_evaluate")
    trace.add_argument("--manual", type=Path,
                       help="JSON object mapping manual criteria to booleans")
    args = parser.parse_args()
    suite = load_suite(args.suite)
    if args.command == "validate":
        print(json.dumps({"benchmark_id": suite["benchmark_id"],
                          "status": suite["status"], "items": len(suite["items"]),
                          "sha256": file_sha256(args.suite)}, indent=2))
        return
    if args.command == "score":
        results = json.loads(args.results.read_text())
    else:
        results = results_from_evaluation(json.loads(args.evaluation.read_text()))
    manual = json.loads(args.manual.read_text()) if args.manual else None
    report = score_item(item_by_id(suite, args.item_id), results, manual)
    report["actual_results"] = results
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] is True else 1)


if __name__ == "__main__":
    main()
