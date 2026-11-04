"""Exploratory deterministic semantic signatures for v3 task ledgers.

This metric was defined after inspecting host-generated ledgers and is never a
replacement for the frozen exact-string plan metric. It canonicalizes only the
operation families present in the frozen v3 generator; unknown language remains
unparsed and must match literally to receive credit.
"""
from __future__ import annotations

import re


def _text(value):
    value = re.sub(r"^\s*(?:task\s*)?\d+[.):\-]\s*", "", str(value), flags=re.I)
    return " ".join(value.strip().lower().split())


def _literal(value):
    value = _text(value).strip(" `\"'")
    aliases = {"a period": ".", "period": ".", "a question mark": "?",
               "question mark": "?", "a hash": "#", "hash": "#"}
    return aliases.get(value, value)


def _number(value):
    words = {"one": "1", "two": "2", "three": "3", "four": "4",
             "five": "5", "six": "6", "seven": "7", "eight": "8"}
    return words.get(value, value)


def task_signature(task):
    """Return a conservative JSON-serializable operation signature."""
    text = _text(task)
    number = r"(-?\d+(?:\.\d+)?)"

    match = re.search(rf"\badd\s+{number}\s+to\s+(?:each|every)\s+(?:number|value)", text)
    if match:
        return ["map_add", match.group(1)]
    match = re.search(rf"\b(?:append|add)\s+(?:the\s+)?number\s+{number}\b", text)
    if match:
        return ["array_append", match.group(1)]
    if "sort" in text and "ascending" in text:
        return ["sort", "ascending"]
    if "sort" in text and "descending" in text:
        return ["sort", "descending"]
    if "remove" in text and "duplicate" in text:
        return ["deduplicate", "preserve_order" if "preserv" in text else None]
    if re.search(r"\bkeep\b.*\beven\b", text):
        return ["filter", "even"]
    if re.search(r"\bkeep\b.*\bodd\b", text):
        return ["filter", "odd"]
    number_word = r"(one|two|three|four|five|six|seven|eight)"
    match = re.search(rf"\bkeep\b.*\bfirst\s+(?:{number}|{number_word})\b", text)
    if match:
        return ["take_first", _number(match.group(1) or match.group(2))]
    if "reverse" in text and ("array" in text or text.startswith("reverse")):
        return ["reverse"]

    match = re.search(rf"\bmultiply\b.*?\bby\s+{number}\b", text)
    if match:
        return ["multiply", match.group(1)]
    match = re.search(rf"\badd\s+{number}\b", text)
    if match:
        return ["add", match.group(1)]
    match = re.search(rf"\bsubtract\s+{number}\b", text)
    if match:
        return ["subtract", match.group(1)]
    if re.search(r"\bdouble\b", text):
        return ["multiply", "2"]

    if "capitalize" in text and "first" in text:
        return ["capitalize_first"]
    match = re.search(r"\buppercase\b.*?\bword\s+(.+)$", text)
    if match:
        return ["uppercase_word", _literal(match.group(1))]
    if "uppercase" in text:
        return ["uppercase_letters"]
    if "lowercase" in text:
        return ["lowercase_letters"]
    if "wrap" in text and ("square bracket" in text or "[" in text):
        return ["wrap", "[]"]
    if "wrap" in text and ("parenthes" in text or "(" in text):
        return ["wrap", "()"]

    match = re.search(r"\breplace\s+(?:every\s+)?(.+?)\s+with\s+(.+)$", text)
    if match:
        before = re.sub(r"^(?:the\s+)?(?:word\s+)?", "", match.group(1))
        after = re.sub(r"^(?:the\s+)?(?:word\s+)?", "", match.group(2))
        aliases = {"hyphen": "-", "an underscore": "_", "underscore": "_",
                   "a slash": "/", "slash": "/"}
        return ["replace", aliases.get(_literal(before), _literal(before)),
                aliases.get(_literal(after), _literal(after))]
    match = re.search(r"\bprefix\s+(.+)$", text)
    if match:
        return ["prefix", _literal(match.group(1))]
    match = re.search(r"\bappend\s+(?:the\s+sentence\s+)?(.+)$", text)
    if match:
        suffix = re.sub(r"\s+to\s+(?:the\s+)?(?:result|text)$", "", match.group(1))
        return ["append", _literal(suffix)]
    return ["unparsed", text]


def semantic_plan_score(expected_tasks, actual_tasks):
    expected = [task_signature(task) for task in expected_tasks]
    actual = ([task_signature(task) for task in actual_tasks]
              if isinstance(actual_tasks, list) else [])
    checks = []
    for index, expected_signature in enumerate(expected):
        actual_signature = actual[index] if index < len(actual) else None
        checks.append({"cycle": index + 1, "pass": actual_signature == expected_signature,
                       "expected_signature": expected_signature,
                       "actual_signature": actual_signature})
    exact = len(actual) == len(expected) and all(check["pass"] for check in checks)
    return {"metric": "exploratory_v3_operation_signature",
            "post_hoc": True, "task_count_ok": len(actual) == len(expected),
            "correct_tasks": sum(check["pass"] for check in checks),
            "total_tasks": len(expected), "exact": exact, "checks": checks}
