"""Generate the frozen BabyAGI execution-gradient v3 benchmark.

The generator is deterministic and contains no model calls.  It emits two
independent template families in each of four categories at chain lengths
2, 3, 5, and 8.  Every item includes gold single-step records so execution
budget can be measured without conflating it with task extraction or state
carry.  The generated JSON and checksum are committed together; do not edit
either artifact by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qfs.benchmark import CATEGORIES, validate_suite


LENGTHS = (2, 3, 5, 8)
FROZEN_ON = "2026-11-03"


def _clauses(tasks):
    labels = []
    for index in range(len(tasks)):
        if index == 0:
            labels.append("First")
        elif index == len(tasks) - 1:
            labels.append("finally")
        elif index == 1:
            labels.append("next")
        else:
            labels.append("then")
    return "; ".join(f"{label} {task}" for label, task in zip(labels, tasks)) + "."


def _gold_steps(initial, tasks, operations):
    current = initial
    out = []
    for cycle, (task, operation) in enumerate(zip(tasks, operations), 1):
        expected = operation(current)
        out.append({"cycle": cycle, "incoming": current,
                    "task": task, "expected": expected})
        current = expected
    return out


def _item(item_id, category, initial, tasks, operations, objective, family,
          normalization):
    steps = _gold_steps(initial, tasks, operations)
    return {
        "id": item_id,
        "category": category,
        "cycles": len(tasks),
        "objective": objective,
        "rubric": {
            "type": "exact_state_chain",
            "normalization": normalization,
            "expected_results": [step["expected"] for step in steps],
        },
        "provenance": {
            "generator": "qfs.generate_benchmark_v3",
            "generator_version": 1,
            "template_family": family,
            "chain_length": len(tasks),
            "normalization": normalization,
            "gold_steps": steps,
        },
    }


def _arithmetic(length, variant):
    initial = length + 4 + variant
    if variant == 0:
        specs = [
            (f"add {length + 1}", lambda x, n=length + 1: x + n),
            ("multiply the result by 2", lambda x: x * 2),
            ("subtract 3", lambda x: x - 3),
            ("add 5", lambda x: x + 5),
            ("multiply the result by 3", lambda x: x * 3),
            (f"subtract {length}", lambda x, n=length: x - n),
            ("double the result", lambda x: x * 2),
            ("add 9", lambda x: x + 9),
        ]
    else:
        specs = [
            ("multiply the result by 3", lambda x: x * 3),
            (f"add {length + 2}", lambda x, n=length + 2: x + n),
            ("subtract 4", lambda x: x - 4),
            ("double the result", lambda x: x * 2),
            ("add 7", lambda x: x + 7),
            ("multiply the result by 2", lambda x: x * 2),
            (f"subtract {length + 1}", lambda x, n=length + 1: x - n),
            ("add 11", lambda x: x + 11),
        ]
    specs = specs[:length]
    tasks, operations = map(list, zip(*specs))
    objective = (
        f"Start with the number {initial}. {_clauses(tasks)} Perform exactly one operation "
        "per task in order and return only the complete current number after each task."
    )
    family = "arithmetic-add-first" if variant == 0 else "arithmetic-multiply-first"
    return _item(f"v3-arithmetic-{family.split('-')[-2]}-{length}",
                 "sequential_arithmetic", initial, tasks, operations, objective,
                 family, "number")


def _literal(length, variant):
    bases = {
        0: {2: "quartz", 3: "willow", 5: "cobalt", 8: "juniper"},
        1: {2: "topaz", 3: "cedar", 5: "saffron", 8: "mariner"},
    }
    initial = bases[variant][length]
    if variant == 0:
        suffix, prefix = f"-n{length}", f"R{length}:"
        specs = [
            (f"append {suffix}", lambda x, s=suffix: x + s),
            ("uppercase only its letters", lambda x: x.upper()),
            (f"prefix {prefix}", lambda x, p=prefix: p + x),
            ("append ?", lambda x: x + "?"),
            ("replace every hyphen with an underscore", lambda x: x.replace("-", "_")),
            ("wrap the complete text in square brackets", lambda x: f"[{x}]"),
            ("lowercase only its letters", lambda x: x.lower()),
            ("append !", lambda x: x + "!"),
        ]
    else:
        suffix, prefix = f"-a{length}", f"K{length}="
        specs = [
            (f"append {suffix}", lambda x, s=suffix: x + s),
            ("wrap the complete text in parentheses", lambda x: f"({x})"),
            ("uppercase only its letters", lambda x: x.upper()),
            (f"prefix {prefix}", lambda x, p=prefix: p + x),
            ("append #", lambda x: x + "#"),
            ("replace every hyphen with a slash", lambda x: x.replace("-", "/")),
            ("lowercase only its letters", lambda x: x.lower()),
            ("append a period", lambda x: x + "."),
        ]
    specs = specs[:length]
    tasks, operations = map(list, zip(*specs))
    objective = (
        f"Begin with the exact text {initial}. {_clauses(tasks)} Apply exactly one text "
        "transformation per task in order and return the complete current text each time."
    )
    family = "literal-route" if variant == 0 else "literal-wrapper"
    return _item(f"v3-literal-{family.split('-')[-1]}-{length}", "ordered_literal",
                 initial, tasks, operations, objective, family, "text")


def _dedupe(values):
    return list(dict.fromkeys(values))


def _structured(length, variant):
    initial = ([length + 3, 2, length + 3, 7, 1, 4] if variant == 0 else
               [9, length, 3, length, 8, 2])
    parity = "even" if variant == 0 else "odd"
    appended = length + 10 + variant
    specs = [
        ("sort it in ascending numeric order", lambda x: sorted(x)),
        ("remove duplicates while preserving order", _dedupe),
        (f"keep only {parity} numbers",
         (lambda x: [n for n in x if n % 2 == 0]) if variant == 0 else
         (lambda x: [n for n in x if n % 2 == 1])),
        ("reverse the array", lambda x: list(reversed(x))),
        (f"append the number {appended}", lambda x, n=appended: [*x, n]),
        ("add 1 to every number", lambda x: [n + 1 for n in x]),
        ("sort it in descending numeric order", lambda x: sorted(x, reverse=True)),
        ("keep only its first three values", lambda x: x[:3]),
    ][:length]
    tasks, operations = map(list, zip(*specs))
    objective = (
        f"Begin with the JSON array {json.dumps(initial, separators=(',', ':'))}. "
        f"{_clauses(tasks)} Do one operation per task and return the complete valid JSON array "
        "after every operation."
    )
    family = "structured-even" if variant == 0 else "structured-odd"
    return _item(f"v3-structured-{parity}-{length}", "structured_operation",
                 initial, tasks, operations, objective, family, "json")


def _replace_once(old, new):
    return lambda text: text.replace(old, new, 1)


def _language(length, variant):
    if variant == 0:
        initial = "quiet lanterns guide boats"
        specs = [
            ("capitalize its first character", lambda x: x[0].upper() + x[1:]),
            ("replace Quiet with Gentle", _replace_once("Quiet", "Gentle")),
            ("append a period", lambda x: x + "."),
            ("prefix Field note: ", lambda x: "Field note: " + x),
            ("replace boats with travelers", _replace_once("boats", "travelers")),
            ("uppercase only the word lanterns", _replace_once("lanterns", "LANTERNS")),
            ("append the sentence Stars appear.", lambda x: x + " Stars appear."),
            ("replace guide with welcome", _replace_once("guide", "welcome")),
        ]
        family = "language-lantern"
    else:
        initial = "calm rivers carry stories"
        specs = [
            ("capitalize its first character", lambda x: x[0].upper() + x[1:]),
            ("replace Calm with Silver", _replace_once("Calm", "Silver")),
            ("append a period", lambda x: x + "."),
            ("prefix Log entry: ", lambda x: "Log entry: " + x),
            ("replace stories with messages", _replace_once("stories", "messages")),
            ("uppercase only the word rivers", _replace_once("rivers", "RIVERS")),
            ("append the sentence Dawn follows.", lambda x: x + " Dawn follows."),
            ("replace carry with preserve", _replace_once("carry", "preserve")),
        ]
        family = "language-river"
    specs = specs[:length]
    tasks, operations = map(list, zip(*specs))
    objective = (
        f"Begin with the exact sentence `{initial}`. {_clauses(tasks)} This is constrained "
        "language editing: perform only the named edit and return the complete current text "
        "after every task, preserving all other wording and punctuation exactly."
    )
    return _item(f"v3-{family}-{length}", "checkable_language", initial, tasks,
                 operations, objective, family, "text")


def generate_suite():
    items = []
    for length in LENGTHS:
        for variant in range(2):
            items.extend((_literal(length, variant), _arithmetic(length, variant),
                          _structured(length, variant), _language(length, variant)))
    counts = {category: sum(item["category"] == category for item in items)
              for category in CATEGORIES}
    suite = {
        "schema_version": 1,
        "benchmark_id": "babyagi-execution-gradient-v3",
        "status": "frozen",
        "frozen_on": FROZEN_ON,
        "description": (
            "Checksum-frozen execution-gradient suite generated before evaluation. It contains "
            "two unseen deterministic template families per category at chain lengths 2, 3, 5, "
            "and 8. Gold single-step records isolate execution from ledger extraction; this suite "
            "does not test open-ended planning. Do not tune prompts after observing results."
        ),
        "objective_count": len(items),
        "category_counts": counts,
        "leak_sentinels": ["quartz", "mariner", "Field note:", "Dawn follows."],
        "items": items,
    }
    return validate_suite(suite)


def write_suite(output):
    output = Path(output)
    suite = generate_suite()
    text = json.dumps(suite, indent=2, ensure_ascii=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    output.with_suffix(".sha256").write_text(f"{digest}  {output.name}\n")
    return digest, suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/babyagi-v3.json"))
    args = parser.parse_args()
    digest, suite = write_suite(args.output)
    print(json.dumps({"output": str(args.output), "sha256": digest,
                      "items": len(suite["items"]),
                      "category_counts": suite["category_counts"]}))


if __name__ == "__main__":
    main()
