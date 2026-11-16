"""E15's frozen v4-horizon suite: long chains, a rule applied until a condition, worklists.

    uv run python -m harness.suite_v4            # write both splits and SHA256SUMS
    uv run python -m harness.suite_v4 --check    # regenerate in memory, byte-compare, re-verify
    uv run python -m harness.suite_v4 --tokens   # Qwen3.5-4B token report (tokenizer, CPU only)

The generator is seeded and deterministic. It makes no model calls and needs no tokenizer, so
`--check` runs anywhere; the token limits are checked by harness/tests/test_suite_v4.py
against models/Qwen3.5-4B. Items follow babyagi-v3 (qfs.generate_benchmark_v3): `cycles` is
the gold length L, the rubric is `exact_state_chain` (one expected state per step, the
starting state excluded), and `provenance.gold_steps` holds {cycle, incoming, task, expected}.

Three horizon categories (design: experiments/drafts/E15-unbounded-design.md, section 1):

- A `long_chain`: every task listed, v3 style ("First ...; next ...; then ...; finally ...").
- B `rule_until`: one rule, applied until a condition on the current state holds.
- C `worklist`: a job table; processing a job adds its follow-up jobs to a queue under a
  stated order rule (FIFO, LIFO, or alphabetical). The state after each step is the JSON
  object {"done": [...], "queue": [...]}, so the stop (an empty queue) and the next job are
  both readable from the state alone.

Two v3 constraints do not fit, and are handled here without touching vendor/:

1. `qfs.benchmark.validate_suite` (and so `load_suite`) requires 1 <= cycles <= 8. Use
   `load()` / `validate()` below: they run the upstream validator unchanged on a view whose
   cycles are capped at 8, then check 1 <= cycles <= 32 and one expected state per cycle.
   `qfs.benchmark.score_item` has no length cap and is used unchanged.
2. `category` must be one of v3's four operation domains, so it keeps that meaning; the
   horizon category is `provenance.horizon_category` (and the id prefix v4-a/-b/-c).

Gold results come from the generator's own construction and, independently, from
`reference_results`, which parses only the objective text and re-executes it. Generation
fails unless the two agree.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from harness.upstream import ROOT, module

SUITE_DIR = ROOT / "experiments" / "E15-adaptive-screen" / "suite"
SUMS = "SHA256SUMS"
TOKENIZER = ROOT / "models" / "Qwen3.5-4B"
FROZEN_ON = "2026-11-07"
GENERATOR_VERSION = 1
SPLITS = {
    "test": {"file": "babyagi-v4-horizon.json", "benchmark_id": "babyagi-v4-horizon",
             "seed": 1548, "prefix": "v4"},
    "dev": {"file": "babyagi-v4-horizon-dev.json", "benchmark_id": "babyagi-v4-horizon-dev",
            "seed": 1516, "prefix": "v4dev"},
}
MAX_CYCLES = 32            # gold length bound (the loop's compiled cap would be 40)
UPSTREAM_MAX_CYCLES = 8    # qfs.benchmark.validate_suite's bound
STATE_TOKENS, LEDGER_TOKENS, STEP_TOKENS = 300, 450, 256
STATE_BYTES_GUARD = 450    # tokenizer-free pre-filter; the token test is the real check
ATTEMPTS = 20000

HORIZON = {"A": "long_chain", "B": "rule_until", "C": "worklist"}
A_FAMILIES = ("arithmetic", "structured", "literal", "language")
# family -> (v3 category, normalization)
DOMAIN = {
    "arithmetic": ("sequential_arithmetic", "number"),
    "structured": ("structured_operation", "json"),
    "literal": ("ordered_literal", "text"),
    "language": ("checkable_language", "text"),
    "digit-sum": ("sequential_arithmetic", "number"),
    "add-reverse": ("sequential_arithmetic", "number"),
    "collatz": ("sequential_arithmetic", "number"),
    "subtract-largest-digit": ("sequential_arithmetic", "number"),
    "caesar-shift": ("ordered_literal", "text"),
    "rotate-words": ("ordered_literal", "text"),
    "uppercase-first-lowercase": ("ordered_literal", "text"),
    "bubble-swap": ("structured_operation", "json"),
    "rotate-increment": ("structured_operation", "json"),
    "decrement-largest": ("structured_operation", "json"),
    "fifo": ("structured_operation", "json"),
    "lifo": ("structured_operation", "json"),
    "alphabetical": ("structured_operation", "json"),
}

PLANS = {
    "test": (
        [("A", family, n) for n in (8, 16, 24, 32) for family in A_FAMILIES]
        + [("B", family, n) for family, n in sorted((
            ("digit-sum", 8), ("add-reverse", 12), ("collatz", 13), ("collatz", 20),
            ("subtract-largest-digit", 22), ("digit-sum", 26),
            ("rotate-words", 5), ("uppercase-first-lowercase", 7), ("caesar-shift", 10),
            ("uppercase-first-lowercase", 16), ("caesar-shift", 24),
            ("rotate-increment", 4), ("decrement-largest", 6), ("bubble-swap", 14),
            ("rotate-increment", 18), ("bubble-swap", 28)), key=lambda row: row[1])]
        + [("C", order, n) for order, n in sorted((
            ("fifo", 4), ("fifo", 7), ("fifo", 10), ("fifo", 13), ("fifo", 16), ("fifo", 20),
            ("alphabetical", 5), ("alphabetical", 8), ("alphabetical", 11),
            ("alphabetical", 14), ("alphabetical", 17),
            ("lifo", 6), ("lifo", 9), ("lifo", 12), ("lifo", 15), ("lifo", 18)),
            key=lambda row: row[1])]
    ),
    "dev": (
        [("A", "language", 8), ("A", "arithmetic", 16), ("A", "literal", 24),
         ("A", "structured", 32)]
        + [("B", family, n) for family, n in (
            ("rotate-words", 5), ("collatz", 9), ("bubble-swap", 11), ("caesar-shift", 15),
            ("digit-sum", 19), ("decrement-largest", 25))]
        + [("C", order, n) for order, n in (
            ("fifo", 4), ("alphabetical", 6), ("lifo", 9), ("fifo", 12),
            ("alphabetical", 15), ("lifo", 19))]
    ),
}

_LOWER = "abcdefghijklmnopqrstuvwxyz"
_UPPER = _LOWER.upper()
LITERAL_WORDS = ("ember", "harbor", "falcon", "meadow", "pixel", "granite", "velvet", "orchid",
                 "lumen", "tundra", "beacon", "nectar", "quill", "sorrel", "marlin", "basalt")
ADJECTIVES = ("quiet", "bright", "calm", "gentle", "silver", "rapid", "warm", "ancient",
              "small", "brave", "golden", "pale", "soft", "bold", "early", "hidden")
NOUNS = ("lanterns", "rivers", "boats", "stories", "birds", "clouds", "roads", "stones",
         "bells", "gardens", "ships", "letters", "maps", "wolves", "trees", "doors", "fields",
         "towers")
VERBS = ("guide", "carry", "follow", "greet", "hold", "shape", "watch", "cross", "guard",
         "find", "keep", "mark")
ADVERBS = ("slowly", "often", "north", "tonight", "again", "softly", "everywhere", "still")
LANGUAGE_POOL = ADJECTIVES + NOUNS + VERBS + ADVERBS
CAESAR_WORDS = ("cold", "frost", "mint", "lamp", "river", "stone", "cloud", "brisk", "quilt",
                "grape", "delta", "plume", "crisp", "dusk", "fable", "glow")
ROTATE_WORDS = ("red", "blue", "green", "gold", "gray", "pink", "teal", "navy", "plum", "rust",
                "sand", "jade", "lime", "mint", "ruby", "onyx", "snow", "ash", "coal", "rose",
                "sage", "wine", "fern")
JOBS = ("acorn", "badger", "cactus", "dahlia", "ember", "falcon", "garnet", "hazel", "iris",
        "jasper", "kiwi", "lemon", "maple", "nutmeg", "olive", "pepper", "quince", "raven",
        "sorrel", "thistle", "umber", "violet", "walnut", "yarrow", "zinnia", "almond",
        "bramble", "clover", "dove", "elm", "fennel", "ginger", "heather", "indigo", "kestrel",
        "lilac", "mango", "nettle", "otter", "pine", "robin", "sage", "tulip", "aster", "birch",
        "coral", "dune", "fern", "gull", "holly", "ivy", "larch", "moss", "oak", "poppy",
        "reed", "spruce", "thyme")


def _no_substrings(words):
    return all(a == b or a not in b for a in words for b in words)


assert _no_substrings(JOBS) and _no_substrings(ROTATE_WORDS)


class GenerationError(RuntimeError):
    """A plan entry could not be generated within its constraints."""


def _rng(seed, key):
    digest = hashlib.sha256(f"babyagi-v4-horizon/{seed}/{key}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _clauses(tasks):
    """v3's clause labels: First; next; then ...; finally."""
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


# ------------------------------------------------------------------ length-leak rule

_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
          "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = {20: "twenty", 30: "thirty", 40: "forty"}
_ORDINAL_UNITS = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth",
                  12: "twelfth"}


def _cardinal(n):
    if n < 20:
        return _UNITS[n]
    tens, unit = n // 10 * 10, n % 10
    return _TENS[tens] + ("" if unit == 0 else "-" + _UNITS[unit])


def _ordinal(n):
    if n in _ORDINAL_UNITS:
        return _ORDINAL_UNITS[n]
    if n < 20:
        return _UNITS[n] + "th"
    tens, unit = n // 10 * 10, n % 10
    if unit == 0:
        return _TENS[tens][:-1] + "ieth"
    return _TENS[tens] + "-" + _ordinal(unit)


def length_leaks(text, length):
    """Every way `text` could state `length`, under the suite's leak rule.

    A leak is a maximal run of digits whose value equals L, or L written as an English
    cardinal or ordinal word ("twelve", "twelfth", "twenty-four"), matched as a whole word
    case-insensitively. Numbers that merely contain L's digits (140 when L = 14) do not count.
    """
    hits = [m.group() for m in re.finditer(r"\d+", text) if int(m.group()) == length]
    lowered = text.lower()
    for word in (_cardinal(length), _ordinal(length)):
        if re.search(rf"(?<![a-z-]){re.escape(word)}(?![a-z-])", lowered):
            hits.append(word)
    return hits


# ------------------------------------------------------------------ A: long chains


def _walk(rng, initial, length, candidates):
    """A random chain in which every step changes the state, never returns to the state two
    steps back, and never repeats the previous task's wording."""
    state, recent, tasks, states = initial, [initial], [], []
    for _ in range(length):
        options = [(task, new) for task, new in candidates(state, rng)
                   if new != state and new != recent[-2:][0]
                   and (not tasks or task != tasks[-1])]
        if not options:
            return None
        task, state = rng.choice(options)
        tasks.append(task)
        states.append(state)
        recent.append(state)
    return tasks, states


def _arith_candidates(x, rng):
    out = []
    k = rng.randint(2, 49)
    if x + k <= 999:
        out.append((f"add {k}", x + k))
    k = rng.randint(2, 49)
    if x - k >= 1:
        out.append((f"subtract {k}", x - k))
    if x * 2 <= 999:
        out.append((rng.choice(("double the result", "multiply the result by 2")), x * 2))
    if x * 3 <= 999:
        out.append(("multiply the result by 3", x * 3))
    if x % 2 == 0 and x >= 4:
        out.append(("divide the result by 2", x // 2))
    if x % 3 == 0 and x >= 6:
        out.append(("divide the result by 3", x // 3))
    return out


def _a_arithmetic(length, rng):
    initial = rng.randint(11, 60)
    walk = _walk(rng, initial, length, _arith_candidates)
    if walk is None:
        return None
    tasks, states = walk
    objective = (f"Start with the number {initial}. {_clauses(tasks)} Perform exactly one "
                 "operation per task in order and return only the complete current number "
                 "after each task.")
    return initial, tasks, states, objective


def _struct_candidates(x, rng):
    out = []
    ascending, descending = sorted(x), sorted(x, reverse=True)
    if x != ascending:
        out.append(("sort the array in ascending numeric order", ascending))
    if x != descending:
        out.append(("sort the array in descending numeric order", descending))
    if x != x[::-1]:
        out.append(("reverse the array", x[::-1]))
    unique = []
    for n in x:
        if n not in unique:
            unique.append(n)
    if unique != x:
        out.append(("remove duplicates while preserving order", unique))
    for parity, keep in (("even", 0), ("odd", 1)):
        kept = [n for n in x if n % 2 == keep]
        if kept != x and len(kept) >= 2:
            out.append((f"keep only the {parity} numbers", kept))
    if len(x) < 8:
        grow = 2 if len(x) <= 3 else 1
        for _ in range(grow):
            n = rng.randint(1, 60)
            out.append((f"append the number {n} to the end", x + [n]))
            n = rng.randint(1, 60)
            out.append((f"insert the number {n} at the start", [n] + x))
    if len(x) > 2:
        out.append(("remove the first number", x[1:]))
        out.append(("remove the last number", x[:-1]))
    if len(x) > 3:
        out.append(("keep only the first three numbers", x[:3]))
    k = rng.randint(1, 9)
    if max(x) + k <= 99:
        out.append((f"add {k} to every number", [n + k for n in x]))
    k = rng.randint(1, 9)
    if min(x) - k >= 0:
        out.append((f"subtract {k} from every number", [n - k for n in x]))
    if max(x) * 2 <= 99:
        out.append(("double every number", [n * 2 for n in x]))
    values = sorted(set(x))
    if len(values) > 1:
        v = rng.choice(values)
        rest = [n for n in x if n != v]
        if len(rest) >= 2:
            out.append((f"remove every occurrence of the number {v}", rest))
    return out


def _a_structured(length, rng):
    initial = [rng.randint(1, 40) for _ in range(5)]
    initial.insert(rng.randint(0, 5), initial[rng.randrange(5)])  # one duplicate
    walk = _walk(rng, initial, length, _struct_candidates)
    if walk is None:
        return None
    tasks, states = walk
    objective = (f"Begin with the JSON array {_compact(initial)}. {_clauses(tasks)} Do one "
                 "operation per task and return the complete valid JSON array after every "
                 "operation.")
    return initial, tasks, states, objective


def _suffix(rng):
    l1, l2, d = rng.choice(_LOWER), rng.choice(_LOWER), str(rng.randint(1, 9))
    return rng.choice((f"_{l1}{d}", f"{l1}-{d}", f".{l1}{l2}", "#", "!", "?", f"x{d}",
                       f"%{d}"))


def _prefix(rng):
    u, l1, l2, d = (rng.choice(_UPPER), rng.choice(_LOWER), rng.choice(_LOWER),
                    str(rng.randint(1, 9)))
    return rng.choice((f"{u}{d}:", f"{l1}{l2}=", f"{l1}-", f"{d}{l1}_", "@", "$"))


def _literal_candidates(x, rng):
    out = []
    s = _suffix(rng)
    if len(x) + len(s) <= 28:
        out.append((f"append `{s}` to the end", x + s))
    p = _prefix(rng)
    if len(x) + len(p) <= 28:
        out.append((f"add `{p}` to the front", p + x))
    if any(c.islower() for c in x):
        out.append(("uppercase every letter", x.upper()))
    if any(c.isupper() for c in x):
        out.append(("lowercase every letter", x.lower()))
    if "-" in x:
        out.append(("replace every hyphen with an underscore", x.replace("-", "_")))
    if "_" in x:
        out.append(("replace every underscore with a hyphen", x.replace("_", "-")))
    if not set("[]") & set(x) and len(x) <= 26:
        out.append(("wrap the whole text in square brackets", "[" + x + "]"))
    if not set("()") & set(x) and len(x) <= 26:
        out.append(("wrap the whole text in parentheses", "(" + x + ")"))
    if len(x) > 4 and x[0] == "[" and x[-1] == "]":
        out.append(("remove the outer square brackets", x[1:-1]))
    if len(x) > 4 and x[0] == "(" and x[-1] == ")":
        out.append(("remove the outer parentheses", x[1:-1]))
    if len(x) > 4:
        out.append(("delete the first character", x[1:]))
        out.append(("delete the last character", x[:-1]))
    stripped = "".join(c for c in x if not c.isdigit())
    if stripped != x and len(stripped) >= 3:
        out.append(("delete every digit", stripped))
    letters = sorted({c for c in x if c.isalpha()})
    if letters:
        a = rng.choice(letters)
        pool = [c for c in (_LOWER if a.islower() else _UPPER) if c not in x]
        if a.swapcase() not in x and pool:
            b = rng.choice(pool)
            case = "lowercase" if a.islower() else "uppercase"
            out.append((f"replace every {case} `{a}` with `{b}`", x.replace(a, b)))
    return out


def _a_literal(length, rng):
    initial = rng.choice(LITERAL_WORDS)
    walk = _walk(rng, initial, length, _literal_candidates)
    if walk is None:
        return None
    tasks, states = walk
    objective = (f"Begin with the exact text `{initial}`. Text inside backticks is literal, "
                 f"and the backticks are not part of it. {_clauses(tasks)} Apply exactly one "
                 "text transformation per task in order and return the complete current text "
                 "each time.")
    return initial, tasks, states, objective


def _fits(word, lowered):
    return all(word != o and word not in o and o not in word for o in lowered)


def _case_like(word, model):
    if len(model) > 1 and model.isupper():
        return word.upper()
    if model[0].isupper():
        return word[0].upper() + word[1:]
    return word


def _language_candidates(text, rng):
    punct = text[-1] if text[-1] in ".!" else ""
    words = (text[:-1] if punct else text).split(" ")

    def render(ws, p=punct):
        return " ".join(ws) + p

    out = []
    lowered = [w.lower() for w in words]
    fresh = [w for w in LANGUAGE_POOL if _fits(w, lowered)]
    if words[0][0].islower():
        out.append(("capitalize its first character",
                    render([words[0][0].upper() + words[0][1:]] + words[1:])))
    for _ in range(2 if fresh else 0):  # word replacement carries double weight, as in v3
        i = rng.randrange(len(words))
        b = _case_like(rng.choice(fresh), words[i])
        out.append((f"replace the word {words[i]} with {b}",
                    render(words[:i] + [b] + words[i + 1:])))
    if len(words) > 3:
        i = rng.randrange(len(words))
        out.append((f"delete the word {words[i]}", render(words[:i] + words[i + 1:])))
    if len(words) < 9 and fresh:
        i, b = rng.randrange(len(words)), rng.choice(fresh)
        out.append((f"insert the word {b} after the word {words[i]}",
                    render(words[:i + 1] + [b] + words[i + 1:])))
        i, b = rng.randrange(1, len(words)), rng.choice(fresh)
        out.append((f"insert the word {b} before the word {words[i]}",
                    render(words[:i] + [b] + words[i:])))
    plain = [i for i, w in enumerate(words) if w.islower()]
    if plain:
        i = rng.choice(plain)
        out.append((f"uppercase only the word {words[i]}",
                    render(words[:i] + [words[i].upper()] + words[i + 1:])))
    cased = [i for i, w in enumerate(words) if not w.islower()]
    if cased:
        i = rng.choice(cased)
        out.append((f"lowercase the word {words[i]}",
                    render(words[:i] + [words[i].lower()] + words[i + 1:])))
    if len(words) >= 3:
        i, j = sorted((rng.randrange(len(words)), rng.randrange(len(words))))
        if i != j:
            swapped = list(words)
            swapped[i], swapped[j] = words[j], words[i]
            out.append((f"swap the words {words[i]} and {words[j]}", render(swapped)))
    if punct == "":
        out.append(("append a period", render(words, ".")))
    elif punct == ".":
        out.append(("replace the final period with an exclamation mark", render(words, "!")))
    else:
        out.append(("replace the final exclamation mark with a period", render(words, ".")))
    if punct:
        out.append(("remove the final punctuation mark", render(words, "")))
    return out


def _a_language(length, rng):
    words = []
    pattern = (ADJECTIVES, NOUNS, VERBS) + ((ADJECTIVES, NOUNS) if rng.randrange(2)
                                           else (NOUNS,))
    for pool in pattern:
        choices = [w for w in pool if _fits(w, words)]
        words.append(rng.choice(choices))
    initial = " ".join(words)
    walk = _walk(rng, initial, length, _language_candidates)
    if walk is None:
        return None
    tasks, states = walk
    objective = (f"Begin with the exact sentence `{initial}`. {_clauses(tasks)} This is "
                 "constrained language editing: perform only the named edit and return the "
                 "complete current text after every task, preserving all other wording and "
                 "punctuation exactly.")
    return initial, tasks, states, objective


A_BUILDERS = {"arithmetic": _a_arithmetic, "structured": _a_structured,
              "literal": _a_literal, "language": _a_language}


def _build_a(family, length, rng):
    built = A_BUILDERS[family](length, rng)
    if built is None:
        return None
    initial, tasks, states, objective = built
    return {"objective": objective, "initial": initial, "step_tasks": tasks, "states": states,
            "ledger": {"initial": initial, "tasks": tasks}, "extra": {}}


# ------------------------------------------------------------------ B: rule until condition

B_RULES = {
    "digit-sum": "add the sum of the number's digits to the number",
    "add-reverse": "add {a}, then reverse the order of the number's digits",
    "collatz": ("if the number is even, divide it by 2; if it is odd, multiply it by 3 "
                "and add 1"),
    "subtract-largest-digit": "subtract the number's largest digit from the number",
    "caesar-shift": ("replace every letter with the next letter of the alphabet, with z "
                     "becoming a"),
    "rotate-words": "move the first word to the end",
    "uppercase-first-lowercase": ("change the first lowercase letter in the text (reading "
                                  "from the left) to uppercase"),
    "bubble-swap": ("swap the first pair of neighbouring numbers in which the left number is "
                    "larger than the right number"),
    "rotate-increment": "remove the first number, add 1 to it, and append it to the end",
    "decrement-largest": ("subtract 1 from the largest number (if several numbers are tied "
                          "for largest, use the leftmost one)"),
}
_B_START = {"number": "Start with the number {x}.",
            "text": "Begin with the exact text `{x}`.",
            "array": "Begin with the JSON array {x}."}
_B_RETURN = {
    "number": "Return only the complete current number after each application, not the "
              "starting number.",
    "text": "Return the complete current text after each application, not the starting text.",
    "array": "Return the complete valid JSON array after each application, not the starting "
             "array.",
}


def _digits(x):
    out = []
    while x:
        x, d = divmod(x, 10)
        out.append(d)
    return out


def _reverse_number(x):
    out = 0
    while x:
        x, d = divmod(x, 10)
        out = out * 10 + d
    return out


def _collatz_path(n, limit=40):
    path = [n]
    while path[-1] != 1 and len(path) <= limit:
        x = path[-1]
        path.append(x // 2 if x % 2 == 0 else 3 * x + 1)
    return path


_COLLATZ = {n: _collatz_path(n) for n in range(2, 1000)}


def _iterate(x, step, length):
    states = []
    for _ in range(length):
        x = step(x)
        states.append(x)
    return states


def _b_candidate(family, n, rng):
    """(initial, rule, condition, states) or None; states[-1] is the first to satisfy it."""
    rule = B_RULES[family]
    if family == "digit-sum":
        x = rng.randint(5, 150)
        states = _iterate(x, lambda v: v + sum(_digits(v)), n)
        prev = ([x] + states)[-2]
        t = rng.randint(prev, states[-1] - 1)
        return x, rule, f"the number is greater than {t}", states
    if family == "add-reverse":
        x, a = rng.randint(10, 99), rng.randint(2, 29)
        states, v = [], x
        for _ in range(n):
            if (v + a) % 10 == 0:
                return None
            v = _reverse_number(v + a)
            states.append(v)
        record = max([x] + states[:-1])
        if states[-1] <= record or max(states) > 9999:
            return None
        t = rng.randint(record, states[-1] - 1)
        return x, rule.format(a=a), f"the number is greater than {t}", states
    if family == "collatz":
        starts = [s for s, path in _COLLATZ.items()
                  if len(path) == n + 1 and path[-1] == 1 and max(path) <= 9999]
        if not starts:
            return None
        x = rng.choice(starts)
        return x, rule, "the number equals 1", _COLLATZ[x][1:]
    if family == "subtract-largest-digit":
        x = rng.randint(150, 999)
        states = _iterate(x, lambda v: v - max(_digits(v)), n)
        prev = ([x] + states)[-2]
        t = rng.randint(states[-1] + 1, prev)
        return x, rule, f"the number is less than {t}", states
    if family == "caesar-shift":
        if n > 25:
            return None
        word = rng.choice(CAESAR_WORDS)
        shift = lambda w: "".join(_LOWER[(_LOWER.index(c) + 1) % 26] for c in w)  # noqa: E731
        target = _LOWER[(_LOWER.index(word[0]) + n) % 26]
        return word, rule, f"the text begins with the letter {target}", _iterate(word, shift, n)
    if family == "rotate-words":
        pool = list(ROTATE_WORDS)
        rng.shuffle(pool)
        words = pool[:n + rng.randint(1, 4)]
        states = _iterate(words, lambda ws: ws[1:] + ws[:1], n)
        return (" ".join(words), rule, f"the first word is {words[n]}",
                [" ".join(ws) for ws in states])
    if family == "uppercase-first-lowercase":
        words, letters = [], 0
        while letters < n + 2 + rng.randint(0, 4):
            choices = [w for w in LANGUAGE_POOL if _fits(w, words)]
            words.append(rng.choice(choices))
            letters = sum(len(w) for w in words)
        text = list(" ".join(words))
        slots = [i for i, c in enumerate(text) if c.isalpha()]
        rng.shuffle(slots)
        for i in slots[n:]:
            text[i] = text[i].upper()
        x = "".join(text)

        def upper_first(s):
            i = next(i for i, c in enumerate(s) if c.islower())
            return s[:i] + s[i].upper() + s[i + 1:]
        return x, rule, "the text contains no lowercase letters", _iterate(x, upper_first, n)
    if family == "bubble-swap":
        k_min = next(k for k in range(2, 12) if k * (k - 1) // 2 >= n)
        k = rng.randint(max(k_min, 5), max(k_min, 5) + 2)
        code, room = [0] * k, [k - 1 - i for i in range(k)]
        for _ in range(n):
            open_slots = [i for i in range(k) if code[i] < room[i]]
            code[rng.choice(open_slots)] += 1
        pool = list(range(1, 61))
        rng.shuffle(pool)
        values = sorted(pool[:k])
        x = [values.pop(c) for c in code]  # Lehmer code: exactly n inversions

        def bubble(a):
            a = list(a)
            i = next(i for i in range(len(a) - 1) if a[i] > a[i + 1])
            a[i], a[i + 1] = a[i + 1], a[i]
            return a
        return x, rule, "the array is in ascending order", _iterate(x, bubble, n)
    if family == "rotate-increment":
        x = [rng.randint(0, 20) for _ in range(rng.randint(3, 5))]
        states = _iterate(x, lambda a: a[1:] + [a[0] + 1], n)
        firsts = [a[0] for a in [x] + states]
        record = max(firsts[:-1])
        if firsts[-1] <= record:
            return None
        t = rng.randint(record, firsts[-1] - 1)
        return x, rule, f"the first number is greater than {t}", states
    if family == "decrement-largest":
        m, low = rng.randint(3, 4), rng.randint(1, 9)
        extra = [0] * (m - 1)
        for _ in range(n):
            extra[rng.randrange(m - 1)] += 1
        x = [low + e for e in extra]
        x.insert(rng.randint(0, m - 1), low)

        def lower_largest(a):
            a, top = list(a), max(a)
            a[a.index(top)] -= 1
            return a
        return x, rule, "all numbers in the array are equal", _iterate(x, lower_largest, n)
    raise ValueError(family)


def _build_b(family, length, rng):
    candidate = _b_candidate(family, length, rng)
    if candidate is None:
        return None
    initial, rule, condition, states = candidate
    unit = {"number": "number", "text": "text", "json": "array"}[DOMAIN[family][1]]
    shown = _compact(initial) if unit == "array" else initial
    objective = (f"{_B_START[unit].format(x=shown)} Apply this rule repeatedly: {rule}. After "
                 f"each application, check whether {condition}, and stop as soon as it is "
                 f"true. {_B_RETURN[unit]}")
    return {"objective": objective, "initial": initial, "step_tasks": [rule] * length,
            "states": states, "ledger": {"initial": initial, "rule": rule, "until": condition},
            "extra": {"rule": rule, "stop_condition": condition}}


# ------------------------------------------------------------------ C: worklists

C_STEPS = {
    "fifo": ("remove the job at the front of the queue and append it to the end of the done "
             "list; then add each of that job's follow-up jobs, in the listed order, to the end "
             "of the queue, skipping any job that is already in the done list or already in "
             "the queue"),
    "lifo": ("remove the job at the end of the queue (the most recently added job) and append "
             "it to the end of the done list; then add each of that job's follow-up jobs, in "
             "the listed order, to the end of the queue, skipping any job that is already in "
             "the done list or already in the queue"),
    "alphabetical": ("remove the alphabetically first job from the queue and append it to the "
                     "end of the done list; then add each of that job's follow-up jobs to the "
                     "queue, skipping any job that is already in the done list or already in "
                     "the queue, and keep the queue in alphabetical order"),
}


def _row(job, follow):
    return f"{job} -> {', '.join(follow) if follow else 'nothing'}"


def _build_c(order, length, rng):
    names = list(JOBS)
    rng.shuffle(names)
    distractors = rng.randint(2, 4)
    reach, others = names[:length], names[length:length + distractors]
    seeds = reach[:1 if length < 8 else rng.randint(1, 2)]
    table = {name: [] for name in reach + others}
    for i in range(len(seeds), length):  # a spanning forest: every reachable job has a parent
        parents = [p for p in reach[:i] if len(table[p]) < 3]
        table[rng.choice(parents)].append(reach[i])
    for _ in range(rng.randint(1, max(1, length // 3))):  # repeats that exercise skipping
        a, b = rng.choice(reach), rng.choice(reach)
        if a != b and b not in table[a] and len(table[a]) < 3:
            table[a].append(b)
    for job in others:  # unreachable rows: nothing reachable adds them
        for _ in range(rng.randint(1, 2)):
            b = rng.choice([x for x in reach + others if x != job])
            if b not in table[job]:
                table[job].append(b)
    for job in table:
        rng.shuffle(table[job])
    table = {job: table[job] for job in sorted(table)}
    queue = sorted(seeds) if order == "alphabetical" else list(seeds)
    initial = {"done": [], "queue": list(queue)}
    done, states, tasks = [], [], []
    while queue:
        if order == "fifo":
            job = queue.pop(0)
        elif order == "lifo":
            job = queue.pop()
        else:
            job = min(queue)
            queue.remove(job)
        done.append(job)
        queue.extend(f for f in table[job] if f not in done and f not in queue)
        if order == "alphabetical":
            queue.sort()
        states.append({"done": list(done), "queue": list(queue)})
        tasks.append(f"process {job} ({_row(job, table[job])})")
    if len(states) != length:
        return None
    rows = "; ".join(_row(job, follow) for job, follow in table.items())
    rule = f"Repeat this step until the queue is empty: {C_STEPS[order]}."
    objective = (f"Begin with the JSON state {_compact(initial)}. Job table, where each job "
                 f"lists the follow-up jobs it adds: {rows}. {rule} Return the complete JSON "
                 "state after every step, not the starting state.")
    return {"objective": objective, "initial": initial, "step_tasks": tasks, "states": states,
            "ledger": {"initial": initial, "rule": rule, "table": table},
            "extra": {"order_rule": order, "table_rows": len(table),
                      "unreachable_jobs": sorted(others)}}


BUILDERS = {"A": _build_a, "B": _build_b, "C": _build_c}


# ------------------------------------------------------------------ assembly


def _state_bytes(state):
    return len(json.dumps(state, ensure_ascii=False).encode())


def build_item(split, label, family, length):
    seed, prefix = SPLITS[split]["seed"], SPLITS[split]["prefix"]
    key = f"{label.lower()}-{family}-{length}"
    rng = _rng(seed, key)
    for _ in range(ATTEMPTS):
        built = BUILDERS[label](family, length, rng)
        if built is None:
            continue
        if label != "A" and length_leaks(built["objective"], length):
            continue
        if max(_state_bytes(s) for s in [built["initial"]] + built["states"]) > STATE_BYTES_GUARD:
            continue
        break
    else:
        raise GenerationError(f"{split} {key}: no candidate within {ATTEMPTS} attempts")
    category, normalization = DOMAIN[family]
    steps, incoming = [], built["initial"]
    for cycle, (task, expected) in enumerate(zip(built["step_tasks"], built["states"]), 1):
        steps.append({"cycle": cycle, "incoming": incoming, "task": task, "expected": expected})
        incoming = expected
    item = {
        "id": f"{prefix}-{key}",
        "category": category,
        "cycles": length,
        "objective": built["objective"],
        "rubric": {"type": "exact_state_chain", "normalization": normalization,
                   "expected_results": list(built["states"])},
        "provenance": {
            "generator": "harness.suite_v4",
            "generator_version": GENERATOR_VERSION,
            "split": split,
            "seed": seed,
            "horizon_category": HORIZON[label],
            "length_stated": label == "A",
            "template_family": family,
            "chain_length": length,
            "normalization": normalization,
            **built["extra"],
            "reference_ledger": built["ledger"],
            "gold_steps": steps,
        },
    }
    reference = reference_results(item["objective"])
    if reference != item["rubric"]["expected_results"]:
        raise GenerationError(f"{item['id']}: reference disagrees with the generator")
    return item


_DESCRIPTIONS = {
    "test": (
        "Checksum-frozen held-out suite for E15, the host-simulation screen of an adaptive-length "
        "loop on Qwen3.5-4B, generated by harness.suite_v4 (seed {seed}) before any decode. 48 "
        "items in three horizon categories, named in provenance.horizon_category: long_chain (16; "
        "every task listed; L 8, 16, 24, 32 over v3's four operation families), rule_until (16; "
        "one rule applied until a stated condition on the current state; L 4-28, not stated) and "
        "worklist (16; a job table whose processing adds follow-up jobs under a FIFO, LIFO or "
        "alphabetical rule; state {{\"done\",\"queue\"}}; L 4-20, not stated). `category` keeps "
        "v3's operation domain. Results exclude the starting state. cycles exceeds 8, so load "
        "with harness.suite_v4.load, not qfs.benchmark.load_suite. Never tune prompts on this "
        "split; use babyagi-v4-horizon-dev."),
    "dev": (
        "Development split of babyagi-v4-horizon for E15, generated by harness.suite_v4 (seed "
        "{seed}) with the same templates: 4 long_chain, 6 rule_until and 6 worklist items. Use "
        "it only for prompt-wording checks; it is never evidence. cycles exceeds 8, so load "
        "with harness.suite_v4.load, not qfs.benchmark.load_suite."),
}


def _sentinels(items):
    by_id = {item["id"].split("-", 1)[1]: item for item in items}
    picks = []
    for key in ("a-literal-32", "a-language-32", "a-language-8", "b-rotate-words-5"):
        if key in by_id:
            picks.append(str(by_id[key]["provenance"]["gold_steps"][0]["incoming"]))
    for item in items:
        if item["provenance"]["horizon_category"] == "worklist":
            ledger = item["provenance"]["reference_ledger"]
            job, follow = next((j, f) for j, f in ledger["table"].items() if f)
            picks.append(_row(job, follow))
            break
    return picks


def generate_suite(split):
    items = [build_item(split, *row) for row in PLANS[split]]
    bench = module("qfs.benchmark")
    counts = {c: sum(item["category"] == c for item in items) for c in bench.CATEGORIES}
    suite = {
        "schema_version": 1,
        "benchmark_id": SPLITS[split]["benchmark_id"],
        "status": "frozen",
        "frozen_on": FROZEN_ON,
        "description": _DESCRIPTIONS[split].format(seed=SPLITS[split]["seed"]),
        "objective_count": len(items),
        "category_counts": counts,
        "leak_sentinels": _sentinels(items),
        "items": items,
    }
    return validate(suite)


def render(split):
    return json.dumps(generate_suite(split), indent=2, ensure_ascii=False) + "\n"


def render_all():
    texts = {SPLITS[s]["file"]: render(s) for s in SPLITS}
    sums = "".join(f"{hashlib.sha256(texts[name].encode()).hexdigest()}  {name}\n"
                   for name in sorted(texts))
    return texts, sums


# ------------------------------------------------------------------ loading and validation


def validate(suite):
    """qfs.benchmark.validate_suite unchanged, except that its 1..8 cycle bound is 1..32."""
    bench = module("qfs.benchmark")
    view = copy.deepcopy(suite)
    for item in view.get("items", []):
        if isinstance(item, dict) and isinstance(item.get("cycles"), int) \
                and item["cycles"] > UPSTREAM_MAX_CYCLES:
            item["cycles"] = UPSTREAM_MAX_CYCLES
            expected = item.get("rubric", {}).get("expected_results")
            if isinstance(expected, list):
                item["rubric"]["expected_results"] = expected[:UPSTREAM_MAX_CYCLES]
    bench.validate_suite(view)
    for item in suite["items"]:
        if not 1 <= item["cycles"] <= MAX_CYCLES:
            raise bench.BenchmarkError(f"{item['id']}: cycles must be 1..{MAX_CYCLES}")
        if len(item["rubric"]["expected_results"]) != item["cycles"]:
            raise bench.BenchmarkError(f"{item['id']}: expected_results must have one value "
                                       "per cycle")
    return suite


def read_sums(directory=SUITE_DIR):
    out = {}
    for line in (Path(directory) / SUMS).read_text().splitlines():
        digest, name = line.split("  ", 1)
        out[name] = digest
    return out


def load(split="test", path=None):
    """Load a split, verified against SHA256SUMS and validated (see `validate`)."""
    bench = module("qfs.benchmark")
    path = Path(path) if path else SUITE_DIR / SPLITS[split]["file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    wanted = read_sums(path.parent).get(path.name)
    if digest != wanted:
        raise bench.BenchmarkError(f"{path.name}: SHA-256 {digest} does not match SHA256SUMS "
                                   f"({wanted})")
    return validate(json.loads(path.read_text()))


def score(item, results):
    """qfs.benchmark.score_item, unchanged."""
    return module("qfs.benchmark").score_item(item, results)


# ------------------------------------------------------------------ independent reference
#
# Everything below reads only the objective text and shares no code with the generator above.


class ReferenceError(ValueError):
    """An objective could not be parsed or executed by the reference."""


def _fail(message):
    raise ReferenceError(message)


def _ref_arith(x, task):
    if m := re.fullmatch(r"add (\d+)", task):
        return x + int(m[1])
    if m := re.fullmatch(r"subtract (\d+)", task):
        return x - int(m[1])
    if m := re.fullmatch(r"multiply the result by (\d+)", task):
        return x * int(m[1])
    if task == "double the result":
        return x + x
    if m := re.fullmatch(r"divide the result by (\d+)", task):
        q, r = divmod(x, int(m[1]))
        return q if r == 0 else _fail(f"inexact division: {x} / {m[1]}")
    return _fail(f"unknown arithmetic task: {task}")


def _ref_struct(a, task):
    if task == "sort the array in ascending numeric order":
        return sorted(a)
    if task == "sort the array in descending numeric order":
        return sorted(a)[::-1]
    if task == "reverse the array":
        return list(reversed(a))
    if task == "remove duplicates while preserving order":
        seen = set()
        return [n for n in a if not (n in seen or seen.add(n))]
    if m := re.fullmatch(r"keep only the (even|odd) numbers", task):
        return [n for n in a if (n % 2 == 0) == (m[1] == "even")]
    if m := re.fullmatch(r"append the number (\d+) to the end", task):
        return a + [int(m[1])]
    if m := re.fullmatch(r"insert the number (\d+) at the start", task):
        return [int(m[1])] + a
    if task == "remove the first number":
        return a[1:]
    if task == "remove the last number":
        return a[:-1]
    if task == "keep only the first three numbers":
        return a[:3]
    if m := re.fullmatch(r"(add|subtract) (\d+) (?:to|from) every number", task):
        k = int(m[2]) * (1 if m[1] == "add" else -1)
        return [n + k for n in a]
    if task == "double every number":
        return [2 * n for n in a]
    if m := re.fullmatch(r"remove every occurrence of the number (\d+)", task):
        return [n for n in a if n != int(m[1])]
    return _fail(f"unknown array task: {task}")


def _ref_literal(s, task):
    if m := re.fullmatch(r"append `([^`]+)` to the end", task):
        return s + m[1]
    if m := re.fullmatch(r"add `([^`]+)` to the front", task):
        return m[1] + s
    if task == "uppercase every letter":
        return s.upper()
    if task == "lowercase every letter":
        return s.lower()
    if task == "replace every hyphen with an underscore":
        return s.replace("-", "_")
    if task == "replace every underscore with a hyphen":
        return s.replace("_", "-")
    if m := re.fullmatch(r"wrap the whole text in (square brackets|parentheses)", task):
        left, right = ("[", "]") if m[1] == "square brackets" else ("(", ")")
        return left + s + right
    if m := re.fullmatch(r"remove the outer (square brackets|parentheses)", task):
        pair = "[]" if m[1] == "square brackets" else "()"
        if not (s.startswith(pair[0]) and s.endswith(pair[1])):
            _fail(f"no outer {m[1]} on {s!r}")
        return s[1:-1]
    if task == "delete the first character":
        return s[1:]
    if task == "delete the last character":
        return s[:-1]
    if task == "delete every digit":
        return re.sub(r"[0-9]", "", s)
    if m := re.fullmatch(r"replace every (lowercase|uppercase) `(.)` with `(.)`", task):
        return s.replace(m[2], m[3])
    return _fail(f"unknown text task: {task}")


def _word_span(text, word):
    spans = [m.span() for m in re.finditer(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])",
                                           text)]
    return spans[0] if len(spans) == 1 else _fail(f"{word!r} occurs {len(spans)} times")


def _ref_language(s, task):
    if task == "capitalize its first character":
        return s[:1].upper() + s[1:]
    if m := re.fullmatch(r"replace the word (\S+) with (\S+)", task):
        i, j = _word_span(s, m[1])
        return s[:i] + m[2] + s[j:]
    if m := re.fullmatch(r"delete the word (\S+)", task):
        i, j = _word_span(s, m[1])
        return s[j + 1:] if i == 0 else s[:i - 1] + s[j:]
    if m := re.fullmatch(r"insert the word (\S+) (after|before) the word (\S+)", task):
        i, j = _word_span(s, m[3])
        return s[:j] + " " + m[1] + s[j:] if m[2] == "after" else s[:i] + m[1] + " " + s[i:]
    if m := re.fullmatch(r"(uppercase only|capitalize|lowercase) the word (\S+)", task):
        i, j = _word_span(s, m[2])
        word = {"uppercase only": str.upper, "lowercase": str.lower,
                "capitalize": lambda w: w[:1].upper() + w[1:]}[m[1]](m[2])
        return s[:i] + word + s[j:]
    if m := re.fullmatch(r"swap the words (\S+) and (\S+)", task):
        (i, j), (k, l) = sorted((_word_span(s, m[1]), _word_span(s, m[2])))
        return s[:i] + s[k:l] + s[j:k] + s[i:j] + s[l:]
    if task == "append a period":
        return s + "." if s[-1:].isalpha() or s[-1:].isdigit() else _fail("already punctuated")
    if task == "replace the final period with an exclamation mark":
        return s[:-1] + "!" if s.endswith(".") else _fail("no final period")
    if task == "replace the final exclamation mark with a period":
        return s[:-1] + "." if s.endswith("!") else _fail("no final exclamation mark")
    if task == "remove the final punctuation mark":
        return s[:-1] if s[-1:] in (".", "!") else _fail("no final punctuation")
    return _fail(f"unknown language task: {task}")


_A_FORMS = (
    (re.compile(r"Start with the number (\d+)\. (First .*)\. Perform exactly one operation per "
                r"task in order and return only the complete current number after each task\."),
     int, _ref_arith),
    (re.compile(r"Begin with the JSON array (\[[\d,]*\])\. (First .*)\. Do one operation per "
                r"task and return the complete valid JSON array after every operation\."),
     json.loads, _ref_struct),
    (re.compile(r"Begin with the exact text `([^`]+)`\. Text inside backticks is literal, and "
                r"the backticks are not part of it\. (First .*)\. Apply exactly one text "
                r"transformation per task in order and return the complete current text each "
                r"time\."),
     str, _ref_literal),
    (re.compile(r"Begin with the exact sentence `([^`]+)`\. (First .*)\. This is constrained "
                r"language editing: perform only the named edit and return the complete current "
                r"text after every task, preserving all other wording and punctuation exactly\."),
     str, _ref_language),
)


def _ref_chain(objective):
    for pattern, parse, apply in _A_FORMS:
        if m := pattern.fullmatch(objective):
            state, out = parse(m[1]), []
            clauses = m[2].split("; ")
            for index, clause in enumerate(clauses):
                label = ("First " if index == 0 else "finally " if index == len(clauses) - 1
                         else "next " if index == 1 else "then ")
                if not clause.startswith(label):
                    _fail(f"clause {index + 1} lacks {label.strip()!r}: {clause}")
                state = apply(copy.deepcopy(state), clause[len(label):])
                out.append(state)
            return out
    return None


_B_FORM = re.compile(
    r"(?:Start with the number (?P<number>\d+)|Begin with the exact text `(?P<text>[^`]*)`|"
    r"Begin with the JSON array (?P<array>\[[\d,]*\]))\. Apply this rule repeatedly: "
    r"(?P<rule>.+?)\. After each application, check whether (?P<cond>.+?), and stop as soon "
    r"as it is true\. Return .+")


def _ref_rule(rule):
    if rule == "add the sum of the number's digits to the number":
        return lambda x: x + sum(int(c) for c in str(x))
    if m := re.fullmatch(r"add (\d+), then reverse the order of the number's digits", rule):
        a = int(m[1])

        def add_reverse(x):
            s = str(x + a)
            return int(s[::-1]) if not s.endswith("0") else _fail(f"{s} ends in 0")
        return add_reverse
    if rule == ("if the number is even, divide it by 2; if it is odd, multiply it by 3 and "
                "add 1"):
        return lambda x: x >> 1 if (x & 1) == 0 else x * 3 + 1
    if rule == "subtract the number's largest digit from the number":
        return lambda x: x - int(max(str(x)))
    if rule == "replace every letter with the next letter of the alphabet, with z becoming a":
        return lambda s: s.translate(str.maketrans(_LOWER, _LOWER[1:] + _LOWER[0]))
    if rule == "move the first word to the end":
        return lambda s: " ".join(s.split(" ")[1:] + s.split(" ")[:1])
    if rule == ("change the first lowercase letter in the text (reading from the left) to "
                "uppercase"):
        return lambda s: re.sub(r"[a-z]", lambda m: m[0].upper(), s, count=1)
    if rule == ("swap the first pair of neighbouring numbers in which the left number is larger "
                "than the right number"):
        def swap_first(a):
            for i in range(len(a) - 1):
                if a[i] > a[i + 1]:
                    return a[:i] + [a[i + 1], a[i]] + a[i + 2:]
            return _fail("no pair to swap")
        return swap_first
    if rule == "remove the first number, add 1 to it, and append it to the end":
        return lambda a: a[1:] + [a[0] + 1]
    if rule == ("subtract 1 from the largest number (if several numbers are tied for largest, "
                "use the leftmost one)"):
        def lower_leftmost_max(a):
            i = max(range(len(a)), key=lambda k: (a[k], -k))
            return a[:i] + [a[i] - 1] + a[i + 1:]
        return lower_leftmost_max
    return _fail(f"unknown rule: {rule}")


def _ref_condition(cond):
    if m := re.fullmatch(r"the number is (greater|less) than (\d+)", cond):
        t = int(m[2])
        return (lambda x: x > t) if m[1] == "greater" else (lambda x: x < t)
    if m := re.fullmatch(r"the number equals (\d+)", cond):
        return lambda x: x == int(m[1])
    if m := re.fullmatch(r"the text begins with the letter ([a-z])", cond):
        return lambda s: s[:1] == m[1]
    if m := re.fullmatch(r"the first word is (\S+)", cond):
        return lambda s: s.split(" ")[0] == m[1]
    if cond == "the text contains no lowercase letters":
        return lambda s: re.search(r"[a-z]", s) is None
    if cond == "the array is in ascending order":
        return lambda a: a == sorted(a)
    if m := re.fullmatch(r"the first number is greater than (\d+)", cond):
        return lambda a: a[0] > int(m[1])
    if cond == "all numbers in the array are equal":
        return lambda a: min(a) == max(a)
    return _fail(f"unknown condition: {cond}")


def _ref_until(objective, cap=40):
    m = _B_FORM.fullmatch(objective)
    if not m:
        return None
    state = (int(m["number"]) if m["number"] is not None else m["text"]
             if m["text"] is not None else json.loads(m["array"]))
    rule, done = _ref_rule(m["rule"]), _ref_condition(m["cond"])
    if done(state):
        _fail("the condition already holds for the starting state")
    out = []
    while not out or not done(out[-1]):
        if len(out) == cap:
            _fail(f"no stop within {cap} applications")
        state = rule(state)
        out.append(state)
    return out


_C_FORM = re.compile(
    r"Begin with the JSON state (?P<state>\{.*?\})\. Job table, where each job lists the "
    r"follow-up jobs it adds: (?P<rows>.+?)\. Repeat this step until the queue is empty: "
    r"(?P<step>.+?)\. Return the complete JSON state after every step, not the starting "
    r"state\.")


def _ref_worklist(objective, cap=40):
    m = _C_FORM.fullmatch(objective)
    if not m:
        return None
    state = json.loads(m["state"])
    table = {}
    for row in m["rows"].split("; "):
        job, _, follow = row.partition(" -> ")
        table[job] = [] if follow == "nothing" else follow.split(", ")
    step = m["step"]
    if step.startswith("remove the job at the front of the queue"):
        take = lambda q: 0  # noqa: E731
    elif step.startswith("remove the job at the end of the queue"):
        take = lambda q: len(q) - 1  # noqa: E731
    elif step.startswith("remove the alphabetically first job"):
        take = lambda q: q.index(sorted(q)[0])  # noqa: E731
    else:
        _fail(f"unknown worklist step: {step}")
    keep_sorted = step.endswith("keep the queue in alphabetical order")
    done, queue, out = list(state["done"]), list(state["queue"]), []
    while queue:
        if len(out) == cap:
            _fail(f"queue not empty after {cap} steps")
        job = queue.pop(take(queue))
        done.append(job)
        for follow in table[job]:
            if follow not in done and follow not in queue:
                queue.append(follow)
        if keep_sorted:
            queue = sorted(queue)
        out.append({"done": list(done), "queue": list(queue)})
    return out


def reference_results(objective):
    """Re-execute an objective from its text alone; the expected state after every step."""
    for parse in (_ref_chain, _ref_until, _ref_worklist):
        out = parse(objective)
        if out is not None:
            return out
    return _fail(f"objective matches no v4 template: {objective[:80]}")


# ------------------------------------------------------------------ tokens (tokenizer only)


def load_tokenizer(path=TOKENIZER):
    from transformers import AutoTokenizer  # tokenizer files only; no weights, no device
    return AutoTokenizer.from_pretrained(path)


def _forms(value):
    forms = [json.dumps(value, ensure_ascii=False), _compact(value)]
    if isinstance(value, str):
        forms.append(value)
    return forms


def token_report(tok, suite):
    """Per item: the largest state, the reference ledger, the largest single-step proxy
    (incoming state + task + new state, each as JSON), and the objective, in tokens."""
    def count(text):
        return len(tok.encode(text, add_special_tokens=False))

    rows = []
    for item in suite["items"]:
        prov = item["provenance"]
        steps = prov["gold_steps"]
        states = [steps[0]["incoming"]] + item["rubric"]["expected_results"]
        rows.append({
            "id": item["id"],
            "cycles": item["cycles"],
            "state_tokens": max(count(f) for s in states for f in _forms(s)),
            "ledger_tokens": max(count(f) for f in _forms(prov["reference_ledger"])),
            "step_tokens": max(count(json.dumps(st["incoming"], ensure_ascii=False))
                               + count(st["task"])
                               + count(json.dumps(st["expected"], ensure_ascii=False))
                               for st in steps),
            "objective_tokens": count(item["objective"]),
        })
    return rows


# ------------------------------------------------------------------ CLI


def write(directory=SUITE_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    texts, sums = render_all()
    for name, text in texts.items():
        (directory / name).write_text(text)
    (directory / SUMS).write_text(sums)
    return sums


def check(directory=SUITE_DIR):
    """Problems found when regenerating and re-verifying; empty when the suite is intact."""
    directory = Path(directory)
    problems = []
    texts, sums = render_all()
    for name, text in texts.items():
        path = directory / name
        if not path.exists() or path.read_bytes() != text.encode():
            problems.append(f"{name}: regenerated bytes differ from the file")
    if not (directory / SUMS).exists() or (directory / SUMS).read_text() != sums:
        problems.append(f"{SUMS}: does not match the regenerated files")
    for split, spec in SPLITS.items():
        try:
            suite = load(split, directory / spec["file"])
        except Exception as exc:  # report every failure, not only the first
            problems.append(f"{spec['file']}: {exc}")
            continue
        for item in suite["items"]:
            try:
                if reference_results(item["objective"]) != item["rubric"]["expected_results"]:
                    problems.append(f"{item['id']}: reference disagrees with expected_results")
            except ReferenceError as exc:
                problems.append(f"{item['id']}: {exc}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and compare byte-for-byte; write nothing")
    parser.add_argument("--tokens", action="store_true",
                        help="print Qwen3.5-4B token maxima per split (tokenizer only)")
    parser.add_argument("--dir", type=Path, default=SUITE_DIR)
    args = parser.parse_args(argv)
    if args.check:
        problems = check(args.dir)
        print(json.dumps({"ok": not problems, "problems": problems,
                          "sha256": read_sums(args.dir) if (args.dir / SUMS).exists() else {}},
                         indent=2))
        return 1 if problems else 0
    if args.tokens:
        tok = load_tokenizer()
        out = {}
        for split in SPLITS:
            rows = token_report(tok, load(split, args.dir / SPLITS[split]["file"]))
            out[split] = {key: max(row[key] for row in rows)
                          for key in ("state_tokens", "ledger_tokens", "step_tokens",
                                      "objective_tokens")}
        print(json.dumps(out, indent=2))
        return 0
    sums = write(args.dir)
    print(sums, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
