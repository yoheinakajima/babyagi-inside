"""E15: host screen of an adaptive-length BabyAGI ledger loop against budget-forced plain chat.

    python -m harness.adaptive_screen check            # tokenizer only: ROM shape, spans, digest
    python -m harness.adaptive_screen run ARM SPLIT    # steps dev-check, H-adapt-test, ...
    python -m harness.adaptive_screen analyze          # step analysis
    python -m harness.adaptive_screen smoke ARM OBJECTIVE [--length L]   # pre-lock GPU check

Stock Qwen3.5-4B, no compiled checkpoint. The host simulates the loop the E15 design
(experiments/drafts/E15-unbounded-design.md, sections 2-4) would compile: M3's looped body
(read result.json, read ledger.json, execute with a 256-token think span, decide) plus M2's
typed CONTINUE/DONE stage, with the worklist (M4a) kept in the result file. `run_native`
walks a static stage list, so this module owns the loop and reuses qfs.agent_native's
NativeRunner, choose_token (through NativeRunner.pick), stage_view and
visible_response_tokens. No ROM text states the gold length: the only number a stage
renders is its own cycle, zero-padded to two tokens as a compiled CYC counter would.

Arms (frozen in experiments/E15-adaptive-screen/prereg.md):

    H-adapt     Qwen chooses CONTINUE (`1`) or DONE (`2`) after every execute, the host
                restricting the choice to those two tokens; cap 40 cycles.
    H-oracle    the same loop and ROM, with the host forcing CONTINUE until the gold length
                L and then DONE: a fixed-length loop.
    P-F2        plain chat, budget-forced as E14 F2 (harness.plain_forced.generate_forced),
                thinking cap (L+1)*256 and answer cap L*256; E14's prompt without its
                "exactly N entries" clause, parsed without knowing L.
    H-adapt-v3  H-adapt on the frozen v3 suite, as a regression check.

Scoring is qfs.benchmark.score_item (exact_state_chain), statistics are
qfs.end_to_end._paired_summary, read with harness.port.verdict. Every H-adapt and H-oracle
trace gets the design's feasibility audit (qwenFS limits the host does not enforce).
Records live under build/e15-screen/<arm>/<split>/ and resume by identity. Each command
exits 0 whenever it executed, with its verdict in metrics.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from harness import plain_forced, port
from harness.upstream import ROOT, SUITE_V3, module
from lab.repo import sha256_file

EXPERIMENT = "E15"
EXPERIMENT_DIR = ROOT / "experiments" / "E15-adaptive-screen"
RECORDS = ROOT / "build" / "e15-screen"
BASE = "models/Qwen3.5-4B"
DEVICE = "mps"
ARMS = ("H-adapt", "H-oracle", "P-F2", "H-adapt-v3")
LOOP_ARMS = ("H-adapt", "H-oracle", "H-adapt-v3")
SPLITS = ("dev", "test")

# Frozen in prereg.md (Arms, Seeds and decoding).
THINK = 256  # reasoning tokens per execute stage (the protocol's free span)
WRITE_LIMIT = 240  # payload tokens per result write (run_native's limit_per_write)
LEDGER_LIMIT = 600  # payload tokens for the one ledger write: qwenFS NMAX (notes.md)
CAP = 40  # compiled cycle cap (design M3, Cmax)
CONTROL_SURROGATE = 0.25  # host view of each control token, as in E9
CONTINUE_TEXT, DONE_TEXT = "1", "2"
CONTINUE, DONE = 16, 17  # their single token ids (checked by `check`)
NATIVE_VOCAB_STOP = 248044  # qfs.agent.NATIVE_VOCAB_STOP: no control tokens in free spans
LEDGER, RESULT = "ledger.json", "result.json"
SETTINGS = {"model": BASE, "device": DEVICE, "dtype": "bfloat16", "temperature": 0.0,
            "think_tokens": THINK, "write_limit": WRITE_LIMIT, "ledger_write_limit": LEDGER_LIMIT,
            "cap": CAP, "control_surrogate_scale": CONTROL_SURROGATE,
            "choice_tokens": [CONTINUE_TEXT, DONE_TEXT]}

# The feasibility audit (design section 4): limits of compiled qwenFS and the controller.
MAX_WRITES_PER_FILE = 62  # 6-bit versions saturate at 63 (circuit.py:553)
MAX_FILES = 32  # NFILE
MAX_STORED_TOKENS = 600  # NMAX: file text is exact only to 600 tokens
FLAG_STORED_TOKENS = 500
MAX_SPAN = 600  # stage span against NMAX
FIT_SPAN = 482  # about what the controller's tail dims allow (design M3); reported only
# Audit failures with a preregistered fix (prereg.md, Analysis plan). Only one exists: past
# 62 writes, result files rotate (E2's rotation item). It cannot occur with a 40-cycle cap.
PREREGISTERED_FIXES = {"writes_per_file": "rotate result files (E2 rotation item)"}

BUCKETS = ((1, 8), (9, 16), (17, 24), (25, 32))
STOP_TARGET = 0.9  # GO needs exact stops on >= 90% of stop-eligible items
BOOTSTRAP = {"samples": 10000, "seed": 0}  # qfs.end_to_end._clustered_chain_delta_ci


# ---------------------------------------------------------------- protocol text

# One ROM for every category. A: the objective lists its operations -> `tasks`.
# B: one rule applied until a stated condition holds -> `rule` and `until`.
# C: a worklist -> also `rule` and `until`; the queue itself lives in the result state.
CREATE = (
    "INITIAL TASK — Write the task ledger for the objective in the first user message. Do not "
    "perform any of its work and do not calculate any result. `initial` is the exact input "
    "state the objective supplies (the value given by Begin with or Start with, or the initial "
    "JSON data), or null if there is none. Input data is never a task. If the objective lists "
    "its operations, `tasks` is an object that numbers those operations in order with "
    "two-digit string keys (\"01\", \"02\", and so on), one short string per operation clause "
    "(First, Next, Then, Finally), each keeping its verb and operands exactly, with no "
    "initialization task; `rule` and `until` are then null. If the objective instead "
    "gives one rule to repeat until a stated condition holds, such as processing a queue or "
    "worklist one item at a time, `tasks` is null, `rule` is that rule together with any "
    "table it uses, and `until` is that stopping condition, all kept exact. Preserve all "
    "literals and constraints. The file "
    "already begins `{`; continue with exactly `\"initial\":...,\"tasks\":...,\"rule\":...,"
    "\"until\":...}`, then close the text parameter, function, and tool call. Emit no "
    "explanation."
)
EXECUTE = (
    "EXECUTE STEP {cc}. The ledger just read is the plan. The result file read before it holds "
    "the current state; if it was not found, this is step 01 and the current state is the "
    "ledger's `initial`. Do exactly one step. If the ledger's `tasks` is set, perform the task "
    "under key \"{cc}\" and no other task. If its `rule` is set "
    "instead, apply that rule once to the current state. Use the private think span to work "
    "through it. When the "
    "controller opens the result object, emit one valid JSON value for the complete new "
    "state, in the form the objective asks for, then `}`, then close the text parameter, "
    "function, and tool call. Do not add commentary."
)
DECIDE = (
    "DECIDE AFTER STEP {cc}. The result just saved is the new state. If the ledger's `tasks` "
    "is set, the objective is finished when \"{cc}\" is its last key. If "
    "its `rule` is set instead, the objective is finished only when `until` is already true "
    "of the new state exactly as saved; check it literally against that state. "
    "Answer with one digit: 1 if another step is needed, 2 if the objective is finished."
)
TEMPLATE_DIGITS = {"01", "02", "1", "2"}  # the only numerals the templates state (tests check this)
OPEN_USER = "<|im_end|>\n<|im_start|>user\n"
OPEN_THINK = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
EMPTY_THINK = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _cc(cycle: int) -> str:
    if not 1 <= cycle <= 99:
        raise ValueError(f"cycle {cycle} has no two-digit rendering")
    return f"{cycle:02d}"


def read_call(name: str) -> str:
    return ("<tool_call>\n<function=read_file>\n<parameter=name>\n" + name
            + "\n</parameter>\n</function>\n</tool_call>")


def write_open(name: str, text_prefix: str) -> str:
    return ("<tool_call>\n<function=write_file>\n<parameter=name>\n" + name
            + "\n</parameter>\n<parameter=text>\n" + text_prefix)


def stage_texts(cycle: int) -> dict:
    """Every forced text of one cycle's body: what the compiled ROM would print."""
    cc = _cc(cycle)
    return {"read_result": read_call(RESULT), "read_ledger": read_call(LEDGER),
            "execute": OPEN_USER + EXECUTE.replace("{cc}", cc) + OPEN_THINK,
            "execute_suffix": "\n</think>\n\n" + write_open(RESULT, '{"result":'),
            "decide": OPEN_USER + DECIDE.replace("{cc}", cc) + EMPTY_THINK}


def create_text() -> str:
    # The first stage follows the objective's prefill directly, as in ledger_pointer_program.
    return "\n<|im_start|>user\n" + CREATE + EMPTY_THINK + write_open(LEDGER, "{")


def plain_prompt(item: dict) -> str:
    """E14's plain prompt (plain_forced.plain_prompt) without the clause that states N."""
    return item["objective"] + PLAIN_SUFFIX


PLAIN_SUFFIX = (
    "\n\nSolve the entire objective in one response. Preserve the result after every "
    "successive operation, not just the final answer. Do not include the initial state as a "
    "result. Each entry must be the complete result after that operation: a JSON number for "
    "numeric work, the requested JSON value for structured work, or a JSON string for text "
    "(use `\\n` inside multiline strings). End with `FINAL_JSON: ` followed by exactly one "
    "valid JSON object shaped `{\"results\":[...]}`, whose `results` array has one entry per "
    "operation, in task order. Do not put text after the object."
)


def rom_sha256() -> str:
    """Digest of every forced text the arms use (cycles 1..CAP) and the plain suffix."""
    payload = {"create": create_text(), "cycles": [stage_texts(c) for c in range(1, CAP + 1)],
               "plain_suffix": PLAIN_SUFFIX, "choices": [CONTINUE_TEXT, DONE_TEXT]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------- stages


def _encode(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def _decision(owner: str) -> dict:
    return {"name": "continue_or_done", "mode": "typed_choice", "owner": owner,
            "allowed_token_ids": [CONTINUE, DONE], "allowed_text": [CONTINUE_TEXT, DONE_TEXT],
            "token_count": 1}


def create_stage(tok) -> dict:
    return {"cycle": 0, "kind": "create", "op": "write", "file": LEDGER, "text_prefix": "{",
            "write_limit": LEDGER_LIMIT, "rom": _encode(tok, create_text())}


def cycle_stages(tok, cycle: int, owner: str = "model", think: int = THINK) -> list[dict]:
    """The looped body for one cycle, in the stage schema qfs.agent_native.stage_view reads."""
    texts = stage_texts(cycle)
    return [
        {"cycle": cycle, "kind": "read", "op": "read", "file": RESULT,
         "rom": _encode(tok, texts["read_result"])},
        {"cycle": cycle, "kind": "read", "op": "read", "file": LEDGER,
         "rom": _encode(tok, texts["read_ledger"])},
        {"cycle": cycle, "kind": "execute", "op": "write", "file": RESULT,
         "text_prefix": '{"result":', "write_limit": WRITE_LIMIT, "free_span": think,
         "free_vocab_stop": NATIVE_VOCAB_STOP, "rom": _encode(tok, texts["execute"]),
         "rom_suffix": _encode(tok, texts["execute_suffix"])},
        {"cycle": cycle, "kind": "decide", "op": "decide", "decision": _decision(owner),
         "rom": _encode(tok, texts["decide"])},
    ]


def span(stage: dict) -> int:
    """The stage's controller span, as qfs.agent.Layout counts it."""
    return (len(stage["rom"]) + int(stage.get("decision", {}).get("token_count", 1)
                                    if "decision" in stage else 0)
            + int(stage.get("free_span", 0)) + len(stage.get("rom_suffix", [])))


def static_check(tok) -> dict:
    """Tokenizer only: typed choice ids, fixed-offset cycle digits, and stage spans."""
    problems = []
    choices = [_encode(tok, CONTINUE_TEXT), _encode(tok, DONE_TEXT)]
    if choices != [[CONTINUE], [DONE]]:
        problems.append(f"choice tokens are {choices}, expected [[{CONTINUE}], [{DONE}]]")
    digits = {tok.convert_tokens_to_ids(str(d)) for d in range(10)}
    bodies = [cycle_stages(tok, c) for c in range(1, CAP + 1)]
    shapes = {}
    for index, kind in enumerate(("read_result", "read_ledger", "execute", "decide")):
        runs = [body[index]["rom"] + body[index].get("rom_suffix", []) for body in bodies]
        lengths = sorted({len(run) for run in runs})
        varying = sorted({n for run in runs for n, token in enumerate(run)
                          if len(run) == len(runs[0]) and token != runs[0][n]})
        if len(lengths) != 1:
            problems.append(f"{kind}: ROM length varies with the cycle ({lengths})")
        if any(run[n] not in digits for run in runs for n in varying if n < len(run)):
            problems.append(f"{kind}: a cycle-dependent position is not a digit token")
        shapes[kind] = {"tokens": lengths[0] if len(lengths) == 1 else lengths,
                        "cycle_digit_positions": varying}
    spans = {"create": span(create_stage(tok)),
             **{kind: span(stage) for kind, stage in
                zip(("read_result", "read_ledger", "execute", "decide"), bodies[0])}}
    if max(spans.values()) > MAX_SPAN:
        problems.append(f"largest stage span {max(spans.values())} exceeds {MAX_SPAN}")
    return {"pass": not problems, "problems": problems, "rom_sha256": rom_sha256(),
            "shapes": shapes, "spans": spans, "max_span": max(spans.values()),
            "within_fit_span": max(spans.values()) <= FIT_SPAN, "fit_span": FIT_SPAN}


# ---------------------------------------------------------------- the loop


def run_loop(runner, tok, objective: str, oracle_length: int | None = None, cap: int = CAP,
             think: int = THINK, surrogate: float | None = CONTROL_SURROGATE) -> dict:
    """One item through the adaptive loop on a NativeRunner-like object.

    create (write ledger.json), then per cycle: read result.json, read ledger.json, execute
    (think span, then write result.json), decide. The decide stage is a typed choice
    between `1` (CONTINUE) and `2` (DONE) with no tool call. DONE, or the end of cycle `cap`,
    halts. With `oracle_length` the host forces CONTINUE before that cycle and DONE at it;
    Qwen's own pick is still recorded. A write that never closes its call stops the run.
    """
    native, F = module("qfs.agent_native"), module("qfs.circuit")
    chat = module("qfs.chat")
    if oracle_length is not None and not 1 <= oracle_length <= cap:
        raise ValueError(f"oracle length {oracle_length} is outside 1..{cap}")
    wall_begin = time.monotonic()
    initial = tok.apply_chat_template(
        [{"role": "user", "content": objective}], tools=chat.TOOLS, tokenize=True,
        add_generation_prompt=False, enable_thinking=False, return_dict=False)
    runner.queue(list(initial))
    files, phases, decisions, results = {}, [], [], []
    tokens = Counter(objective_prefill=len(initial))

    def write(stage: dict) -> bool:
        view = native.stage_view(stage)
        limit = stage["write_limit"]
        begin = time.monotonic()
        generated = []
        logits = runner.start(view.prefix)
        for _ in range(view.free_span):
            token = runner.pick(logits, vocab_stop=view.free_vocab_stop)
            generated.append(token)
            logits = runner.feed([token])
        if view.free_span:
            logits = runner.feed(view.suffix)
        payload = 0
        while payload < limit:
            token = runner.pick(logits)
            generated.append(token)
            payload += 1
            logits = runner.feed([token])
            if token == F.TCEND:
                break
        call = (view.prefix + generated[:view.free_span] + view.suffix
                + generated[view.free_span:])
        function, parameters = native._call(stage, tok, call)
        name, text = parameters.get("name"), parameters.get("text")
        completed = F.TCEND in call and function == "write_file" and bool(name) \
            and text is not None
        stored = None
        if completed:
            files[name] = _encode(tok, text)
            stored = len(files[name])
            response = native.visible_response_tokens(tok, function, name)
            runner.queue(response)
            tokens["forced_responses"] += len(response)
        tokens["forced_rom"] += len(view.prefix) + len(view.suffix)
        tokens["model_reasoning"] += view.free_span
        tokens["model_write"] += payload
        phases.append({"cycle": stage["cycle"], "kind": stage["kind"], "op": "write",
                       "file": stage["file"], "span": span(stage),
                       "forced_prefix_tokens": len(view.prefix),
                       "forced_suffix_tokens": len(view.suffix),
                       "reasoning_tokens": view.free_span, "payload_tokens": payload,
                       "write_limit": limit, "hit_limit": payload >= limit and not (
                           generated and generated[-1] == F.TCEND),
                       "model_token_ids": generated,
                       "reasoning_text": tok.decode(generated[:view.free_span]),
                       "payload_text": tok.decode(generated[view.free_span:]),
                       "function": function, "parameters": parameters,
                       "stored_name": name if completed else None, "stored_tokens": stored,
                       "completed": completed, "seconds": time.monotonic() - begin})
        return completed

    def read(stage: dict) -> None:
        view = native.stage_view(stage)
        function, parameters = native._call(stage, tok, view.prefix)
        name = parameters.get("name")
        body = files.get(name)
        response = native.visible_response_tokens(tok, function, name, body)
        runner.queue(view.prefix + response)  # fed at the next stage's control surrogate
        tokens["forced_rom"] += len(view.prefix)
        tokens["forced_responses"] += len(response)
        phases.append({"cycle": stage["cycle"], "kind": "read", "op": "read",
                       "file": stage["file"], "span": span(stage), "function": function,
                       "parameters": parameters, "read_found": body is not None,
                       "read_tokens": len(body) if body is not None else 0})

    def decide(stage: dict) -> int:
        view = native.stage_view(stage)
        logits = runner.start(view.prefix)
        allowed = {str(token): float(logits[token]) for token in view.allowed}
        unrestricted = int(logits.argmax())
        pick = runner.pick(logits, view.allowed)  # greedy over {1, 2}; no RNG at T = 0
        cycle = stage["cycle"]
        forced = None
        if oracle_length is not None:
            forced = DONE if cycle >= oracle_length else CONTINUE
        token = pick if forced is None else forced
        runner.queue([token])  # the next control token follows the choice itself
        tokens["forced_rom"] += len(view.prefix)
        tokens["model_choices" if forced is None else "forced_choices"] += 1
        entry = {"cycle": cycle, "token": token, "model_pick": pick, "forced": forced,
                 "unrestricted_top_token": unrestricted, "native_allowed_logits": allowed,
                 "continue_margin": allowed[str(CONTINUE)] - allowed[str(DONE)]}
        decisions.append(entry)
        phases.append({"cycle": cycle, "kind": "decide", "op": "decide", "span": span(stage),
                       "decision": entry})
        return token

    def control() -> None:
        if surrogate is not None:
            runner.flush_control_surrogate(surrogate)
            tokens["control_surrogates"] += 1

    stop = None
    control()
    if not write(create_stage(tok)):
        stop = {"reason": "write_failed", "cycle": 0, "stage": "create"}
    cycle = 0
    while stop is None:
        cycle += 1
        owner = "model" if oracle_length is None else "host"
        read_result, read_ledger, execute, decision = cycle_stages(tok, cycle, owner, think)
        for stage in (read_result, read_ledger):
            control()
            read(stage)
        control()
        if not write(execute):
            stop = {"reason": "write_failed", "cycle": cycle, "stage": "execute"}
            break
        results.append(result_value(phases[-1]["parameters"].get("text")))
        control()
        if decide(decision) == DONE:
            stop = {"reason": "done", "cycle": cycle, "stage": "decide"}
        elif cycle >= cap:
            stop = {"reason": "cap", "cycle": cycle, "stage": "decide"}

    return {"schema_version": 1, "harness": "e15-adaptive-loop",
            "compiled_controller_executed": False,
            "mode": "adaptive" if oracle_length is None else "oracle",
            "oracle_length": oracle_length, "cap": cap, "think_tokens": think,
            "control_surrogate_scale": surrogate, "cycles": len(results), "stop": stop,
            "decisions": decisions, "results": results, "phases": phases,
            "latest_files": {name: tok.decode(body) for name, body in files.items()},
            "token_provenance": dict(tokens),
            "model_tokens": tokens["model_reasoning"] + tokens["model_write"]
                            + tokens["model_choices"],
            "context_tokens": len(runner.context),
            "forward_seconds": runner.forward_seconds,
            "wall_seconds": time.monotonic() - wall_begin}


def result_value(text):
    """The `result` of one result.json write, kept aligned by cycle (None if unparsable).

    qfs.agent_suite.native_results skips an unparsable write, which shifts every later
    result to an earlier step; here a bad write stays a wrong answer at its own step.
    """
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return value.get("result") if isinstance(value, dict) else None


def ledger_shape(trace: dict) -> dict:
    """Which ledger form Qwen wrote: `tasks` (A) or `rule` + `until` (B and C)."""
    text = trace["latest_files"].get(LEDGER)
    try:
        ledger = json.loads(text) if text is not None else None
    except json.JSONDecodeError:
        ledger = None
    if not isinstance(ledger, dict):
        return {"parse_ok": False, "shape": None, "tasks": None, "has_until": False}
    tasks = ledger.get("tasks")
    has_rule = isinstance(ledger.get("rule"), str) and bool(ledger["rule"].strip())
    shape = ("tasks" if isinstance(tasks, (dict, list)) and tasks and not has_rule else
             "rule" if has_rule and not tasks else "mixed" if has_rule else None)
    # The ROM asks for keys "01".."NN" in order; a list or other keys are recorded, not failed.
    numbered = isinstance(tasks, dict) and list(tasks) == [f"{i:02d}" for i in range(1, len(tasks) + 1)]
    return {"parse_ok": True, "shape": shape,
            "tasks": len(tasks) if isinstance(tasks, (dict, list)) else None,
            "numbered": numbered, "has_until": isinstance(ledger.get("until"), str)}


# ---------------------------------------------------------------- audit


def audit(trace: dict) -> dict:
    """The design's feasibility audit: what compiled qwenFS and the controller can hold.

    Violations: more than 62 writes to one file, more than 32 files, a stored text over 600
    tokens, a stage span over 600, more than 40 cycles. Flags (reported, not failures): a
    stored text over 500 tokens, a payload write that reached its limit.
    """
    violations, flags = [], []
    writes = Counter()
    for phase in trace["phases"]:
        if phase.get("span", 0) > MAX_SPAN:
            violations.append({"check": "stage_span", "cycle": phase["cycle"],
                               "kind": phase["kind"], "value": phase["span"],
                               "limit": MAX_SPAN})
        if phase["op"] != "write":
            continue
        if phase.get("hit_limit"):
            flags.append({"check": "write_hit_limit", "cycle": phase["cycle"],
                          "kind": phase["kind"], "value": phase["payload_tokens"],
                          "limit": phase["write_limit"]})
        if not phase["completed"]:
            continue
        writes[phase["stored_name"]] += 1
        stored = phase["stored_tokens"]
        if stored > MAX_STORED_TOKENS:
            violations.append({"check": "stored_tokens", "cycle": phase["cycle"],
                               "file": phase["stored_name"], "value": stored,
                               "limit": MAX_STORED_TOKENS})
        elif stored > FLAG_STORED_TOKENS:
            flags.append({"check": "stored_tokens_over_500", "cycle": phase["cycle"],
                          "file": phase["stored_name"], "value": stored})
    for name, count in sorted(writes.items()):
        if count > MAX_WRITES_PER_FILE:
            violations.append({"check": "writes_per_file", "file": name, "value": count,
                               "limit": MAX_WRITES_PER_FILE})
    if len(writes) > MAX_FILES:
        violations.append({"check": "files", "value": len(writes), "limit": MAX_FILES})
    if trace["cycles"] > CAP:
        violations.append({"check": "cycles", "value": trace["cycles"], "limit": CAP})
    stored = [p["stored_tokens"] for p in trace["phases"]
              if p["op"] == "write" and p["completed"]]
    return {"pass": not violations, "violations": violations, "flags": flags,
            "unfixed": sorted({v["check"] for v in violations} - set(PREREGISTERED_FIXES)),
            "max_writes_per_file": max(writes.values(), default=0), "files": len(writes),
            "max_stored_tokens": max(stored, default=0),
            "max_span": max((p.get("span", 0) for p in trace["phases"]), default=0),
            "cycles": trace["cycles"]}


# ---------------------------------------------------------------- outcomes


def caps(length: int) -> tuple[int, int]:
    """P-F2's (thinking cap, answer cap): the loop's maximum model-token allowance at the gold
    length L, split as the loop splits it. Thinking gets the ledger write and the L reasoning
    spans; the answer gets the L result writes and the L one-token decisions."""
    return LEDGER_LIMIT + length * THINK, length * (WRITE_LIMIT + 1)


def parse_open(text: str):
    """The last JSON object with a `results` list, of any length (L is not used)."""
    values = module("qfs.ceiling")._json_values(text)
    found = None
    for value in values:
        if isinstance(value, dict) and isinstance(value.get("results"), list):
            found = value["results"]
    return found


def bucket(length: int) -> str:
    for low, high in BUCKETS:
        if low <= length <= high:
            return f"{low}-{high}"
    return f">{BUCKETS[-1][1]}"


def first_failure(score: dict, count: int) -> dict:
    """The first failed step and how it failed: wrong, missing (stopped early), or extra."""
    checks = score["automated_checks"]
    for index, check in enumerate(checks, 1):
        if not check["pass"]:
            return {"first_failed_step": index,
                    "failure_kind": "missing" if index > count else "wrong"}
    if count > len(checks):
        return {"first_failed_step": None, "failure_kind": "extra"}
    return {"first_failed_step": None, "failure_kind": None}


def loop_stop(trace: dict, length: int, checks: list[dict]) -> dict:
    """Stop outcome (exact, early, late, cap, aborted) and stop eligibility.

    An item is stop-eligible when every step executed before the gold stop point was exact,
    so its stop decision was made on a correct state: steps 1..k for an early DONE at k,
    steps 1..L otherwise. A run aborted by a failed write at or before cycle L made no
    judgeable stop; aborted after L, it had already chosen CONTINUE at L (late).
    """
    tokens = [d["token"] for d in trace["decisions"]]
    done = next((n for n, token in enumerate(tokens, 1) if token == DONE), None)
    if done is not None:
        outcome = "exact" if done == length else "early" if done < length else "late"
    elif trace["stop"]["reason"] == "write_failed":
        outcome = "late" if len(tokens) >= length else "aborted"
    else:
        outcome = "cap"
    upto = done if outcome == "early" else length
    eligible = outcome != "aborted" and all(c["pass"] for c in checks[:upto])
    return {"stop_outcome": outcome, "stop_cycle": done, "stop_eligible": eligible}


def plain_stop(results, length: int) -> str:
    if results is None:
        return "unparsed"
    return "exact" if len(results) == length else "early" if len(results) < length else "late"


# ---------------------------------------------------------------- suites


def suite_file(arm: str, split: str) -> Path:
    """The suite JSON an arm runs. Only the path comes from the suite helper (harness.suite_v4)."""
    if arm == "H-adapt-v3":
        if split != "test":
            raise ValueError("H-adapt-v3 runs the frozen v3 suite, which has only a test split")
        return SUITE_V3
    from harness.suite_v4 import SPLITS as V4_SPLITS, SUITE_DIR as V4_DIR  # the path only

    return V4_DIR / V4_SPLITS[split]["file"]


def load_items(arm: str, split: str) -> tuple[list[dict], str]:
    path = suite_file(arm, split)
    digest = sha256_file(path)
    if path == SUITE_V3:
        suite = module("qfs.benchmark").load_suite(path)  # verifies the frozen checksum
    else:  # v4 lengths exceed qfs.benchmark.validate_suite's bound of 8
        sums = {name: value for value, name in (
            line.split("  ", 1) for line in (path.parent / "SHA256SUMS").read_text().splitlines())}
        if sums.get(path.name) != digest:
            raise ValueError(f"{path.name}: SHA-256 {digest} is not the one in SHA256SUMS")
        suite = json.loads(path.read_text())
    items = suite["items"]
    for item in items:
        check_item(item)
    if len({item["id"] for item in items}) != len(items):
        raise ValueError(f"{path}: duplicate item ids")
    return items, digest


def horizon(item: dict) -> str:
    """v4's horizon category (long_chain, rule_until, worklist); v3 items are `v3`."""
    return item.get("provenance", {}).get("horizon_category", "v3")


def check_item(item: dict) -> None:
    """What this harness relies on (v4 lengths exceed qfs.benchmark.validate_suite's 8)."""
    rubric = item.get("rubric", {})
    ok = (isinstance(item.get("id"), str) and isinstance(item.get("objective"), str)
          and isinstance(item.get("category"), str) and isinstance(item.get("cycles"), int)
          and 1 <= item["cycles"] <= 32 and rubric.get("type") == "exact_state_chain"
          and len(rubric.get("expected_results", [])) == item["cycles"])
    if not ok:
        raise ValueError(f"item {item.get('id')!r} is not an exact_state_chain item of length "
                         "1..32")


# ---------------------------------------------------------------- run


def runner_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(plain_forced.__file__)):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def identity(arm: str, split: str, item: dict, suite_hash: str, runner: str | None = None) -> dict:
    row = {"experiment": EXPERIMENT, "arm": arm, "split": split, "item_id": item["id"],
           "suite_sha256": suite_hash, "rom_sha256": rom_sha256(),
           "runner_sha256": runner or runner_sha256(), "settings": SETTINGS}
    if arm == "H-oracle":
        row["oracle_length"] = item["cycles"]
    if arm == "P-F2":
        row["think_cap"], row["answer_cap"] = caps(item["cycles"])
    return row


def loop_record(arm: str, item: dict, trace: dict, ident: dict) -> dict:
    score = module("qfs.benchmark").score_item(item, trace["results"])
    return {"schema_version": 1, "identity": ident, "item_id": item["id"],
            "horizon": horizon(item), "category": item["category"], "cycles": item["cycles"],
            "actual_results": trace["results"], "score": score,
            **first_failure(score, len(trace["results"])),
            **loop_stop(trace, item["cycles"], score["automated_checks"]),
            "audit": audit(trace), "ledger": ledger_shape(trace), "trace": trace}


def plain_record(item: dict, generation: dict, ident: dict) -> dict:
    ceiling, benchmark = module("qfs.ceiling"), module("qfs.benchmark")
    actual = parse_open(generation["text"])
    score = benchmark.score_item(item, actual or [])
    return {"schema_version": 1, "identity": ident, "item_id": item["id"],
            "horizon": horizon(item), "category": item["category"], "cycles": item["cycles"],
            "prompt_suffix_sha256": hashlib.sha256(PLAIN_SUFFIX.encode()).hexdigest(),
            "generation": generation, "parse_ok": actual is not None,
            "answer_parse_ok": parse_open(generation["answer_text"]) is not None,
            # Diagnostic only: E14's parser, which picks an array of exactly L entries.
            "gold_length_parse_ok": ceiling.parse_plain(generation["text"],
                                                        item["cycles"]) is not None,
            "actual_results": actual or [], "score": score,
            **first_failure(score, len(actual or [])),
            "stop_outcome": plain_stop(actual, item["cycles"])}


def run_items(arm: str, split: str, items: list[dict], suite_hash: str, execute,
              folder: Path) -> dict:
    """Run every item without a record of matching identity; keep the rest."""
    folder.mkdir(parents=True, exist_ok=True)
    runner = runner_sha256()
    ran = reused = 0
    for index, item in enumerate(items, 1):
        ident = identity(arm, split, item, suite_hash, runner)
        path = folder / f"{item['id']}.json"
        if path.exists() and json.loads(path.read_text()).get("identity") == ident:
            reused += 1
            continue
        record = execute(item, ident)
        port._write(path, record)
        ran += 1
        print(json.dumps({"item": index, "item_id": item["id"], "cycles": item["cycles"],
                          "pass": record["score"]["automated_pass"],
                          "stop": record.get("stop_outcome"),
                          "results": len(record["actual_results"])}), flush=True)
    return {"ran": ran, "reused": reused}


def _config() -> dict:
    from ruamel.yaml import YAML

    return YAML(typ="safe").load((EXPERIMENT_DIR / "config.yaml").read_text()) or {}


def run(arm: str, split: str, output: Path, only: list[str] | None = None) -> dict:
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    items, suite_hash = load_items(arm, split)
    if only:
        if split != "dev":
            raise ValueError("--only selects dev items; test items always run in full")
        unknown = sorted(set(only) - {item["id"] for item in items})
        if unknown:
            raise ValueError(f"not dev items: {', '.join(unknown)}")
        items = [item for item in items if item["id"] in set(only)]
    frozen_rom = _config().get("rom_sha256")
    if split == "test" and frozen_rom != rom_sha256():  # stop before any decode
        return {"lab_status": "invalid", "arm": arm, "split": split,
                "invalid_reason": f"ROM digest {rom_sha256()} is not config.yaml's {frozen_rom}"}
    base = ROOT / BASE
    tok = AutoTokenizer.from_pretrained(base)
    static = static_check(tok)
    if not static["pass"]:
        return {"lab_status": "invalid", "arm": arm, "split": split,
                "invalid_reason": "static check failed", "static_check": static}
    model = Qwen3_5ForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(DEVICE).eval()
    native = module("qfs.agent_native")

    def execute(item: dict, ident: dict) -> dict:
        if arm == "P-F2":
            generation = plain_forced.generate_forced(model, tok, plain_prompt(item),
                                                      *caps(item["cycles"]))
            return plain_record(item, generation, ident)
        runner = native.NativeRunner(model, tok, DEVICE, temperature=0.0, top_p=0.9, seed=0)
        trace = run_loop(runner, tok, item["objective"],
                         oracle_length=item["cycles"] if arm == "H-oracle" else None)
        return loop_record(arm, item, trace, ident)

    counts = run_items(arm, split, items, suite_hash, execute, RECORDS / arm / split)
    summary = arm_summary(load_records(arm, split))
    port._write(output / f"{arm}-{split}.json", summary)
    return {"arm": arm, "split": split, **counts, "static_check": static, **summary}


# ---------------------------------------------------------------- analysis


def load_records(arm: str, split: str) -> dict:
    folder = RECORDS / arm / split
    return {path.stem: json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))}


def _group(records: list[dict], key) -> dict:
    groups = {}
    for record in records:
        group = groups.setdefault(key(record), {"items": 0, "chain_pass": 0,
                                                "exact_steps": 0, "steps": 0})
        group["items"] += 1
        group["chain_pass"] += record["score"]["automated_pass"]
        group["exact_steps"] += sum(c["pass"] for c in record["score"]["automated_checks"])
        group["steps"] += record["cycles"]
    return dict(sorted(groups.items()))


def _median(values: list[float]):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def arm_summary(records: dict) -> dict:
    rows = list(records.values())
    if not rows:
        return {"items": 0}
    longest = max(row["cycles"] for row in rows)
    per_step = {}
    for step in range(1, longest + 1):
        eligible = [row for row in rows if row["cycles"] >= step]
        per_step[str(step)] = {
            "eligible": len(eligible),
            "exact": sum(row["score"]["automated_checks"][step - 1]["pass"] for row in eligible),
            "prefix_exact": sum(all(c["pass"] for c in row["score"]["automated_checks"][:step])
                                for row in eligible)}
    summary = {
        "items": len(rows), "chain_pass": sum(row["score"]["automated_pass"] for row in rows),
        "exact_steps": sum(sum(c["pass"] for c in row["score"]["automated_checks"])
                           for row in rows),
        "steps": sum(row["cycles"] for row in rows),
        "by_category": _group(rows, lambda row: row["horizon"]),
        "by_length_bucket": _group(rows, lambda row: bucket(row["cycles"])),
        "by_domain": _group(rows, lambda row: row["category"]),
        "stop_outcomes": dict(sorted(Counter(row["stop_outcome"] for row in rows).items())),
        "failure_kinds": dict(sorted(Counter(row["failure_kind"] or "none"
                                             for row in rows).items())),
        "first_failed_step": {row["item_id"]: row["first_failed_step"] for row in rows
                              if row["first_failed_step"] is not None},
        "per_step": per_step,
        "records_sha256": hashlib.sha256(json.dumps(
            {row["item_id"]: hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
             for row in rows}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    if "trace" in rows[0]:
        summary.update(_loop_summary(rows))
    else:
        gens = [row["generation"] for row in rows]
        summary.update({
            "parse_ok": sum(row["parse_ok"] for row in rows),
            "answer_parse_ok": sum(row["answer_parse_ok"] for row in rows),
            "gold_length_parse_ok": sum(row["gold_length_parse_ok"] for row in rows),
            "closed_naturally": sum(g["closed_naturally"] for g in gens),
            "forced_close": sum(g["forced_close"] for g in gens),
            "answer_hit_limit": sum(g["answer_hit_limit"] for g in gens),
            "model_tokens": sum(g["model_tokens"] for g in gens),
            "seconds": round(sum(g["seconds"] for g in gens), 1)})
    return summary


def _loop_summary(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["stop_eligible"]]
    exact = sum(row["stop_outcome"] == "exact" for row in eligible)
    margins = {"continue_due": [], "done_due": []}
    agree = {"continue_due": [0, 0], "done_due": [0, 0]}
    for row in eligible:
        for decision in row["trace"]["decisions"]:
            if decision["cycle"] > row["cycles"]:
                continue
            key = "done_due" if decision["cycle"] == row["cycles"] else "continue_due"
            margins[key].append(decision["continue_margin"])
            agree[key][0] += decision["model_pick"] == (DONE if key == "done_due" else CONTINUE)
            agree[key][1] += 1
    audits = [row["audit"] for row in rows]
    traces = [row["trace"] for row in rows]
    provenance = Counter()
    for trace in traces:
        provenance.update(trace["token_provenance"])
    return {
        "stop_eligible": len(eligible), "exact_stops": exact,
        "exact_stop_rate": round(exact / len(eligible), 6) if eligible else None,
        "stop_cycles": {row["item_id"]: [row["cycles"], row["trace"]["cycles"]] for row in rows},
        "decision_margins": {key: {"decisions": len(values), "median": _median(values),
                                   "model_pick_correct": agree[key][0]}
                             for key, values in margins.items()},
        "ledger_shapes": {category: dict(sorted(Counter(
            str(row["ledger"]["shape"]) for row in rows if row["horizon"] == category).items()))
            for category in sorted({row["horizon"] for row in rows})},
        "audit": {"pass": all(a["pass"] for a in audits),
                  "traces_with_violations": sorted(row["item_id"] for row in rows
                                                   if not row["audit"]["pass"]),
                  "violation_checks": dict(sorted(Counter(
                      v["check"] for a in audits for v in a["violations"]).items())),
                  "unfixed": sorted({check for a in audits for check in a["unfixed"]}),
                  "flags": dict(sorted(Counter(f["check"] for a in audits
                                               for f in a["flags"]).items())),
                  "max_writes_per_file": max(a["max_writes_per_file"] for a in audits),
                  "max_files": max(a["files"] for a in audits),
                  "max_stored_tokens": max(a["max_stored_tokens"] for a in audits),
                  "max_span": max(a["max_span"] for a in audits),
                  "max_cycles": max(a["cycles"] for a in audits)},
        "model_tokens": sum(trace["model_tokens"] for trace in traces),
        "token_provenance": dict(sorted(provenance.items())),
        "max_context_tokens": max(trace["context_tokens"] for trace in traces),
        "forward_seconds": round(sum(trace["forward_seconds"] for trace in traces), 1),
        "seconds": round(sum(trace["wall_seconds"] for trace in traces), 1)}


def _as_scored(verdicts: dict) -> dict:
    return {key: {"score": {"automated_pass": bool(value)}} for key, value in verdicts.items()}


def paired(after: dict, before: dict, label: str) -> dict:
    """after minus before over the same items (qfs.end_to_end._paired_summary)."""
    stats = module("qfs.end_to_end")._paired_summary(_as_scored(before), _as_scored(after))
    return {"difference": label, "items": stats["items"],
            "after_rate": stats["host_chain_accuracy"],
            "before_rate": stats["plain_chain_accuracy"], "delta": stats["delta"],
            "after_only": stats["host_improvements"], "before_only": stats["host_regressions"],
            "exact_mcnemar_p": stats["exact_mcnemar_p"],
            "item_bootstrap_delta_95": stats["item_bootstrap_delta_95"],
            "reading": port.verdict(stats["item_bootstrap_delta_95"])}


def decision_rule(adapt: dict, oracle: dict, audit_ok: bool, stop_rate) -> dict:
    """The preregistered screen decision (design section 4; McNemar p takes no part).

    GO when the H-adapt minus P-F2 CI lower bound is above 0, the audit passes (or every
    failure has a preregistered fix), and H-adapt stops exactly at L on at least 90% of
    stop-eligible items. Otherwise REDIRECT when the H-oracle minus P-F2 lower bound is
    above 0; NO-GO when that difference's upper bound is below 0 or both point estimates
    are <= 0; INCONCLUSIVE otherwise.
    """
    conditions = {
        "adapt_ci_lower_above_0": adapt["item_bootstrap_delta_95"][0] > 0,
        "audit_ok": bool(audit_ok),
        "exact_stop_rate_at_least_0.9": stop_rate is not None and stop_rate >= STOP_TARGET,
        "oracle_ci_lower_above_0": oracle["item_bootstrap_delta_95"][0] > 0,
        "oracle_ci_upper_below_0": oracle["item_bootstrap_delta_95"][1] < 0,
        "both_deltas_at_most_0": adapt["delta"] <= 0 and oracle["delta"] <= 0,
    }
    if (conditions["adapt_ci_lower_above_0"] and conditions["audit_ok"]
            and conditions["exact_stop_rate_at_least_0.9"]):
        decision = "GO"
    elif conditions["oracle_ci_lower_above_0"]:
        decision = "REDIRECT"
    elif conditions["oracle_ci_upper_below_0"] or conditions["both_deltas_at_most_0"]:
        decision = "NO-GO"
    else:
        decision = "INCONCLUSIVE"
    return {"decision": decision, "conditions": conditions, "exact_stop_rate": stop_rate}


def _check_identities(arm: str, split: str, records: dict, items: list[dict],
                      suite_hash: str) -> list[str]:
    """Every record of an arm was made by one runner, with this ROM, suite and settings."""
    problems = []
    runners = {row["identity"]["runner_sha256"] for row in records.values()}
    if len(runners) > 1:
        problems.append(f"{arm}/{split}: records come from {len(runners)} runner versions")
    for item in items:
        row = records.get(item["id"])
        if row is None:
            problems.append(f"{arm}/{split}: no record for {item['id']}")
        elif row["identity"] != identity(arm, split, item, suite_hash,
                                         row["identity"]["runner_sha256"]):
            problems.append(f"{arm}/{split}: {item['id']} identity differs from this protocol")
    extra = sorted(set(records) - {item["id"] for item in items})
    problems += [f"{arm}/{split}: {key} is not a suite item" for key in extra]
    return problems


def _by(records: dict, key) -> dict:
    groups = {}
    for item_id, row in records.items():
        groups.setdefault(key(row), {})[item_id] = row["score"]["automated_pass"]
    return groups


def compare(after: dict, before: dict, label: str) -> dict:
    """A paired comparison plus its per-category and per-length-bucket deltas."""
    a = {key: row["score"]["automated_pass"] for key, row in after.items()}
    b = {key: row["score"]["automated_pass"] for key, row in before.items()}
    out = paired(a, b, label)
    for name, key in (("by_category", lambda row: row["horizon"]),
                      ("by_length_bucket", lambda row: bucket(row["cycles"]))):
        groups_a, groups_b = _by(after, key), _by(before, key)
        out[name] = {group: {"items": len(groups_a[group]), "after": sum(groups_a[group].values()),
                             "before": sum(groups_b[group].values()),
                             "delta": sum(groups_a[group].values())
                                      - sum(groups_b[group].values())}
                     for group in sorted(groups_a)}
    return out


def _v3_references() -> dict:
    """Frozen v3 comparisons for H-adapt-v3: E12 compiled 4B, E9 host 4B, E14 F2-4B if present."""
    refs = {}
    compiled = ROOT / "provenance" / "e14-compiled.json"
    if compiled.exists():
        rows = json.loads(compiled.read_text())["4b"]["records"]
        refs["E12 compiled 4B"] = {key: row["chain_pass"] for key, row in rows.items()}
    frozen = ROOT / "provenance" / "e12-frozen-comparisons.json"
    if frozen.exists():
        rows = json.loads(frozen.read_text())["host_4b"]["records"]
        refs["E9 host 4B (fixed-length ledger pointer)"] = {
            key: row["chain_pass"] for key, row in rows.items()}
    f2 = plain_forced.RECORDS / "4b" / "F2"
    rows = {path.stem: json.loads(path.read_text()) for path in sorted(f2.glob("*.json"))}
    if len(rows) == 32:
        refs["E14 F2 plain 4B"] = {key: row["score"]["automated_pass"] for key, row in rows.items()}
    return refs


def analyze(output: Path) -> dict:
    items, suite_hash = load_items("H-adapt", "test")
    problems, records = [], {}
    for arm in ("H-adapt", "H-oracle", "P-F2"):
        records[arm] = load_records(arm, "test")
        problems += _check_identities(arm, "test", records[arm], items, suite_hash)
    v3_items, v3_hash = load_items("H-adapt-v3", "test")
    v3 = load_records("H-adapt-v3", "test")
    v3_problems = _check_identities("H-adapt-v3", "test", v3, v3_items, v3_hash)
    result = {"items": len(items), "suite_sha256": suite_hash, "rom_sha256": rom_sha256(),
              "complete": not problems, "problems": problems,
              "arms": {arm: arm_summary(rows) for arm, rows in records.items()}}
    if problems:
        result["decision"] = {"decision": "INCONCLUSIVE",
                              "reason": "an arm is incomplete or off-protocol"}
    else:
        adapt, oracle, plain = records["H-adapt"], records["H-oracle"], records["P-F2"]
        comparisons = {
            "adapt_minus_plain": compare(adapt, plain, "H-adapt minus P-F2"),
            "oracle_minus_plain": compare(oracle, plain, "H-oracle minus P-F2"),
            "adapt_minus_oracle": compare(adapt, oracle, "H-adapt minus H-oracle")}
        loop = result["arms"]["H-adapt"]
        audit_ok = not loop["audit"]["unfixed"]
        result.update({
            "comparisons": comparisons,
            "adapt_minus_plain_delta": comparisons["adapt_minus_plain"]["delta"],
            "audit_ok": audit_ok,
            "decision": decision_rule(comparisons["adapt_minus_plain"],
                                      comparisons["oracle_minus_plain"], audit_ok,
                                      loop["exact_stop_rate"])})
    secondary = {"complete": not v3_problems, "problems": v3_problems,
                 "summary": arm_summary(v3)}
    if not v3_problems:
        mine = {key: row["score"]["automated_pass"] for key, row in v3.items()}
        secondary["paired"] = {name: paired(mine, ref, f"H-adapt-v3 minus {name}")
                               for name, ref in _v3_references().items() if set(ref) == set(mine)}
    result["H-adapt-v3"] = secondary
    return result


# ---------------------------------------------------------------- smoke


def refuse_suite_objective(objective: str) -> None:
    """Pre-lock smoke tests may decode only objectives that are not suite items (as E14)."""
    for arm, split in (("H-adapt", "test"), ("H-adapt", "dev"), ("H-adapt-v3", "test")):
        if any(item["objective"].strip() == objective.strip()
               for item in load_items(arm, split)[0]):
            raise ValueError(f"that objective is a {arm} {split} suite item")


def smoke(arm: str, objective: str, length: int | None, output: Path) -> dict:
    """One free-form objective through one arm, for GPU checks before the lock (not evidence)."""
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if arm == "H-adapt-v3":
        raise ValueError("smoke runs H-adapt, H-oracle or P-F2")
    if arm != "H-adapt" and not (length and 1 <= length <= CAP):
        raise ValueError(f"{arm} needs --length L in 1..{CAP}")
    refuse_suite_objective(objective)
    tok = AutoTokenizer.from_pretrained(ROOT / BASE)
    static = static_check(tok)
    if not static["pass"]:
        return {"lab_status": "invalid", "invalid_reason": "static check failed",
                "static_check": static}
    model = Qwen3_5ForCausalLM.from_pretrained(ROOT / BASE, dtype=torch.bfloat16).to(
        DEVICE).eval()
    if arm == "P-F2":
        generation = plain_forced.generate_forced(model, tok, objective + PLAIN_SUFFIX,
                                                  *caps(length))
        port._write(output / "smoke.json", {"arm": arm, "objective": objective,
                                            "length": length, "generation": generation})
        return {"arm": arm, "results": parse_open(generation["text"]),
                **{key: generation[key] for key in ("thinking_tokens", "answer_tokens",
                                                     "forced_close", "seconds")}}
    runner = module("qfs.agent_native").NativeRunner(model, tok, DEVICE, temperature=0.0,
                                                     top_p=0.9, seed=0)
    trace = run_loop(runner, tok, objective,
                     oracle_length=length if arm == "H-oracle" else None)
    report = audit(trace)
    port._write(output / "smoke.json", {"arm": arm, "objective": objective, "length": length,
                                        "trace": trace, "audit": report})
    return {"arm": arm, "cycles": trace["cycles"], "stop": trace["stop"],
            "results": trace["results"], "decisions": [d["token"] for d in trace["decisions"]],
            "ledger": ledger_shape(trace), "audit_pass": report["pass"],
            "audit_flags": report["flags"], "model_tokens": trace["model_tokens"],
            "context_tokens": trace["context_tokens"],
            "seconds": round(trace["wall_seconds"], 1)}


# ---------------------------------------------------------------- main


def check(output: Path | None) -> dict:
    from transformers import AutoTokenizer

    return static_check(AutoTokenizer.from_pretrained(ROOT / BASE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=os.environ.get("LAB_OUTPUT_DIR"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    runner = commands.add_parser("run")
    runner.add_argument("arm", choices=ARMS)
    runner.add_argument("split", choices=SPLITS)
    runner.add_argument("--only", action="append", help="dev item id (repeatable)")
    commands.add_parser("analyze")
    smoker = commands.add_parser("smoke", help="a free-form objective (not a suite item)")
    smoker.add_argument("arm", choices=("H-adapt", "H-oracle", "P-F2"))
    smoker.add_argument("objective")
    smoker.add_argument("--length", type=int, help="L, for H-oracle and P-F2")
    args = parser.parse_args()
    if args.command == "check":
        result = check(args.output)
        print(json.dumps(result, indent=2))
        if args.output is not None:
            port._write(port._output(args) / "metrics.json", result)
        return
    output = port._output(args)
    if args.command == "smoke":
        metrics = smoke(args.arm, args.objective, args.length, output)
    elif args.command == "run":
        metrics = run(args.arm, args.split, output, args.only)
    else:
        metrics = analyze(output)
    port._write(output / "metrics.json", metrics)
    print(json.dumps({key: value for key, value in metrics.items()
                      if key not in ("first_failed_step", "per_step", "stop_cycles", "arms")},
                     indent=2))


if __name__ == "__main__":
    main()
