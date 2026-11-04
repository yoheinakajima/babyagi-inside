"""A bounded BabyAGI microprogram compiled into qwenFS's existing layers.

The host supplies one objective and START, then samples tokens. Gated DeltaNet
registers hold the program counter; an MLP ROM prints the phase instructions.
Qwen creates/prioritizes/executes tasks and qwenFS answers its file operations.
No Python callback advances a phase or supplies a subsequent prompt.

This smallest experiment uses the spare residual dimensions of Qwen3.5-0.8B.
The semantic policy is pretrained Qwen, not a hand-constructed reasoning system.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from qfs import circuit as F
from qfs import merge as M

PHASE0 = F.V_QWEN
HALT = PHASE0 + 32
START = PHASE0
KINDS = ("create", "prioritize", "execute")
PROTOCOLS = (
    "baseline", "literal", "generative", "babyagi", "babyagi_kshot", "babyagi_json",
    "babyagi_typed", "babyagi_pointer", "babyagi_ledger_pointer",
)
CHOICE_GAIN = 65536.0
NATIVE_VOCAB_STOP = 248044
POINTER_THINK_TOKENS = 128


def phase_prompts(cycle, cycles, protocol):
    """Return static semantic instructions; never include a runtime objective."""
    if protocol == "baseline":
        shared = f"Cycle {cycle}/{cycles}. Work toward the original objective. "
        instructions = (
            "CREATE: Write two short, concrete next tasks for the objective. "
            + ("Use the previous result just read; avoid completed work. " if cycle > 1 else "")
            + "Write task descriptions only, not their answers.",
            "PRIORITIZE: Choose the most important next task from the task list just read. "
            "Write only its description. Do not perform it yet.",
            "EXECUTE: Perform the selected task just read. Write its actual result, "
            "not a promise or a plan. State input and output when applicable.",
        )
    elif protocol == "babyagi_typed":
        shared = (
            f"Cycle {cycle}/{cycles}. Follow the objective in the first user message literally. "
            "Treat its exact words, strings, symbols, numbers, order, and requested format as "
            "requirements. Do not reinterpret its subject as a business, brand, or fictional "
            "project. Do not invent external actions or facts. "
        )
        if cycle == 1:
            create1 = (
                "INITIAL TASK — Create the first task. Convert the actual objective into exactly "
                "one first executable task. For an ordered objective, combine its initial state "
                "with only its earliest operation. For an open goal, choose one concrete action "
                "that produces part of the requested deliverable. Valid one-task shapes using "
                "fake values are `Append -hot to elm`, `Double 4`, and `Draft the launch note "
                "headline`. They show brevity and format only; never copy their content. The file "
                "already begins `FIRST TASK: `. Add the actual task in the same short one-line "
                "shape, at most 12 words, then close the tool call. Do not add a list, explanation, "
                "or answer."
            )
        else:
            create1 = (
                "CREATE TASK 1. The previous result was just read; its Output is the current state. "
                "Write exactly one concrete executable task: the earliest explicit operation not "
                "yet reflected in that state. For a transformation, name the complete current "
                "value and only the next requested change. For an open goal, name one action that "
                "adds to the deliverable. Preserve the original objective's exact values; do not "
                "invent a value or action. The file already begins `TASK 1: `. Complete that one "
                "line in at most 12 words, then "
                "close the tool call. Do not execute the task."
            )
        create2 = (
            "CREATE TASK 2. Task 1 was just generated. Write a verification task that checks Task "
            "1's result without performing another objective operation. Do not introduce any new "
            "literal, calculation, transformation, or deliverable content. The file already begins "
            "`TASK 2: `. Complete that one line in at most 12 words, then close the tool call. Do "
            "not repeat, explain, or execute Task 1."
        )
        instructions = (
            create1,
            create2,
            "PRIORITIZE: Choose the candidate that should be executed now. Select Task 1 or Task 2. "
            "The selected file will be read automatically; emit only its single-digit ID.",
            "EXECUTE: The selected task was just read and is the sole instruction for this phase. "
            "Ignore every other candidate and every later operation in the objective. Do not copy "
            "the task. Perform its one named action and stop. For a text transformation, use the complete value "
            "named in the task and apply only its requested change. For text creation, write the "
            "requested text itself, not a claim that it was written. The file already begins "
            "`Output: `. Complete that one line with only the actual completed result, then close "
            "the tool call. Do not repeat the task description or add an explanation."
        )
    elif protocol in ("literal", "generative", "babyagi", "babyagi_kshot", "babyagi_json"):
        shared = (
            f"Cycle {cycle}/{cycles}. Follow the objective in the first user message literally. "
            "Treat its exact words, strings, symbols, numbers, order, and requested format as "
            "requirements. Do not reinterpret its subject as a business, brand, or fictional "
            "project. Do not invent external actions or facts. "
        )
        if protocol == "babyagi_json":
            shared = (
                f"Cycle {cycle}/{cycles}. Obey the original objective literally. Preserve its "
                "values, order, and format. Do not invent external facts or actions. "
            )
            schema = (
                'The file is one-line JSON with exactly these fields: '
                '`pending` (array of objects with string `id` and string `task`), '
                '`completed` (array of finished task strings), and `result` (actual current '
                'result or null). Emit valid JSON only and escape newlines in strings.'
            )
            if cycle == 1:
                create = (
                    "INITIAL TASK — Create the first task. Convert the objective into exactly one "
                    "concrete first executable task. For ordered work, combine its initial state "
                    "with only the earliest operation. For an open goal, choose one action that "
                    "produces part of the requested deliverable. Put only that task in `pending` "
                    "with ID `T1`; set `result` to null and `completed` to an empty array. For a "
                    "transformation, the task string must name the initial value, action, and "
                    "operand; a string containing only the value or only the action is invalid. "
                    "Valid task-string shapes using unrelated fake values are `Append -hot to elm`, "
                    "`Double 4`, and `Sketch a garden layout`. They show brevity and format only; "
                    "never copy their content. The file already begins `{`; complete the JSON "
                    "object without another opening brace. " + schema
                )
            else:
                create = (
                    "CREATE: Read the ledger. Completed tasks are finished and must never be "
                    "repeated. Treat `result` as the current state. Preserve `completed` and "
                    "`result`, and replace `pending` with at most two concrete unfinished tasks "
                    "identified as `T1` then `T2`, ranked with the best next task as `T1`. For "
                    "ordered work, `T1` is the earliest unfinished operation. For an open goal, "
                    "propose work that adds to or improves the requested deliverable. Do not "
                    "perform a task. The file text already begins "
                    "`{`; complete the object without another opening brace. " + schema
                )
            instructions = (
                create,
                "PRIORITIZE: Read the ranked ledger. Its best next task is `T1`. The file text "
                "already contains the complete choice `T1`; close the tool call immediately. Do "
                "not add task text, JSON, punctuation, or an explanation. Do not perform the task.",
                "EXECUTE: Read the chosen ID. Find its task in the most recently read ledger and "
                "perform exactly that task. Remove its object from `pending`, append its task string "
                "to `completed`, and write the actual result—not the task description or a promise—"
                "into `result`. After execution `result` must not be null. Preserve every other "
                "ledger value exactly; never invent or rename a pending task. Execution examples "
                "using fake values: `Append -hot to elm` produces `elm-hot`; `Double 4` produces "
                "`8`. Never copy example values. For transformations, "
                "apply the requested change to the prior `result`; use the objective's initial state "
                "only when the prior result is null. For text creation, write the requested text. "
                "The file text already begins `{`; complete the object without another opening "
                "brace. " + schema,
            )
        elif protocol in ("literal", "babyagi", "babyagi_kshot"):
            create = (
                "CREATE: Identify the explicit operations still unfinished. Write exactly two "
                "candidate tasks, one per line. Task 1 must be the earliest unfinished operation. "
                "Task 2 must be the next unfinished operation, or a final verification if only one "
                "operation remains. Each task must describe only one operation and preserve all "
                "literal values. Do not perform either task. Use exactly this format:\n"
                "1. task\n2. task"
            )
            if protocol == "babyagi" and cycle == 1:
                create = (
                    "INITIAL TASK — Create the first task. Perform that meta-task now: inspect the "
                    "objective and write exactly one concrete first executable task, the earliest "
                    "explicit operation. Preserve every literal value and requested format. Write "
                    "only the resulting executable task. The file text already begins 'FIRST TASK:'. "
                    "Complete that one line in at most 15 words, then close the tool call immediately. "
                    "Do not make a list, add a newline, or execute the resulting task yet."
                )
            elif protocol == "babyagi_kshot" and cycle == 1:
                create = (
                    "INITIAL TASK — Create the first task. Convert the actual objective into exactly "
                    "one first executable task. For an ordered objective, combine its initial state "
                    "with only its earliest operation. For an open goal, choose one concrete action "
                    "that produces part of the requested deliverable. Valid one-task shapes using "
                    "fake values: `Append -hot to elm`; `Double 4`; `Draft the launch note headline`. "
                    "These examples show brevity and format only; never copy their content. The file "
                    "already begins `FIRST TASK:`. Add the actual task in the same short one-line "
                    "shape, at most 12 words, and close the tool call. Never restate the objective or "
                    "mention later operations. Do not add a number, label, list, semicolon, newline, "
                    "explanation, or answer."
                )
            elif protocol == "babyagi":
                create = (
                    "BOOTSTRAP COMPLETE: The task labeled FIRST TASK was already executed. Never "
                    "copy, repeat, or select it again. Use the previous result just read; its Output "
                    "is the current state. " + create
                )
            elif protocol == "babyagi_kshot":
                create = (
                    "BOOTSTRAP COMPLETE: The task labeled FIRST TASK was already executed. Never "
                    "copy, repeat, or select it again. Use the previous result just read; its Output "
                    "is the current state. If the objective has ordered operations, preserve them "
                    "literally and make Task 1 the earliest unfinished operation. If the objective "
                    "has no ordered operations, propose two concrete tasks that produce or improve "
                    "the requested deliverable using the current state. Do not claim external work. "
                    "Write exactly two candidates, one per line, as `1. task` and `2. task`. Do not "
                    "perform them."
                )
            execute = (
                "EXECUTE: Perform exactly the one selected operation. For a text transformation, "
                "copy the prior value exactly and apply only the requested change. For text creation, "
                "write the requested text itself, not a claim that it was written. Use exactly two "
                "plain-text lines:\nInput: actual input\nOutput: actual output"
            )
            if protocol == "babyagi_kshot":
                execute += (
                    ". When building a multi-part deliverable, retain useful prior output and extend "
                    "or refine it only as the selected task requires"
                )
            instructions = (
                create,
                "PRIORITIZE: Select the earliest unfinished operation from the task list. Copy only "
                "that one task without its number. Do not copy the other task, explain, or perform it.",
                execute,
            )
        else:
            instructions = (
                "CREATE: Decompose the objective into its explicit incremental operations. The "
                "overall goal, initial state, and constraints are context, not tasks. Never copy the "
                "whole objective as one task. If it says First, Next, and Finally, each named clause "
                "is a separate operation. Write at most the next two unfinished operations and stop; "
                "write only one if only one remains. Task 1 must be the earliest unfinished operation. "
                "Renumber candidates from Task 1 in every cycle; do not retain an original step number. "
                "Each task must produce only one incremental result and preserve all literal values. "
                "Do not perform a task. Use one line per task: Task 1: operation; Task 2: operation.",
                "PRIORITIZE: Select the earliest unfinished operation from the task list. Write only "
                "the operation text after Task 1:. Do not copy a label or another task, explain, or "
                "perform it.",
                "EXECUTE: Perform exactly the one selected operation. For a text transformation, "
                "copy the prior value exactly and apply only the requested change. For text creation, "
                "write the requested text itself, not a claim that it was written. Use exactly two "
                "plain-text lines:\nInput: actual input\nOutput: actual output\n"
                + ("The previous result was just read. Input must be its exact Output, and Output "
                   "must retain it unchanged before adding only the new content. " if cycle > 1 else
                   "There is no previous result. Use the objective's explicit initial state as Input. ")
                + "End after Output. Do not add explanations or another tool parameter.",
            )
    else:
        raise ValueError(f"Unknown protocol {protocol!r}; choose from {PROTOCOLS}")
    return shared, instructions


def program(tok, cycles=3, protocol="baseline", think_tokens=None):
    """Only static instructions are compiled; objectives/tasks are never baked in."""
    if cycles < 1:
        raise ValueError("cycles must be positive")
    if think_tokens is not None and protocol not in (
            "babyagi_pointer", "babyagi_ledger_pointer"):
        raise ValueError("think_tokens is only supported by pointer protocols")
    bounded_tokens = POINTER_THINK_TOKENS if think_tokens is None else think_tokens
    if protocol == "babyagi_pointer":
        return pointer_program(tok, cycles, bounded_tokens)
    if protocol == "babyagi_ledger_pointer":
        return ledger_pointer_program(tok, cycles, bounded_tokens)
    stages = []
    for cycle in range(1, cycles + 1):
        shared, instructions = phase_prompts(cycle, cycles, protocol)
        if protocol == "babyagi_typed":
            create1, create2, prioritize, execute = instructions
            if cycle > 1:
                read = ("<tool_call>\n<function=read_file>\n<parameter=name>\nresult.txt"
                        "\n</parameter>\n</function>\n</tool_call>")
                stages.append({"cycle": cycle, "kind": "create", "op": "read",
                               "file": "result.txt",
                               "rom": tok.encode(read, add_special_tokens=False)})

            for candidate, instruction in enumerate((create1, create2), 1):
                prefix = "<|im_end|>" if stages else ""
                frame = ("\n<|im_start|>user\n" + shared + instruction
                         + " Keep it under 30 words. Finish by closing the tool call."
                         + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
                filename = f"task{candidate}.txt"
                text_prefix = ("FIRST TASK: " if candidate == 1 and cycle == 1 else
                               f"TASK {candidate}: ")
                write = ("<tool_call>\n<function=write_file>\n<parameter=name>\n" + filename
                         + "\n</parameter>\n<parameter=text>\n" + text_prefix)
                stages.append({"cycle": cycle, "kind": "create", "candidate": candidate,
                               "op": "write", "file": filename, "text_prefix": text_prefix,
                               "rom": tok.encode(prefix + frame + write,
                                                 add_special_tokens=False)})

            choices = [tok.encode(value, add_special_tokens=False) for value in ("1", "2")]
            if any(len(choice) != 1 for choice in choices):
                raise ValueError("typed choices 1 and 2 must each be one tokenizer token")
            prefix = "<|im_end|>" if stages else ""
            frame = ("\n<|im_start|>user\n" + shared + prioritize
                     + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            read_prefix = ("<tool_call>\n<function=read_file>\n<parameter=name>\ntask")
            stages.append({
                "cycle": cycle, "kind": "prioritize", "op": "read",
                "file": "task{1|2}.txt", "selected_files": ["task1.txt", "task2.txt"],
                "rom": tok.encode(prefix + frame + read_prefix, add_special_tokens=False),
                "decision": {
                    "mode": "typed_file_choice",
                    "allowed_token_ids": [choice[0] for choice in choices],
                    "allowed_text": ["1", "2"], "token_count": 1, "owner": "model",
                },
                "rom_suffix": tok.encode(
                    ".txt\n</parameter>\n</function>\n</tool_call>",
                    add_special_tokens=False),
            })

            prefix = "<|im_end|>" if stages else ""
            frame = ("\n<|im_start|>user\n" + shared + execute
                     + " Keep it under 60 words. Finish by closing the tool call."
                     + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            write = ("<tool_call>\n<function=write_file>\n<parameter=name>\nresult.txt"
                     "\n</parameter>\n<parameter=text>\nOutput: ")
            stages.append({"cycle": cycle, "kind": "execute", "op": "write",
                           "file": "result.txt", "text_prefix": "Output: ",
                           "rom": tok.encode(prefix + frame + write,
                                             add_special_tokens=False)})
            continue
        if protocol == "babyagi_json":
            write_names = ("tasks.json", "choice.txt", "tasks.json")
            read_names = ("tasks.json" if cycle > 1 else None, "tasks.json", "choice.txt")
        else:
            write_names = ("tasks.txt", "current.txt", "result.txt")
            read_names = ("result.txt" if cycle > 1 else None, "tasks.txt", "current.txt")
        for kind, filename, read_name, instruction in zip(
                KINDS, write_names, read_names, instructions):
            if read_name:
                read = ("<tool_call>\n<function=read_file>\n<parameter=name>\n" + read_name +
                        "\n</parameter>\n</function>\n</tool_call>")
                stages.append({"cycle": cycle, "kind": kind, "op": "read", "file": read_name,
                               "rom": tok.encode(read, add_special_tokens=False)})
            if protocol == "generative" and kind == "execute" and cycle > 1:
                # Refresh the accumulated state immediately before semantic execution. Small
                # models otherwise tend to replace a creative result instead of extending it.
                read = ("<tool_call>\n<function=read_file>\n<parameter=name>\nresult.txt"
                        "\n</parameter>\n</function>\n</tool_call>")
                stages.append({"cycle": cycle, "kind": kind, "op": "read", "file": "result.txt",
                               "rom": tok.encode(read, add_special_tokens=False)})
            prefix = "<|im_end|>" if stages else ""
            finish = (" Keep it under 60 words." if
                      protocol == "babyagi_typed" and kind == "prioritize" else
                      " Keep it under 60 words. Finish by closing the tool call.")
            frame = ("\n<|im_start|>user\n" + shared + instruction + finish +
                     "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            if protocol == "babyagi_json":
                text_prefix = "T1" if kind == "prioritize" else "{"
            else:
                text_prefix = ("FIRST TASK: " if protocol in ("babyagi", "babyagi_kshot")
                               and cycle == 1 and kind == "create" else "")
            write = ("<tool_call>\n<function=write_file>\n<parameter=name>\n" + filename +
                     "\n</parameter>\n<parameter=text>\n" + text_prefix)
            stage = {"cycle": cycle, "kind": kind, "op": "write", "file": filename,
                     "text_prefix": text_prefix,
                     "rom": tok.encode(prefix + frame + write, add_special_tokens=False)}
            stages.append(stage)
    phase_budget = HALT - PHASE0
    if len(stages) > phase_budget:
        raise ValueError(
            f"{protocol} protocol needs {len(stages)} microstates for {cycles} cycles; "
            f"the compiled controller reserves {phase_budget} before HALT")
    return stages


def pointer_program(tok, cycles=3, think_tokens=POINTER_THINK_TOKENS):
    """An ordered qwenFS task ledger with an unrolled deterministic pointer.

    Qwen extracts the complete queue into separate files before execution.  The
    controller then reads ``task1.txt``, ``task2.txt``, ... in order.  Execute
    owns exactly ``think_tokens`` ordinary vocabulary tokens before ROM resumes,
    closes the think span, and opens the result write.  No host scheduler or
    semantic selector participates.
    """
    if cycles < 1:
        raise ValueError("cycles must be positive")
    if think_tokens < 1:
        raise ValueError("think_tokens must be positive")
    stages = []

    # Materialize the ledger first.  Separate files make the pointer operation
    # token-observable without asking the compiled circuit to parse JSON.
    for cycle in range(1, cycles + 1):
        homage = (
            "INITIAL TASK — Create the first task. " if cycle == 1 else
            f"CREATE TASK {cycle}. "
        )
        instruction = (
            homage
            + f"Extract operation {cycle} of {cycles} from the objective in the first user "
              "message. Treat Begin, Start, initial values, and supplied records as input, not "
              "operations. Preserve every literal value, symbol, order constraint, and requested "
              "format. Write exactly one short executable task for this operation. Do not perform "
              "it, explain it, combine it with another operation, or repeat an earlier task. For "
              "an objective without numbered operations, choose the next distinct action that adds "
              "to the requested deliverable. The file already begins with its TASK label; finish "
              "that one line in at most 24 words and close the tool call."
        )
        prefix = "<|im_end|>" if stages else ""
        frame = (
            "\n<|im_start|>user\n" + instruction
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        filename = f"task{cycle}.txt"
        text_prefix = f"TASK {cycle}: "
        write = (
            "<tool_call>\n<function=write_file>\n<parameter=name>\n" + filename
            + "\n</parameter>\n<parameter=text>\n" + text_prefix
        )
        stages.append({
            "cycle": cycle, "kind": "create", "op": "write", "file": filename,
            "text_prefix": text_prefix, "ledger_index": cycle,
            "semantic_owner": "model_free_text",
            "rom": tok.encode(prefix + frame + write, add_special_tokens=False),
        })

    for cycle in range(1, cycles + 1):
        if cycle > 1:
            read_result = (
                "<tool_call>\n<function=read_file>\n<parameter=name>\nresult.json"
                "\n</parameter>\n</function>\n</tool_call>"
            )
            stages.append({
                "cycle": cycle, "kind": "execute", "op": "read", "file": "result.json",
                "pointer_index": cycle, "semantic_owner": "deterministic_circuit_content",
                "rom": tok.encode(read_result, add_special_tokens=False),
            })

        filename = f"task{cycle}.txt"
        read_task = (
            "<tool_call>\n<function=read_file>\n<parameter=name>\n" + filename
            + "\n</parameter>\n</function>\n</tool_call>"
        )
        stages.append({
            "cycle": cycle, "kind": "prioritize", "op": "read", "file": filename,
            "pointer_index": cycle, "selection": "deterministic_ledger_pointer",
            "semantic_owner": "deterministic_circuit_choice",
            "rom": tok.encode(read_task, add_special_tokens=False),
        })

        instruction = (
            f"EXECUTE TASK {cycle}/{cycles}. The task file just read is the sole instruction for "
            "this cycle. Perform exactly that task and no later task. "
            + ("There is no prior result; use the objective's supplied initial state. "
               if cycle == 1 else
               "The result file read immediately before the task is the current state; preserve it "
               "except for the requested operation. ")
            + "Use the private think span to work through the operation. When the controller opens "
              "the result object, emit one valid JSON value for the complete new state, then `}`, "
              "then close the text parameter, function, and tool call. Do not add commentary."
        )
        prefix = "<|im_end|>" if stages else ""
        frame = (
            "\n<|im_start|>user\n" + instruction
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
        suffix = (
            "\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=name>\nresult.json"
            "\n</parameter>\n<parameter=text>\n{\"result\":"
        )
        stages.append({
            "cycle": cycle, "kind": "execute", "op": "write", "file": "result.json",
            "text_prefix": '{"result":', "pointer_index": cycle,
            "semantic_owner": "model_free_text", "free_span": think_tokens,
            "free_vocab_stop": NATIVE_VOCAB_STOP,
            "rom": tok.encode(prefix + frame, add_special_tokens=False),
            "rom_suffix": tok.encode(suffix, add_special_tokens=False),
        })

    phase_budget = HALT - PHASE0
    if len(stages) > phase_budget:
        raise ValueError(
            f"babyagi_pointer protocol needs {len(stages)} microstates for {cycles} cycles; "
            f"the compiled controller reserves {phase_budget} before HALT")
    return stages


def ledger_pointer_program(tok, cycles=3, think_tokens=POINTER_THINK_TOKENS):
    """One model-written JSON ledger, indexed by a compiled cycle pointer.

    The controller never parses the ledger and the host never sees it.  qwenFS
    returns the complete ledger to Qwen in each cycle; the static cycle index in
    the compiled instruction identifies the sole task to execute.
    """
    if cycles < 1:
        raise ValueError("cycles must be positive")
    if think_tokens < 1:
        raise ValueError("think_tokens must be positive")
    stages = []
    instruction = (
        "INITIAL TASK — Create the first task, then the complete ordered task ledger. Separate "
        "the objective's supplied input state from its executable work. `initial` is the exact "
        "value supplied by Begin with, Start with, or the initial JSON data; use null if there is "
        "no single input value. Input data is never a task. `tasks` contains only executable "
        "operation clauses. For an ordered objective, transcribe clauses introduced by First, "
        f"Next, Then, and Finally into exactly {cycles} short tasks in that order. Do not add an "
        "initialization task and do not calculate any result. Every task must preserve its clause's "
        "operation verb and operand exactly: add stays add, subtract stays subtract, multiply stays "
        "multiply, and so on. Once `initial` is separated, a task must never say Start, Begin, Set, "
        "or Initialize; it applies its operation to the current result. For an open objective, propose "
        f"exactly {cycles} distinct atomic actions that build the requested deliverable. Preserve "
        "all literals and constraints. The file already begins `{`; continue with exactly "
        f"`\"initial\":...,\"tasks\":[...]}}`, using exactly {cycles} non-empty task strings, "
        "then close the text parameter, function, and tool call. Emit no explanation."
    )
    frame = (
        "\n<|im_start|>user\n" + instruction
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    write = (
        "<tool_call>\n<function=write_file>\n<parameter=name>\nledger.json"
        "\n</parameter>\n<parameter=text>\n{"
    )
    stages.append({
        "cycle": 1, "kind": "create", "op": "write", "file": "ledger.json",
        "text_prefix": "{", "semantic_owner": "model_free_text",
        "rom": tok.encode(frame + write, add_special_tokens=False),
    })

    for cycle in range(1, cycles + 1):
        if cycle > 1:
            read_result = (
                "<tool_call>\n<function=read_file>\n<parameter=name>\nresult.json"
                "\n</parameter>\n</function>\n</tool_call>"
            )
            stages.append({
                "cycle": cycle, "kind": "create", "op": "read", "file": "result.json",
                "pointer_index": cycle, "semantic_owner": "deterministic_circuit_content",
                "rom": tok.encode(read_result, add_special_tokens=False),
            })

        read_ledger = (
            "<tool_call>\n<function=read_file>\n<parameter=name>\nledger.json"
            "\n</parameter>\n</function>\n</tool_call>"
        )
        stages.append({
            "cycle": cycle, "kind": "prioritize", "op": "read", "file": "ledger.json",
            "pointer_index": cycle, "selection": "deterministic_ledger_pointer",
            "semantic_owner": "deterministic_circuit_choice",
            "rom": tok.encode(read_ledger, add_special_tokens=False),
        })

        instruction = (
            f"EXECUTE LEDGER TASK {cycle}/{cycles}. The ledger file just read contains `initial` "
            f"and an ordered `tasks` array. Use exactly array entry {cycle - 1} (the {cycle}th "
            "task); this index is fixed by the controller, not a choice. Perform that task and no "
            "other task. "
            + ("Use the ledger's `initial` value as the incoming state. " if cycle == 1 else
               "Use the result file read immediately before the ledger as the incoming state. ")
            + "Use the private think span to work through the operation. When the controller opens "
              "the result object, emit one valid JSON value for the complete new state, then `}`, "
              "then close the text parameter, function, and tool call. Do not add commentary."
        )
        frame = (
            "<|im_end|>\n<|im_start|>user\n" + instruction
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
        suffix = (
            "\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=name>\nresult.json"
            "\n</parameter>\n<parameter=text>\n{\"result\":"
        )
        stages.append({
            "cycle": cycle, "kind": "execute", "op": "write", "file": "result.json",
            "text_prefix": '{"result":', "pointer_index": cycle,
            "semantic_owner": "model_free_text", "free_span": think_tokens,
            "free_vocab_stop": NATIVE_VOCAB_STOP,
            "rom": tok.encode(frame, add_special_tokens=False),
            "rom_suffix": tok.encode(suffix, add_special_tokens=False),
        })

    phase_budget = HALT - PHASE0
    if len(stages) > phase_budget:
        raise ValueError(
            f"babyagi_ledger_pointer protocol needs {len(stages)} microstates for {cycles} cycles; "
            f"the compiled controller reserves {phase_budget} before HALT")
    return stages


class Layout:
    """Controller scratch, disjoint from native Qwen and qwenFS dimensions."""
    def __init__(self, stages, start=1024, stop=F.OFF):
        self.stages = stages
        normalized = [stage if isinstance(stage, dict) else {"rom": stage} for stage in stages]
        for stage in normalized:
            if "decision" in stage and stage.get("free_span"):
                raise ValueError("a stage cannot have both a typed decision and a free span")
        spans = [len(stage["rom"])
                 + int(stage.get("decision", {}).get("token_count", 1)
                       if "decision" in stage else 0)
                 + int(stage.get("free_span", 0))
                 + len(stage.get("rom_suffix", [])) for stage in normalized]
        d = F.Dims(start)
        self.anchor = d.take()
        self.phase_in = d.take(len(stages))
        self.phase = d.take(len(stages))
        self.counter = d.take()
        self.offset = d.take(max(spans) + 1)
        self.choice_mask = d.take() if any("decision" in stage for stage in normalized) else None
        self.free_mask = d.take() if any(stage.get("free_span") for stage in normalized) else None
        self.dims = list(range(start, d.n))
        if d.n > stop:
            raise ValueError(f"agent scratch needs {d.n - start} dims in [{start}, {stop})")


def compile_into(model, stages, xs, native_width=1024, scratch_start=1024,
                 scratch_stop=F.OFF):
    """Install the agent by changing weights only, before serialization."""
    stages = [stage if isinstance(stage, dict) else {"rom": stage} for stage in stages]
    a = Layout(stages, scratch_start, scratch_stop)
    layers = model.model.layers
    E = model.model.embed_tokens.weight
    norm_scale = M.ballast_scale(model.config.hidden_size)
    with torch.no_grad():
        for layer in layers[:len(F.LAYERS)]:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                norm.weight[a.dims] = norm_scale - 1
        for i in range(len(stages)):
            E[PHASE0 + i, a.anchor] = 1
            E[PHASE0 + i, a.phase_in[i]] = 1
        # Control symbols look like a newline to the pretrained language stack.
        E[PHASE0:HALT + 1, :native_width] = 0.25 * E[F.NL, :native_width]
        decisions = [stage["decision"] for stage in stages if "decision" in stage]
        if decisions:
            allowed_sets = {tuple(decision["allowed_token_ids"]) for decision in decisions}
            if len(allowed_sets) != 1:
                raise ValueError("one compiled checkpoint currently supports one typed-choice set")
            allowed = next(iter(allowed_sets))
            E[:, a.choice_mask] = 0
            E[list(allowed), a.choice_mask] = 1
            model.model.norm.weight[a.choice_mask] = CHOICE_GAIN - 1
        free_stages = [stage for stage in stages if stage.get("free_span")]
        if free_stages:
            stops = {int(stage.get("free_vocab_stop", NATIVE_VOCAB_STOP))
                     for stage in free_stages}
            if len(stops) != 1:
                raise ValueError("one compiled checkpoint currently supports one free-span mask")
            free_vocab_stop = next(iter(stops))
            if not 0 < free_vocab_stop <= E.shape[0]:
                raise ValueError(f"invalid free_vocab_stop {free_vocab_stop}")
            # Preserve the native logits between ordinary tokens exactly.  A
            # positive constant on every legal token is not safe in bf16: at a
            # large gate value it rounds nearby native logits into ties.  Give
            # only illegal control tokens a negative embedding coordinate, so
            # activating the gate subtracts from them while legal rows remain
            # untouched (and feed no mask value back on the next token).
            E[:, a.free_mask] = -1
            E[:free_vocab_stop, a.free_mask] = 0
            model.model.norm.weight[a.free_mask] = CHOICE_GAIN - 1

        delta = F.Delta(layers[0].linear_attn)
        delta.register(2, {a.anchor: 1.0},
                       [("01", {x: 1.0}, y) for x, y in zip(a.phase_in, a.phase)])
        F.counter(delta, 3, [a.anchor], a.counter)

        # Chat tool schemas contain literal example tool calls. Those examples
        # can leave qwenFS's transient parser in INCALL/LISTMODE at prefill's
        # end, causing its response-code output to collide with our ROM. A
        # phase control token clears only transient parser flags; file/version
        # tables, path lookup, copy rows, and links remain untouched.
        delta = F.Delta(layers[1].linear_attn)
        resets = [
            (3, {F.NSTART: 1.0, F.FL["TRESPEND"]: 1.0}, F.NSTART, F.PENDING),
            (4, {F.NSTART: 1.0, F.TSTART: 1.0}, F.TSTART, F.WROTE),
            (5, {F.FL["TRESPEND"]: 1.0, F.LAG1["THINKEND"]: 1.0}, F.FL["TRESPEND"], F.TAIL),
            (6, {F.FL["TCSTART"]: 1.0, F.FL["TRESPEND"]: 1.0}, F.FL["TCSTART"], F.INCALL),
            (7, {F.ISMARK: 1.0, F.FL["TRESPEND"]: 1.0}, F.ISMARK, F.LISTMODE),
        ]
        for head, key, value, out in resets:
            delta.register(head, {**key, a.anchor: 1.0}, [("bit01", {value: 1.0}, out)])

        # Decode the same calibrated decay counter used by qwenFS. offset[n]
        # is one-hot for n tokens after a control symbol (including n=0).
        m = F.mlp_p()
        m.relu({F.ONE: 1.0}, {a.offset[0]: 1.0})
        for n in range(1, len(a.offset)):
            tau = math.sqrt(xs[n - 1] * xs[n])
            down = {a.offset[n]: 1.0, a.offset[n - 1]: -1.0}
            m.unit({F.ONE: 126.0, F.ONE2: 1.0, a.counter: -126.0 / tau},
                   {F.ONE: 1.0}, down)
            m.unit({F.ONE: 126.0, a.counter: -126.0 / tau},
                   {F.ONE: 1.0}, {d: -v for d, v in down.items()})
        # Clear the preexisting MLP before rewriting its expanded unit list.
        for p in layers[0].mlp.parameters():
            p.zero_()
        m.write(layers[0].mlp)

        exit_mlp = F.mlp6()
        # The regular exit clears FS scratch and converts OUTC into token bits.
        # Add the ROM's output code to those same token bits; ROM and filesystem
        # response intervals are disjoint. No logits are edited by the decoder.
        for i, stage in enumerate(stages):
            prompt = stage["rom"]
            for n, token in enumerate(prompt):
                exit_mlp.all_of([a.phase[i], a.offset[n]],
                                {d: F.EPS * c for d, c in zip(F.IDC, F.code(token))})
            span = len(prompt)
            if "decision" in stage:
                decision_count = int(stage["decision"].get("token_count", 1))
                for n in range(span, span + decision_count):
                    exit_mlp.all_of([a.phase[i], a.offset[n]], {a.choice_mask: F.EPS})
                span += decision_count
            elif stage.get("free_span"):
                for n in range(span, span + int(stage["free_span"])):
                    exit_mlp.all_of([a.phase[i], a.offset[n]], {a.free_mask: F.EPS})
                span += int(stage["free_span"])
            for n, token in enumerate(stage.get("rom_suffix", []), start=span):
                exit_mlp.all_of([a.phase[i], a.offset[n]],
                                {d: F.EPS * c for d, c in zip(F.IDC, F.code(token))})
            span += len(stage.get("rom_suffix", []))
            nxt = PHASE0 + i + 1 if i + 1 < len(stages) else HALT
            # A qwenFS response finishes by opening an empty assistant turn.
            # Its final NL2 follows THINKEND, after the whole ROM. At this
            # position the FS circuit has released output control. Advance
            # immediately, without asking Qwen to choose another operation.
            exit_mlp.all_of([a.phase[i], F.FL["NL2"], F.LAG1["THINKEND"]],
                            {d: F.EPS * c for d, c in zip(F.IDC, F.code(nxt))},
                            nots=a.offset[:span])
        for d in a.dims:
            exit_mlp.unit({F.ONE: F.K}, {d: -1 / F.K}, {d: 1.0})
        for p in layers[7].mlp.parameters():
            p.zero_()
        if len(exit_mlp.units) > layers[7].mlp.gate_proj.out_features:
            raise ValueError(f"ROM needs {len(exit_mlp.units)} MLP units")
        exit_mlp.write(layers[7].mlp)
    return a


def build(src, dst, cycles=3, device="mps", protocol="baseline", think_tokens=None):
    tok = AutoTokenizer.from_pretrained(src)
    stages = program(tok, cycles, protocol, think_tokens)
    prompts = [s["rom"] for s in stages]
    cfg = M.text_config(src)
    native_width = cfg["hidden_size"]
    if native_width <= 1024:
        target_hidden, scratch_start, scratch_stop = F.H, 1024, F.OFF
    elif native_width == F.OFF:
        # Qwen3.5-2B occupies 0..2047 and qwenFS occupies 2048..3071.
        # Give the controller a new tail rather than corrupting either one.
        target_hidden, scratch_start, scratch_stop = 4096, F.H, 4096
    else:
        raise ValueError(f"No verified agent layout for native width {native_width}")
    xs = F.calibrate(device)
    model = M.merge(src, xs, 4096.0, target_hidden=target_hidden)
    layout = compile_into(model, stages, xs, native_width, scratch_start, scratch_stop)
    model.config.bos_token_id = START
    model.config.eos_token_id = HALT
    model.generation_config.bos_token_id = START
    model.generation_config.eos_token_id = HALT
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, max_shard_size="1GB")
    tok.save_pretrained(dst)
    (dst / "agent.json").write_text(json.dumps({
        "base": str(src), "cycles": cycles, "protocol": protocol,
        "think_tokens": (next((stage["free_span"] for stage in stages
                               if stage.get("free_span")), None)),
        "start": START, "halt": HALT,
        "native_hidden": native_width, "merged_hidden": target_hidden,
        "scratch": [layout.dims[0], layout.dims[-1] + 1],
        "choice_mask_dim": layout.choice_mask, "choice_gain": CHOICE_GAIN,
        "free_mask_dim": layout.free_mask,
        "phase_tokens": list(range(PHASE0, PHASE0 + len(prompts))),
        "stages": [{k: v for k, v in s.items() if k != "rom"} for s in stages],
        "phases": [s["kind"] for s in stages],
        "rom": prompts, "fs_layers": len(F.LAYERS), "code_gain": 4096.0,
    }, indent=2) + "\n")
    print(f"Compiled {cycles} cycles / {len(prompts)} microstates into {dst}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="baseline")
    parser.add_argument("--think-tokens", type=int,
                        help="bounded reasoning tokens for pointer protocols")
    args = parser.parse_args()
    build(args.src, args.dst, args.cycles, args.device, args.protocol,
          args.think_tokens)


if __name__ == "__main__":
    main()
