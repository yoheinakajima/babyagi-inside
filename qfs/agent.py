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
)


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


def program(tok, cycles=3, protocol="baseline"):
    """Only static instructions are compiled; objectives/tasks are never baked in."""
    if cycles < 1:
        raise ValueError("cycles must be positive")
    stages = []
    for cycle in range(1, cycles + 1):
        shared, instructions = phase_prompts(cycle, cycles, protocol)
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
            frame = ("\n<|im_start|>user\n" + shared + instruction +
                     " Keep it under 60 words. Finish by closing the tool call."
                     "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            if protocol == "babyagi_json":
                text_prefix = "T1" if kind == "prioritize" else "{"
            else:
                text_prefix = ("FIRST TASK: " if protocol in ("babyagi", "babyagi_kshot")
                               and cycle == 1 and kind == "create" else "")
            write = ("<tool_call>\n<function=write_file>\n<parameter=name>\n" + filename +
                     "\n</parameter>\n<parameter=text>\n" + text_prefix)
            stages.append({"cycle": cycle, "kind": kind, "op": "write", "file": filename,
                           "text_prefix": text_prefix,
                           "rom": tok.encode(prefix + frame + write, add_special_tokens=False)})
    phase_budget = HALT - PHASE0
    if len(stages) > phase_budget:
        raise ValueError(
            f"{protocol} protocol needs {len(stages)} microstates for {cycles} cycles; "
            f"the compiled controller reserves {phase_budget} before HALT")
    return stages


class Layout:
    """Controller scratch, disjoint from native Qwen and qwenFS dimensions."""
    def __init__(self, prompts, start=1024, stop=F.OFF):
        self.prompts = prompts
        d = F.Dims(start)
        self.anchor = d.take()
        self.phase_in = d.take(len(prompts))
        self.phase = d.take(len(prompts))
        self.counter = d.take()
        self.offset = d.take(max(map(len, prompts)) + 1)
        self.dims = list(range(start, d.n))
        if d.n > stop:
            raise ValueError(f"agent scratch needs {d.n - start} dims in [{start}, {stop})")


def compile_into(model, prompts, xs, native_width=1024, scratch_start=1024,
                 scratch_stop=F.OFF):
    """Install the agent by changing weights only, before serialization."""
    a = Layout(prompts, scratch_start, scratch_stop)
    layers = model.model.layers
    E = model.model.embed_tokens.weight
    norm_scale = M.ballast_scale(model.config.hidden_size)
    with torch.no_grad():
        for layer in layers[:len(F.LAYERS)]:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                norm.weight[a.dims] = norm_scale - 1
        for i in range(len(prompts)):
            E[PHASE0 + i, a.anchor] = 1
            E[PHASE0 + i, a.phase_in[i]] = 1
        # Control symbols look like a newline to the pretrained language stack.
        E[PHASE0:HALT + 1, :native_width] = 0.25 * E[F.NL, :native_width]

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
        for i, prompt in enumerate(prompts):
            for n, token in enumerate(prompt):
                exit_mlp.all_of([a.phase[i], a.offset[n]],
                                {d: F.EPS * c for d, c in zip(F.IDC, F.code(token))})
            nxt = PHASE0 + i + 1 if i + 1 < len(prompts) else HALT
            # A qwenFS response finishes by opening an empty assistant turn.
            # Its final NL2 follows THINKEND, after the whole ROM. At this
            # position the FS circuit has released output control. Advance
            # immediately, without asking Qwen to choose another operation.
            exit_mlp.all_of([a.phase[i], F.FL["NL2"], F.LAG1["THINKEND"]],
                            {d: F.EPS * c for d, c in zip(F.IDC, F.code(nxt))},
                            nots=a.offset[:len(prompt)])
        for d in a.dims:
            exit_mlp.unit({F.ONE: F.K}, {d: -1 / F.K}, {d: 1.0})
        for p in layers[7].mlp.parameters():
            p.zero_()
        if len(exit_mlp.units) > layers[7].mlp.gate_proj.out_features:
            raise ValueError(f"ROM needs {len(exit_mlp.units)} MLP units")
        exit_mlp.write(layers[7].mlp)
    return a


def build(src, dst, cycles=3, device="mps", protocol="baseline"):
    tok = AutoTokenizer.from_pretrained(src)
    stages = program(tok, cycles, protocol)
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
    layout = compile_into(model, prompts, xs, native_width, scratch_start, scratch_stop)
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
        "start": START, "halt": HALT,
        "native_hidden": native_width, "merged_hidden": target_hidden,
        "scratch": [layout.dims[0], layout.dims[-1] + 1],
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
    args = parser.parse_args()
    build(args.src, args.dst, args.cycles, args.device, args.protocol)


if __name__ == "__main__":
    main()
