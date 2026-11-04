"""Cheap invariants for the two verified compiled-agent residual layouts."""
from qfs import circuit as F
from qfs.agent import Layout, phase_prompts, program


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in ("1", "2"):
            return [int(text)]
        return list(range(100, 100 + len(text.split())))


def check():
    prompts = [[1, 2, 3], [4, 5]]
    small = Layout(prompts, 1024, F.OFF)
    two_b = Layout(prompts, F.H, 4096)
    assert max(small.dims) < F.OFF
    assert min(two_b.dims) >= F.H
    assert max(two_b.dims) < 4096
    assert set(two_b.dims).isdisjoint(range(F.OFF, F.H))
    typed_layout = Layout([
        {"rom": [1, 2], "decision": {"allowed_token_ids": [16, 17]},
         "rom_suffix": [3, 4]},
    ], F.H, 4096)
    assert typed_layout.choice_mask is not None
    assert typed_layout.free_mask is None
    assert len(typed_layout.offset) == 6
    bounded_layout = Layout([
        {"rom": [1, 2], "free_span": 4, "free_vocab_stop": 100,
         "rom_suffix": [3, 4]},
    ], F.H, 4096)
    assert bounded_layout.choice_mask is None
    assert bounded_layout.free_mask is not None
    assert len(bounded_layout.offset) == 9
    shared, (create, prioritize, execute) = phase_prompts(2, 3, "generative")
    text = shared + create + prioritize + execute
    assert "first user message" in shared
    assert "earliest unfinished operation" in create
    assert "Never copy the whole objective" in create
    assert "at most the next two" in create
    assert "Renumber candidates" in create
    assert "Write only" in prioritize
    assert "previous result was just read" in execute
    assert "Input: actual input" in execute
    assert "Output: actual output" in execute
    for objective_specific_value in ("oak", "red", "blue", "gold"):
        assert objective_specific_value not in text
    babyagi_create = phase_prompts(1, 3, "babyagi")[1][0]
    assert "INITIAL TASK — Create the first task" in babyagi_create
    assert "exactly one concrete first executable task" in babyagi_create
    kshot_create = phase_prompts(1, 3, "babyagi_kshot")[1][0]
    assert "Valid one-task shapes using fake values" in kshot_create
    assert "For an open goal" in kshot_create
    kshot_later = phase_prompts(2, 3, "babyagi_kshot")[1][0]
    assert "objective has ordered operations" in kshot_later
    assert "objective has no ordered operations" in kshot_later
    for objective_specific_value in ("oak", "pine", "-red", "-blue", "-gold", "-silver"):
        assert objective_specific_value not in babyagi_create
        assert objective_specific_value not in kshot_create
    json_shared, json_instructions = phase_prompts(2, 5, "babyagi_json")
    assert "one-line JSON" in " ".join(json_instructions)
    assert "Completed tasks are finished" in json_instructions[0]
    assert "complete choice `T1`" in json_instructions[1]
    assert "actual result" in json_instructions[2]
    assert "task description" in json_instructions[2]
    assert "original objective" in json_shared
    tok = TinyTokenizer()
    assert len(program(tok, 5, "babyagi")) == 29
    assert len(program(tok, 5, "babyagi_kshot")) == 29
    json_program = program(tok, 5, "babyagi_json")
    assert len(json_program) == 29
    assert {stage["file"] for stage in json_program} == {"tasks.json", "choice.txt"}
    for stage in json_program:
        if stage["op"] != "write":
            continue
        assert stage["text_prefix"] == ("T1" if stage["kind"] == "prioritize" else "{")
    typed_program = program(tok, 5, "babyagi_typed")
    typed_decisions = [stage for stage in typed_program if "decision" in stage]
    assert len(typed_program) == 24
    assert len(typed_decisions) == 5
    assert all(stage["decision"]["allowed_token_ids"] == [1, 2]
               and stage["decision"]["owner"] == "model"
               and stage["op"] == "read"
               and stage["selected_files"] == ["task1.txt", "task2.txt"]
               and stage["rom_suffix"] for stage in typed_decisions)
    assert "INITIAL TASK — Create the first task" in phase_prompts(
        1, 5, "babyagi_typed")[1][0]
    assert "previous result was just read" in phase_prompts(
        2, 5, "babyagi_typed")[1][0]
    pointer_program = program(tok, 5, "babyagi_pointer")
    pointer_execute = [stage for stage in pointer_program
                       if stage["kind"] == "execute" and stage["op"] == "write"]
    pointer_reads = [stage for stage in pointer_program if stage["kind"] == "prioritize"]
    assert len(pointer_program) == 19
    assert [stage["file"] for stage in pointer_reads] == [
        "task1.txt", "task2.txt", "task3.txt", "task4.txt", "task5.txt"]
    assert all(stage["selection"] == "deterministic_ledger_pointer"
               for stage in pointer_reads)
    assert len(pointer_execute) == 5
    assert all(stage["free_span"] == 128 and stage["rom_suffix"]
               and stage["file"] == "result.json" for stage in pointer_execute)
    pointer_layout = Layout(pointer_program, F.H, 4096)
    assert pointer_layout.free_mask is not None
    assert max(pointer_layout.dims) < 4096
    ledger_pointer = program(tok, 5, "babyagi_ledger_pointer")
    assert len(ledger_pointer) == 15
    assert sum(stage["op"] == "write" and stage["file"] == "ledger.json"
               for stage in ledger_pointer) == 1
    assert [stage["pointer_index"] for stage in ledger_pointer
            if stage["kind"] == "prioritize"] == [1, 2, 3, 4, 5]
    assert all(stage["file"] == "ledger.json" and
               stage["selection"] == "deterministic_ledger_pointer"
               for stage in ledger_pointer if stage["kind"] == "prioritize")
    ledger_layout = Layout(ledger_pointer, F.H, 4096)
    assert ledger_layout.free_mask is not None and max(ledger_layout.dims) < 4096
    short_budget = program(tok, 5, "babyagi_ledger_pointer", think_tokens=32)
    assert all(stage.get("free_span") in (None, 32) for stage in short_budget)
    assert max(Layout(short_budget, F.H, 4096).dims) < max(ledger_layout.dims)
    long_chain = program(tok, 8, "babyagi_ledger_pointer", think_tokens=256)
    assert len(long_chain) == 24
    assert max(Layout(long_chain, F.H, 4096).dims) < 4096
    assert len(program(tok, 4, "generative")) == 26
    for cycles, protocol in ((6, "babyagi"), (6, "babyagi_json"),
                             (7, "babyagi_typed"), (9, "babyagi_pointer"),
                             (12, "babyagi_ledger_pointer"),
                             (5, "generative")):
        try:
            program(tok, cycles, protocol)
        except ValueError as exc:
            assert "microstates" in str(exc) and "before HALT" in str(exc)
        else:
            raise AssertionError(f"{protocol} unexpectedly accepted {cycles} cycles")
    print({"0.8b_scratch": [min(small.dims), max(small.dims) + 1],
           "2b_scratch": [min(two_b.dims), max(two_b.dims) + 1],
           "qwenfs": [F.OFF, F.H]})


if __name__ == "__main__":
    check()
