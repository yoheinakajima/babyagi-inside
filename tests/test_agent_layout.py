"""Cheap invariants for the two verified compiled-agent residual layouts."""
from qfs import circuit as F
from qfs.agent import Layout, phase_prompts, program


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))


def check():
    prompts = [[1, 2, 3], [4, 5]]
    small = Layout(prompts, 1024, F.OFF)
    two_b = Layout(prompts, F.H, 4096)
    assert max(small.dims) < F.OFF
    assert min(two_b.dims) >= F.H
    assert max(two_b.dims) < 4096
    assert set(two_b.dims).isdisjoint(range(F.OFF, F.H))
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
    assert len(program(tok, 4, "generative")) == 26
    for cycles, protocol in ((6, "babyagi"), (6, "babyagi_json"), (5, "generative")):
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
