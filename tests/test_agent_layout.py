"""Cheap invariants for the two verified compiled-agent residual layouts."""
from qfs import circuit as F
from qfs.agent import Layout, phase_prompts


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
    print({"0.8b_scratch": [min(small.dims), max(small.dims) + 1],
           "2b_scratch": [min(two_b.dims), max(two_b.dims) + 1],
           "qwenfs": [F.OFF, F.H]})


if __name__ == "__main__":
    check()
