"""Cheap tests for the stock-weight semantic harness; no model checkpoint required."""
import torch

from qfs import circuit as F
from qfs.agent_native import choose_token, stage_view, visible_response_tokens
from qfs.agent_suite import native_results


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [1000 + ord(char) for char in text]


def check():
    current = stage_view({"rom": [1, 2, 3]})
    assert current.prefix == [1, 2, 3]
    assert not current.suffix and not current.allowed
    assert current.free_span == 0 and current.free_vocab_stop is None

    future = stage_view({
        "rom_prefix": [4, 5], "rom_suffix": [6, F.TCEND],
        "decision": {"name": "task_id", "allowed_token_ids": [16, 17]},
    })
    assert future.prefix == [4, 5]
    assert future.suffix == [6, F.TCEND]
    assert future.allowed == [16, 17]
    assert not future.allowed_text
    assert future.decision_name == "task_id"

    bounded = stage_view({
        "rom": [7, 8], "rom_suffix": [9, 10],
        "free_span": 128, "free_vocab_stop": 24,
    })
    assert bounded.free_span == 128 and bounded.free_vocab_stop == 24

    logits = torch.full((32,), -10.0)
    logits[3], logits[16], logits[17] = 50.0, 2.0, 4.0
    gen = torch.Generator(device="cpu").manual_seed(0)
    assert choose_token(logits, 0.0, 0.9, gen) == 3
    assert choose_token(logits, 0.0, 0.9, gen, allowed=[16, 17]) == 17
    assert choose_token(logits, 0.0, 0.9, gen, vocab_stop=3) == 0

    tok = FakeTokenizer()
    write = visible_response_tokens(tok, "write_file", "tasks.txt")
    assert write[:6] == [F.NL, F.IMSTART, F.USER, F.NL, F.TRESP, F.NL]
    assert [F.NL, F.SAVED, F.NL, F.DASH] == write[6:10]
    assert write[-len([F.NL, F.TRESPEND] +
                      [F.NL, F.IMSTART, F.ASSISTANT, F.NL, F.THINK,
                       F.NL2, F.THINKEND, F.NL2]):] == [F.NL, F.TRESPEND] + [
                           F.NL, F.IMSTART, F.ASSISTANT, F.NL, F.THINK,
                           F.NL2, F.THINKEND, F.NL2]

    body = [7, 8, 9]
    read = visible_response_tokens(tok, "read_file", "tasks.txt", body)
    assert read[6:10] == [F.NL, *body]
    missing = visible_response_tokens(tok, "read_file", "missing.txt", None)
    assert missing[6:9] == [F.NL, F.NOT, F.FOUND_]
    trace = {"phases": [
        {"kind": "create", "op": "write", "parameters": {"text": "task"}},
        {"kind": "execute", "op": "write", "parameters": {
            "name": "result.txt", "text": "Output:  oak-red"}},
        {"kind": "execute", "op": "write", "parameters": {
            "name": "result.json", "text": '{"result":[1,2,3]}',
        }},
    ]}
    assert native_results(trace) == ["oak-red", [1, 2, 3]]
    print({"stage_adapter": "PASS", "typed_choice": "PASS", "responses": "PASS"})


if __name__ == "__main__":
    check()
