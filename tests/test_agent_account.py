"""Cheap exact tests for compiled-agent trace provenance accounting."""
from qfs import circuit as F
from qfs.agent_account import account, account_data


class FakeTokenizer:
    close = [701, F.TCEND]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if "</parameter>" in text:
            return self.close
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(str(token) for token in tokens)


def synthetic_check():
    tok = FakeTokenizer()
    meta = {
        "protocol": "synthetic",
        "cycles": 1,
        "halt": 902,
        "phase_tokens": [900, 901],
        "rom": [[100, 101], [110, F.TCEND]],
        "stages": [
            {"cycle": 1, "kind": "create", "op": "write", "file": "tasks.txt",
             "text_prefix": ""},
            {"cycle": 1, "kind": "prioritize", "op": "read", "file": "tasks.txt",
             "text_prefix": ""},
        ],
    }
    # write: ROM, one payload, two closing tokens, one FS response, transition;
    # read: ROM already closes the call, one FS response, HALT.
    record = {"tokens": [100, 101, 200, *tok.close, 300, 901,
                         110, F.TCEND, 301, 902]}
    report = account_data(record, meta, tok)
    assert report["totals"]["emitted_tokens"] == 11
    assert report["totals"]["rom_forced"] == 4
    assert report["totals"]["filesystem_forced"] == 2
    assert report["totals"]["controller_control"] == 2
    assert report["totals"]["model_generated"] == 3
    assert report["totals"]["model_payload"] == 1
    assert report["totals"]["model_tool_closure"] == 2
    assert report["decision_summary"] == {"model_free_text": 1}
    assert all(report["invariants"].values())


def typed_check():
    tok = FakeTokenizer()
    meta = {
        "protocol": "typed",
        "cycles": 1,
        "halt": 902,
        "phase_tokens": [900],
        "rom": [[120]],
        "stages": [{
            "cycle": 1, "kind": "prioritize", "op": "write", "file": "choice.txt",
            "text_prefix": "T", "rom_suffix": tok.close,
            "decision": {"mode": "typed_choice", "allowed_token_ids": [16, 17],
                         "token_count": 1},
        }],
    }
    record = {"tokens": [120, 17, *tok.close, 300, 902]}
    report = account_data(record, meta, tok)
    assert report["totals"]["emitted_tokens"] == 6
    assert report["totals"]["rom_forced"] == 3
    assert report["totals"]["filesystem_forced"] == 1
    assert report["totals"]["controller_control"] == 1
    assert report["totals"]["model_generated"] == 1
    assert report["decisions"][0]["owner"] == "typed_model_choice"
    assert report["decisions"][0]["chosen_token_ids"] == [17]


def recorded_trace_check():
    cases = (
        ("results/m5-2b-homage-5cycle/live-pine-seed0.json", 5284, 4383, 901,
         {"model_free_text": 15}),
        ("results/m5-2b-homage-kshot-5cycle/live-pine-seed0.json", 4293, 4002, 291,
         {"model_free_text": 15}),
        ("results/m5-2b-json-5cycle/live-pine-seed0.json", 6105, 5234, 871,
         {"model_free_text": 10, "deterministic_circuit_choice": 5}),
    )
    checked = 0
    for path, emitted, forced, model, decisions in cases:
        from pathlib import Path
        if not Path(path).is_file():
            continue
        report = account(path)
        assert report["totals"]["emitted_tokens"] == emitted
        assert report["totals"]["forced_total"] == forced
        assert report["totals"]["model_generated"] == model
        assert report["decision_summary"] == decisions
        checked += 1
    return checked


def check():
    synthetic_check()
    typed_check()
    checked = recorded_trace_check()
    print({"synthetic": "PASS", "typed": "PASS", "recorded_traces": checked})


if __name__ == "__main__":
    check()
