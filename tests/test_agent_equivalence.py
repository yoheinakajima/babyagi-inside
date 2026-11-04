"""Small format tests for host/compiled semantic equivalence."""

from qfs.agent_equivalence import EquivalenceError, _native_trace


def test_native_trace_accepts_bare_trace():
    trace = {"phases": [{"model_text": "answer"}]}
    assert _native_trace(trace) is trace


def test_native_trace_unwraps_end_to_end_record():
    trace = {"phases": [{"model_text": "answer"}]}
    assert _native_trace({"item_id": "example", "trace": trace}) is trace


def test_native_trace_rejects_malformed_wrapper():
    try:
        _native_trace({"trace": []})
    except EquivalenceError as exc:
        assert "non-object trace" in str(exc)
    else:
        raise AssertionError("malformed native wrapper was accepted")


def main():
    test_native_trace_accepts_bare_trace()
    test_native_trace_unwraps_end_to_end_record()
    test_native_trace_rejects_malformed_wrapper()
    print("agent equivalence format tests passed")


if __name__ == "__main__":
    main()
