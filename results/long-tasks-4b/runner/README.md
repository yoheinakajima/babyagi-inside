# Runner source, as run

These files are the research lab's runner for the longer-task screen, copied verbatim from
the lab commit every test step ran at (`b3e3158`). Every record in `../` carries its
`runner_sha256` (`d55cd616…`) and ROM SHA-256 (`571d2f5d…`) in its identity.

- `adaptive_screen.py`: the loop arms (H-adapt, H-oracle, H-adapt-v3), the budget-forced
  plain arm (P-F2), scoring, the audit, and the preregistered analysis (`analyze`).
- `suite_v4.py`: the seeded generator of `benchmarks/babyagi-v4-horizon*.json`.
- `plain_forced.py`: the lab's E14 plain runner; P-F2 uses its `generate_forced`, whose
  forcing [`qfs/plain_forced.py`](../../../qfs/plain_forced.py) carries.
- `port.py`, `upstream.py`: lab utilities (output writing and importing this repository's
  `qfs` modules).

They import the lab's own packages (`harness`, `lab`), so they are published for reading
and audit, not as a module of this repository. The records are checked on CPU by
[`tests/test_long_tasks.py`](../../../tests/test_long_tasks.py).
