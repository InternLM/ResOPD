# Release validation

Validated on 2026-09-29 using Python 3.12 and PyTorch 2.11.0, with tensor
operations restricted to CPU and one thread. No model weights were loaded.

| Check | Result |
| --- | --- |
| `pytest tests/resopd/test_estimator.py tests/resopd/test_teacher_payload.py -q` | 22 passed |
| `pytest tests/resopd/test_verl_integration.py -q` | 13 passed |
| Comparison with the authors' original research kernel | 45 action/seed/support combinations passed; maximum absolute FP32 gradient difference `1.341104507446289e-07` |
| Compose `ppo_trainer` with every override emitted by `run_resopd.sh` | Passed |
| Python syntax, Ruff on changed Python files, shell syntax and `git diff --check` | Passed |

The original-kernel comparison checked both the logged forward estimator and
backward gradients. It used the research implementation as a separate local
reference; that experiment tree and its history are not part of this release.

The integration tests exercise actual verl configuration, registration, nested
logits dispatch, teacher-manager calls, agent serialization, response alignment,
masking and synthetic padding. Teacher RPC responses are mocked. Model serving,
multi-GPU execution and an end-to-end training run of this extracted upstream
port have not been rerun as part of the release validation.

On 2026-10-06, all 13 CPU integration checks passed again after removing the
deprecated loss alias. The registry check now verifies that the ResOPD collector
is registered only as `resopd`. GitHub's Markdown API also recognized both
README estimator equations as display math outside the collapsible section.
