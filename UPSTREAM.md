# Upstream and scope

ResOPD implements the estimator described in
[ResOPD: Tail Residualization for Sparse On-Policy Distillation](https://arxiv.org/abs/2610.04882).

This repository starts from the public [verl](https://github.com/verl-project/verl)
snapshot [`1dda039b9c6311889a9c9a26040677e1254814f9`](https://github.com/verl-project/verl/commit/1dda039b9c6311889a9c9a26040677e1254814f9).
The first commit contains that snapshot, including the original public `recipe`
submodule reference. The following commit adds ResOPD and its required integration.

The method was extracted from the authors' research implementation. The release
retains its centered support-event surrogate, teacher Top-k plus sampled-token interface, and numerical
tail-mass guards. The standalone kernel promotes FP16/BF16 arithmetic to FP32 and
preserves FP64 for checks. Use `loss_mode: resopd` to select it.

Changes to the verl runtime are restricted to:

- the ResOPD kernel, logits dispatch, response-mask-aware collector and configuration checks;
- retaining the observed token score from vLLM's existing prompt-logprob response;
- carrying that channel through teacher management, agent outputs and batch packing;
- shortening sparse teacher tensors when constructing synthetic padding samples.

The teacher manager's original two-tensor return contract is retained for other
loss modes. ResOPD requests the additional pair. This release supports vLLM teachers
and eager FSDP/FSDP2/VeOmni students. The example uses synchronous FSDP2 training.

The original upstream README is preserved as `README.verl.md`. Upstream CI
workflows that depend on verl's infrastructure are replaced with a small ResOPD
CPU test workflow. Other upstream code and its Apache-2.0 notices are retained.
The repository does not import the authors' experiment branch history.
