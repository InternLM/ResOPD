# Copyright 2026 Penghui Yang and the ResOPD authors
# SPDX-License-Identifier: Apache-2.0
"""Use verl's boxed-answer math scorer for the released DeepMath data source."""

from verl.utils.reward_score.math_reward import compute_score as math_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return math_score(solution_str, ground_truth)
