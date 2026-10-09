# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Optional


def extract_prompt_logprobs(output: Any, num_prompt_logprobs: Optional[int], result_dict: dict[str, list]):
    """Extract prompt log probabilities from generation output."""
    if num_prompt_logprobs is None:
        return

    prompt_logprobs_ls, prompt_ids_ls = [], []
    sampled_prompt_logprobs_ls, sampled_prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for sampled_token_id, logprobs_dict in zip(output.prompt_token_ids[1:], output.prompt_logprobs[1:], strict=True):
        sampled_token_id = int(sampled_token_id)
        sampled_entry = next(
            token_logprob for token_id, token_logprob in logprobs_dict.items() if int(token_id) == sampled_token_id
        )
        sampled_prompt_ids_ls.append([sampled_token_id])
        sampled_prompt_logprobs_ls.append([sampled_entry.logprob])
        if num_prompt_logprobs == 0:
            prompt_logprobs_ls.append([sampled_entry.logprob])
            prompt_ids_ls.append([sampled_token_id])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))
    sampled_prompt_logprobs_ls.append([0.0])
    sampled_prompt_ids_ls.append([0])

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls
    # vLLM returns the observed prompt token in addition to Top-K when that
    # token falls outside Top-K. Preserve it separately for estimators that use
    # exact head integration and sampled tail correction.
    result_dict["sampled_prompt_ids"] = sampled_prompt_ids_ls
    result_dict["sampled_prompt_logprobs"] = sampled_prompt_logprobs_ls
