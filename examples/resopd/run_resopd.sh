#!/usr/bin/env bash
# ResOPD on the released DeepMath subset: full-support sampling + sparse vLLM teacher.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
# Download train.parquet and heldout.parquet from ygyjrc/ResOPD-deepmath (see README).
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/resopd-deepmath}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/heldout.parquet}

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3.5-4B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3.5-27B}
STUDENT_GPUS=${STUDENT_GPUS:-4}
TEACHER_GPUS=${TEACHER_GPUS:-4}
TEACHER_TP=${TEACHER_TP:-4}
TOPK=${TOPK:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

cmd=(
    "${PYTHON_BIN:-python}" -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    "data.train_files=$TRAIN_FILE"
    "data.val_files=$VAL_FILE"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=error
    "actor_rollout_ref.model.path=$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0.0
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE:-1e-6}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.calculate_log_probs=True
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN"
    distillation.enabled=True
    distillation.nnodes=1
    "distillation.n_gpus_per_node=$TEACHER_GPUS"
    "distillation.teacher_models.teacher_model.model_path=$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.name=vllm
    "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP"
    "distillation.teacher_models.teacher_model.inference.max_model_len=$MAX_MODEL_LEN"
    "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$MAX_MODEL_LEN"
    distillation.distillation_loss.loss_mode=resopd
    "distillation.distillation_loss.topk=$TOPK"
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.loss_max_clamp=null
    distillation.distillation_loss.log_prob_min_clamp=null
    "reward.custom_reward_function.path=$SCRIPT_DIR/deepmath_reward.py"
    reward.custom_reward_function.name=compute_score
    trainer.nnodes=1
    "trainer.n_gpus_per_node=$STUDENT_GPUS"
    trainer.project_name=ResOPD
    "trainer.experiment_name=${EXPERIMENT_NAME:-resopd-topk$TOPK}"
    'trainer.logger=["console"]'
    trainer.val_before_train=False
    "trainer.total_epochs=${TOTAL_EPOCHS:-1}"
    "trainer.save_freq=${SAVE_FREQ:-50}"
    "trainer.test_freq=${TEST_FREQ:-50}"
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${cmd[@]}" "$@"
    printf '\n'
else
    exec "${cmd[@]}" "$@"
fi
