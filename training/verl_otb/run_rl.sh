#!/usr/bin/env bash
# Dr.GRPO on Board Simulation, with the reward in reward.py.
# All judge calls go to one local server started by start_judge.sh.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_SCRATCH=${PROJECT_SCRATCH:-$HERE/outputs/rl}
REFERENCE_SCOPE_DEFAULT="$PROJECT_SCRATCH/reference_scope.jsonl"


EXPERIMENT_NAME=${EXPERIMENT_NAME:-otb_drgrpo}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/otb_ray}
mkdir -p "$RAY_TEMP_DIR"

export CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDS:-2,3}
export PYTHONUNBUFFERED=1
export OTB_JUDGE_BASE_URL=${OTB_JUDGE_BASE_URL:-http://127.0.0.1:8094/v1}
export OTB_JUDGE_MODEL=${OTB_JUDGE_MODEL:-judge}
export OTB_JUDGE_MAX_TOKENS=${OTB_JUDGE_MAX_TOKENS:-768}
export OTB_JUDGE_TIMEOUT=${OTB_JUDGE_TIMEOUT:-300}
export OTB_JUDGE_RETRIES=${OTB_JUDGE_RETRIES:-3}
export OTB_REFERENCE_SCOPE_PATH=${OTB_REFERENCE_SCOPE_PATH:-$REFERENCE_SCOPE_DEFAULT}

unset MAX_PIXELS

if [ ! -f "$OTB_REFERENCE_SCOPE_PATH" ]; then
  echo "Frozen reference-scope cache not found: $OTB_REFERENCE_SCOPE_PATH" >&2
  exit 2
fi

INIT_MODE=sft \
SFT_MODEL=${SFT_MODEL:?set SFT_MODEL to the finetuned checkpoint} \
RUN_MODE=${RUN_MODE:-train} \
REWARD_PATH="$HERE/reward.py" \
DATA_DIR=${DATA_DIR:-$HERE/data} \
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2} \
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5} \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4} \
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4} \
ROLLOUT_N=${ROLLOUT_N:-16} \
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7} \
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0} \
ACTOR_LR=${ACTOR_LR:-1e-6} \
ENTROPY_COEFF=${ENTROPY_COEFF:-0.0} \
USE_KL_LOSS=false \
KL_LOSS_COEF=0.0 \
LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-sum-norm} \
ROLLOUT_IS=token \
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0} \
ROLLOUT_IS_BATCH_NORMALIZE=false \
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40} \
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-65536} \
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192} \
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-74240} \
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true} \
SAVE_GENERATIONS=${SAVE_GENERATIONS:-true} \
SAVE_FREQ=${SAVE_FREQ:-10} \
TEST_FREQ=${TEST_FREQ:-10} \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
PROJECT_SCRATCH="$PROJECT_SCRATCH" \
  bash "$HERE/run_grpo.sh" \
    algorithm.norm_adv_by_std_in_grpo=false \
    actor_rollout_ref.actor.loss_scale_factor=8192 \
    reward.num_workers=16 \
    ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-32} \
    +ray_kwargs.ray_init.include_dashboard=false \
    +ray_kwargs.ray_init._temp_dir="$RAY_TEMP_DIR" \
    trainer.resume_mode=disable \
    trainer.max_actor_ckpt_to_keep=92 \
    +ray_kwargs.ray_init.runtime_env.env_vars.OTB_REFERENCE_SCOPE_PATH="$OTB_REFERENCE_SCOPE_PATH" \
    +actor_rollout_ref.actor.policy_loss.response_mask_start_marker="'<conclusion'" \
    +actor_rollout_ref.actor.policy_loss.response_mask_end_marker="'conclusion>'" \
    +actor_rollout_ref.actor.policy_loss.response_mask_preceding_tokens=8192 \
    +actor_rollout_ref.actor.policy_loss.response_mask_missing_start_tail_tokens=8192 \
    +actor_rollout_ref.actor.policy_loss.split_advantage_match_key=match_score \
    +actor_rollout_ref.actor.policy_loss.split_advantage_format_key=format_score \
    +actor_rollout_ref.actor.policy_loss.split_advantage_format_weight=0.1 \
    +data.mm_processor_kwargs.size.longest_edge=589824 \
    +data.mm_processor_kwargs.size.shortest_edge=3136 \
    
    "$@"
