#!/usr/bin/env bash
# OpenTumorBoard Task 1 GRPO on H200 GPUs. Defaults to a two-step smoke run.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

INIT_MODE=${INIT_MODE:-base}       # base | sft
RUN_MODE=${RUN_MODE:-smoke}        # smoke | train
BASE_MODEL=${BASE_MODEL:-}   # local copy of the base model, used when INIT_MODE=base
SFT_MODEL=${SFT_MODEL:-}    # finetuned checkpoint, used when INIT_MODE=sft
PYTHON_BIN=${PYTHON_BIN:-$REPO/.venv/bin/python}
PROJECT_SCRATCH=${PROJECT_SCRATCH:-$HERE/outputs/rl}
REWARD_PATH=${REWARD_PATH:-$HERE/reward.py}
DATA_DIR=${DATA_DIR:-$HERE/data}

case "$INIT_MODE" in
  base) MODEL_PATH=$BASE_MODEL ;;
  sft) MODEL_PATH=$SFT_MODEL ;;
  *) echo "INIT_MODE must be base or sft" >&2; exit 2 ;;
esac

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "Model config not found: $MODEL_PATH" >&2
  exit 2
fi
if [ -f "$MODEL_PATH/model.safetensors" ] && [ ! -r "$MODEL_PATH/model.safetensors" ]; then
  echo "Model weights are not readable by $(id -un): $MODEL_PATH/model.safetensors" >&2
  echo "Ask the checkpoint owner to grant group read permission; the recipe will not modify source permissions." >&2
  exit 2
fi
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/validation.parquet" ]; then
  echo "Prepared data missing under $DATA_DIR" >&2
  exit 2
fi
if [ ! -f "$REWARD_PATH" ]; then
  echo "Reward implementation not found: $REWARD_PATH" >&2
  exit 2
fi

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
# Use project-scoped caches even when activate_verl inherited generic cache
# variables. OTB_* variables remain available for explicit relocation.
export UV_CACHE_DIR=${OTB_UV_CACHE_DIR:-$PROJECT_SCRATCH/cache/uv}
export HF_HOME=${OTB_HF_HOME:-$PROJECT_SCRATCH/cache/huggingface}
export TMPDIR=${OTB_TMPDIR:-$PROJECT_SCRATCH/cache/tmp}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_DEDUP_LOGS=0
mkdir -p \
  "$TMPDIR" \
  "$UV_CACHE_DIR" \
  "$HF_HOME" \
  "$PROJECT_SCRATCH/outputs/checkpoints" \
  "$PROJECT_SCRATCH/eval" \
  "$PROJECT_SCRATCH/logs"

NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-40960}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-49152}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 256))
# vLLM and the FSDP actor have a large measured log-probability gap even at
# step 1. Use verl's decoupled token-level truncated importance sampling to
# correct the behavior-policy mismatch instead of treating the two backends as
# the same policy. Set ROLLOUT_IS=null only for an explicit ablation.
ROLLOUT_IS=${ROLLOUT_IS:-token}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}
ROLLOUT_IS_BATCH_NORMALIZE=${ROLLOUT_IS_BATCH_NORMALIZE:-false}
ACTOR_LR=${ACTOR_LR:-5e-7}
USE_KL_LOSS=${USE_KL_LOSS:-true}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}

case "$USE_KL_LOSS" in
  true|false) ;;
  *) echo "USE_KL_LOSS must be true or false" >&2; exit 2 ;;
esac

if [ "$RUN_MODE" = smoke ]; then
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
  PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
  ROLLOUT_N=${ROLLOUT_N:-2}
  TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
  EXTRA_TRAINER=(trainer.total_training_steps=2 trainer.save_freq=-1 trainer.test_freq=-1)
elif [ "$RUN_MODE" = train ]; then
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
  PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
  ROLLOUT_N=${ROLLOUT_N:-4}
  TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
  SAVE_FREQ=${SAVE_FREQ:-10}
  TEST_FREQ=${TEST_FREQ:-10}
  EXTRA_TRAINER=(trainer.save_freq="$SAVE_FREQ" trainer.test_freq="$TEST_FREQ")
else
  echo "RUN_MODE must be smoke or train" >&2
  exit 2
fi

EXPERIMENT_NAME=${EXPERIMENT_NAME:-otb_${INIT_MODE}_${RUN_MODE}}
# Checkpoints are large and the source SFT run already exposed severe NFS
# writeback stalls. Keep them under this project's node-local scratch tree.
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_SCRATCH/outputs/checkpoints/$EXPERIMENT_NAME}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
if [ "$RUN_MODE" = train ]; then
  SAVE_GENERATIONS=${SAVE_GENERATIONS:-true}
else
  SAVE_GENERATIONS=${SAVE_GENERATIONS:-false}
fi
if [ "$SAVE_GENERATIONS" = true ]; then
  ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$PROJECT_SCRATCH/outputs/rollouts/$EXPERIMENT_NAME}
  VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-$PROJECT_SCRATCH/eval/validation/$EXPERIMENT_NAME}
  mkdir -p "$ROLLOUT_DATA_DIR" "$VALIDATION_DATA_DIR"
else
  ROLLOUT_DATA_DIR=null
  VALIDATION_DATA_DIR=null
fi
export OTB_JUDGE_BASE_URL=${OTB_JUDGE_BASE_URL:-http://127.0.0.1:8094/v1}
export OTB_JUDGE_MODEL=${OTB_JUDGE_MODEL:-judge}
export OTB_JUDGE_AUDIT_LOG=${OTB_JUDGE_AUDIT_LOG:-$PROJECT_SCRATCH/eval/$EXPERIMENT_NAME/judge_rewards.jsonl}
mkdir -p "$OUTPUT_DIR" "$(dirname "$OTB_JUDGE_AUDIT_LOG")"

JUDGE_MODELS_URL=${OTB_JUDGE_BASE_URL%/v1}/v1/models
if ! curl --fail --silent --max-time 10 "$JUDGE_MODELS_URL" | grep -q "$OTB_JUDGE_MODEL"; then
  echo "Judge is not ready at $JUDGE_MODELS_URL" >&2
  exit 2
fi

cd "$REPO"
"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is="$ROLLOUT_IS" \
  algorithm.rollout_correction.rollout_is_threshold="$ROLLOUT_IS_THRESHOLD" \
  algorithm.rollout_correction.rollout_is_batch_normalize="$ROLLOUT_IS_BATCH_NORMALIZE" \
  algorithm.rollout_correction.bypass_mode=False \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/validation.parquet" \
  data.custom_cls.path="$HERE/otb_dataset.py" \
  data.custom_cls.name=OTBTumorBoardDataset \
  data.image_key=images \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=4 \
  data.truncation=error \
  +data.cache_dir="$PROJECT_SCRATCH/cache/data" \
  `# Do not impose a repository-wide pixel policy here. Public Task 1 evaluation` \
  `# uses each model's native processor, while the historical Qwen SFT training` \
  `# domain used max_pixels=589824. Experiment launchers must select and record` \
  `# either protocol explicitly; current agent-loop code threads data-level` \
  `# mm_processor_kwargs through token building, vLLM rollout, and actor rescoring.` \
  `# With current Transformers use size.longest_edge, not the legacy max_pixels` \
  `# spelling, which is silently ignored by the HF Qwen2.5-VL processor.` \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  reward.num_workers=8 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  +actor_rollout_ref.rollout.limit_images=39 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.multi_stage_wake_up=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=otb_grpo \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  ray_kwargs.ray_init.runtime_env.py_executable="$PYTHON_BIN" \
  +ray_kwargs.ray_init.runtime_env.env_vars.OTB_JUDGE_BASE_URL="$OTB_JUDGE_BASE_URL" \
  +ray_kwargs.ray_init.runtime_env.env_vars.OTB_JUDGE_MODEL="$OTB_JUDGE_MODEL" \
  +ray_kwargs.ray_init.runtime_env.env_vars.OTB_JUDGE_MAX_TOKENS="'${OTB_JUDGE_MAX_TOKENS:-512}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.OTB_JUDGE_TIMEOUT="'${OTB_JUDGE_TIMEOUT:-300}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.OTB_JUDGE_RETRIES="'${OTB_JUDGE_RETRIES:-3}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.OTB_JUDGE_AUDIT_LOG="$OTB_JUDGE_AUDIT_LOG" \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME="$HF_HOME" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR="$TMPDIR" \
  "${EXTRA_TRAINER[@]}" \
  "$@"
