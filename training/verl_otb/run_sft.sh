#!/usr/bin/env bash
# Checkpoint-wise OTB SFT sweep using the same verl FSDP model path as GRPO.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

RUN_MODE=${RUN_MODE:-smoke}  # smoke | sweep
GPU_LIST=${GPU_LIST:-1,2,3}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to a local copy of the base model}
PYTHON_BIN=${PYTHON_BIN:-$REPO/.venv/bin/python}
DATA_DIR=${DATA_DIR:-$HERE/sft_data}
PROJECT_SCRATCH=${PROJECT_SCRATCH:-$HERE/outputs/sft_verl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-otb_sft}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_SCRATCH/outputs/$EXPERIMENT_NAME}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-$PROJECT_SCRATCH/hf_checkpoints/$EXPERIMENT_NAME}

IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
NPROC=${#GPU_IDS[@]}
if [ "$NPROC" -ne 3 ]; then
  echo "This recipe is calibrated for exactly three GPUs; got GPU_LIST=$GPU_LIST" >&2
  exit 2
fi
if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "Base model not found: $MODEL_PATH" >&2
  exit 2
fi
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/validation.parquet" ]; then
  echo "SFT parquet data missing under $DATA_DIR" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=$GPU_LIST
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
export HF_HOME=${OTB_HF_HOME:-$PROJECT_SCRATCH/cache/huggingface}
export UV_CACHE_DIR=${OTB_UV_CACHE_DIR:-$PROJECT_SCRATCH/cache/uv}
export TMPDIR=${OTB_TMPDIR:-$PROJECT_SCRATCH/cache/tmp}
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "$OUTPUT_DIR" "$SNAPSHOT_DIR" "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR" "$PROJECT_SCRATCH/logs"

TRAIN_BATCH_SIZE=12
STEPS_PER_EPOCH=30
if [ "$RUN_MODE" = smoke ]; then
  TOTAL_EPOCHS=1
  TRAIN_MAX_SAMPLES=12
  VAL_MAX_SAMPLES=12
  NUM_WORKERS=0
  RESUME_MODE=disable
  TRAINER_ARGS=(
    trainer.total_training_steps=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.max_ckpt_to_keep=1
    'checkpoint.save_contents=[]'
    'checkpoint.load_contents=[]'
  )
elif [ "$RUN_MODE" = sweep ]; then
  TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
  TRAIN_MAX_SAMPLES=-1
  VAL_MAX_SAMPLES=-1
  NUM_WORKERS=4
  RESUME_MODE=auto
  TRAINER_ARGS=(
    trainer.total_training_steps=null
    trainer.save_freq=after_each_epoch
    trainer.test_freq=after_each_epoch
    trainer.max_ckpt_to_keep=2
    'checkpoint.save_contents=[model,optimizer,extra,hf_model]'
    'checkpoint.load_contents=[model,optimizer,extra]'
  )
else
  echo "RUN_MODE must be smoke or sweep" >&2
  exit 2
fi

archive_selected_checkpoints() {
  local epoch step source target model_file tracker_file completed_step
  tracker_file="$OUTPUT_DIR/latest_checkpointed_iteration.txt"
  if [ ! -f "$tracker_file" ]; then
    return
  fi
  completed_step=$(tr -cd '0-9' < "$tracker_file")
  if [ -z "$completed_step" ]; then
    return
  fi
  for epoch in ${ARCHIVE_EPOCHS:-1 2 3 5 8 10 15 20 30 40 50}; do
    if [ "$epoch" -gt "$TOTAL_EPOCHS" ]; then
      continue
    fi
    step=$((epoch * STEPS_PER_EPOCH))
    if [ "$completed_step" -lt "$step" ]; then
      continue
    fi
    source="$OUTPUT_DIR/global_step_$step/huggingface"
    target="$SNAPSHOT_DIR/epoch_$epoch"
    if [ -e "$target/.complete" ]; then
      continue
    fi
    model_file=$(find "$source" -maxdepth 1 -type f \( -name 'model.safetensors' -o -name 'model.safetensors.index.json' \) -print -quit 2>/dev/null || true)
    if [ -z "$model_file" ]; then
      continue
    fi
    if [ -e "$target" ]; then
      echo "Refusing to replace incomplete snapshot: $target" >&2
      continue
    fi
    rm -rf "$target.tmp"
    mkdir -p "$target.tmp"
    cp -al "$source"/. "$target.tmp"/
    touch "$target.tmp/.complete"
    mv "$target.tmp" "$target"
    echo "Archived epoch $epoch HF checkpoint -> $target"
  done
}

if [ "$RUN_MODE" = sweep ]; then
  (
    while true; do
      archive_selected_checkpoints
      sleep 15
    done
  ) &
  ARCHIVER_PID=$!
  trap 'kill "$ARCHIVER_PID" 2>/dev/null || true' EXIT
fi

cd "$REPO"
set +e
"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC" \
  -m verl.trainer.sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/validation.parquet" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu=1 \
  data.train_max_samples="$TRAIN_MAX_SAMPLES" \
  data.val_max_samples="$VAL_MAX_SAMPLES" \
  data.max_length=65536 \
  data.max_token_len_per_gpu=49152 \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.use_dynamic_bsz=True \
  data.num_workers="$NUM_WORKERS" \
  data.custom_cls.path="$HERE/otb_sft_dataset.py" \
  data.custom_cls.name=OTBTumorBoardSFTDataset \
  +data.image_key=images \
  +data.apply_chat_template_kwargs.processor_kwargs.size.longest_edge=589824 \
  +data.apply_chat_template_kwargs.processor_kwargs.size.shortest_edge=3136 \
  data.enable_thinking_default=null \
  data.ignore_input_ids_mismatch=False \
  model.path="$MODEL_PATH" \
  model.trust_remote_code=True \
  model.use_remove_padding=True \
  model.use_liger=False \
  model.use_fused_kernels=True \
  model.fused_kernel_options.impl_backend=torch \
  model.enable_gradient_checkpointing=True \
  model.freeze_vision_tower=True \
  model.freeze_multi_modal_projector=True \
  model.expected_trainable_parameters=3085938688 \
  +model.override_config.attn_implementation=flash_attention_2 \
  engine=fsdp \
  engine.strategy=fsdp2 \
  engine.fsdp_size=-1 \
  engine.ulysses_sequence_parallel_size=1 \
  engine.model_dtype=fp32 \
  engine.dtype=bfloat16 \
  engine.use_torch_compile=True \
  engine.param_offload=False \
  engine.optimizer_offload=False \
  optim=fsdp \
  optim.lr=1.0e-5 \
  optim.weight_decay=0.0 \
  'optim.betas=[0.9,0.999]' \
  optim.clip_grad=1.0 \
  optim.lr_warmup_steps_ratio=0.03 \
  optim.lr_scheduler_type=cosine \
  optim.min_lr_ratio=0.0 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.logger='["console"]' \
  trainer.project_name=otb_sft_sweep \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NPROC" \
  trainer.nnodes=1 \
  "${TRAINER_ARGS[@]}" \
  "$@"
RC=$?
set -e

if [ "$RUN_MODE" = sweep ]; then
  archive_selected_checkpoints
  kill "$ARCHIVER_PID" 2>/dev/null || true
  wait "$ARCHIVER_PID" 2>/dev/null || true
fi
exit "$RC"
