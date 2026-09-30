#!/usr/bin/env bash
# Parallel tumor board pipeline — 4 workers, one per A100 GPU.
# Progress + ETA printed to terminal every 30 s; full logs in logs/worker_N.log.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VIDEO_DIR="${VIDEO_DIR:-data/videos}"
ROOT_DIR="${ROOT_DIR:-data/processed}"
: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this script}"
FILTER_JSON="${PROJECT_ROOT}/src/case_presentation_filter.json"
LOG_DIR="${PROJECT_ROOT}/logs"
NUM_WORKERS=4
MONITOR_INTERVAL=30   # seconds between progress updates

mkdir -p "$ROOT_DIR" "$LOG_DIR"

# ── Shared progress tracking ──────────────────────────────────────────────────
PROGRESS_DONE="${LOG_DIR}/progress.done"
PROGRESS_ERROR="${LOG_DIR}/progress.error"
: > "$PROGRESS_DONE"
: > "$PROGRESS_ERROR"
START_TIME=$(date +%s)

# ── Extract video list ────────────────────────────────────────────────────────
mapfile -t ALL_VIDEOS < <(python3 - <<PYEOF
import json, os

with open("${FILTER_JSON}") as f:
    data = json.load(f)

for name, info in data.items():
    if not info.get("is_case_presentation"):
        continue
    mp4_name = name if name.endswith(".mp4") else name + ".mp4"
    if os.path.exists(os.path.join("${VIDEO_DIR}", mp4_name)):
        print(mp4_name)
    elif os.path.exists(os.path.join("${VIDEO_DIR}", name)):
        print(name)
PYEOF
)

TOTAL=${#ALL_VIDEOS[@]}
echo "Found ${TOTAL} case-presentation videos on disk."

# ── Round-robin distribution ──────────────────────────────────────────────────
declare -a WORKER_VIDEOS_0=() WORKER_VIDEOS_1=() WORKER_VIDEOS_2=() WORKER_VIDEOS_3=()
for i in "${!ALL_VIDEOS[@]}"; do
    case $(( i % NUM_WORKERS )) in
        0) WORKER_VIDEOS_0+=("${ALL_VIDEOS[$i]}") ;;
        1) WORKER_VIDEOS_1+=("${ALL_VIDEOS[$i]}") ;;
        2) WORKER_VIDEOS_2+=("${ALL_VIDEOS[$i]}") ;;
        3) WORKER_VIDEOS_3+=("${ALL_VIDEOS[$i]}") ;;
    esac
done

printf "Worker 0: %3d videos  (GPU 0)\n" "${#WORKER_VIDEOS_0[@]}"
printf "Worker 1: %3d videos  (GPU 1)\n" "${#WORKER_VIDEOS_1[@]}"
printf "Worker 2: %3d videos  (GPU 2)\n" "${#WORKER_VIDEOS_2[@]}"
printf "Worker 3: %3d videos  (GPU 3)\n" "${#WORKER_VIDEOS_3[@]}"
echo ""

# ── Helpers ───────────────────────────────────────────────────────────────────
fmt_dur() {
    local s=$1
    printf '%02d:%02d:%02d' $(( s/3600 )) $(( (s%3600)/60 )) $(( s%60 ))
}

# ── Progress monitor (runs in background, writes to terminal) ─────────────────
run_monitor() {
    local total=$1 start=$2

    while true; do
        sleep "$MONITOR_INTERVAL"

        local now elapsed done_count error_count remaining pct eta_str
        now=$(date +%s)
        elapsed=$(( now - start ))
        done_count=$(wc -l < "$PROGRESS_DONE" 2>/dev/null | tr -d ' ')
        error_count=$(wc -l < "$PROGRESS_ERROR" 2>/dev/null | tr -d ' ')
        remaining=$(( total - done_count - error_count ))

        if [[ $total -gt 0 ]]; then
            pct=$(( done_count * 100 / total ))
        else
            pct=0
        fi

        if [[ $done_count -gt 0 && $elapsed -gt 0 ]]; then
            local eta=$(( remaining * elapsed / done_count ))
            eta_str=$(fmt_dur $eta)
        else
            eta_str="--:--:--"
        fi

        printf '\n━━━ [%s]  %d/%d done (%d%%)  |  %d errors  |  Elapsed %s  |  ETA %s\n' \
            "$(date '+%H:%M:%S')" \
            "$done_count" "$total" "$pct" \
            "$error_count" \
            "$(fmt_dur $elapsed)" \
            "$eta_str"

        local i
        for i in 0 1 2 3; do
            local cur_file="${LOG_DIR}/worker_${i}.current"
            if [[ -f "$cur_file" ]]; then
                printf '  GPU %d → %s\n' "$i" "$(cat "$cur_file")"
            fi
        done
    done
}

# ── Per-video pipeline ────────────────────────────────────────────────────────
process_video() {
    local WID="$1"
    local VIDEO_NAME="$2"
    local VIDEO_PATH="${VIDEO_DIR}/${VIDEO_NAME}"
    local VIDEO_OUT="${ROOT_DIR}/${VIDEO_NAME}"

    local TRANSCRIPT_PATH="${VIDEO_OUT}/transcript.raw.json"
    local TRANSCRIPT_SPEAKER_PATH="${VIDEO_OUT}/transcript.with_speakers.json"
    local CASE_SEGMENTATION_PATH="${VIDEO_OUT}/case_segmentation.json"
    local SLIDES_PATH="${VIDEO_OUT}/slides.raw.json"
    local SPEAKER_PROFILES_PATH="${VIDEO_OUT}/speaker_profiles.json"
    local ALIGN_PATH="${VIDEO_OUT}/align.json"

    # Update the terminal monitor
    step() { printf '%s  [%s]\n' "$VIDEO_NAME" "$1" > "${LOG_DIR}/worker_${WID}.current"; }

    step "checking"

    if [[ -f "${VIDEO_OUT}/sharegpt.json" ]]; then
        echo "[SKIP] Already complete: ${VIDEO_NAME}"
        return 0
    fi

    mkdir -p "$VIDEO_OUT"
    echo "=== Processing: ${VIDEO_NAME} ==="

    step "ASR"
    if [[ ! -f "$TRANSCRIPT_PATH" ]]; then
        conda run -n mtb python src/asr.py \
            --video "$VIDEO_PATH" --output "$TRANSCRIPT_PATH"
    fi

    step "diarization"
    if [[ ! -f "$TRANSCRIPT_SPEAKER_PATH" ]]; then
        conda run -n mtb python src/diarize.py \
            --video "$VIDEO_PATH" --transcript "$TRANSCRIPT_PATH" \
            --output "$TRANSCRIPT_SPEAKER_PATH" --hf-token "$HF_TOKEN"
    fi

    step "case segmentation"
    if [[ ! -f "$CASE_SEGMENTATION_PATH" ]]; then
        conda run -n mtb python src/case.py \
            --transcript "$TRANSCRIPT_SPEAKER_PATH" --output "$CASE_SEGMENTATION_PATH"
    fi

    step "slides"
    if [[ ! -f "$SLIDES_PATH" ]]; then
        conda run -n mtb bash scripts/pipeline/run_slides.sh \
            -VIDEO "$VIDEO_PATH" -OUT "${VIDEO_OUT}/" \
            -CASE_SEG "$CASE_SEGMENTATION_PATH" \
            -TRANSCRIPT "$TRANSCRIPT_SPEAKER_PATH"
    fi

    step "role inference"
    if [[ ! -f "$SPEAKER_PROFILES_PATH" ]]; then
        conda run -n mtb python src/role.py \
            --transcript "$TRANSCRIPT_SPEAKER_PATH" --output "$SPEAKER_PROFILES_PATH"
    fi

    step "alignment"
    if [[ ! -f "$ALIGN_PATH" ]]; then
        conda run -n mtb python src/align.py \
            --slides "$SLIDES_PATH" --transcript "$TRANSCRIPT_SPEAKER_PATH" \
            --output "$ALIGN_PATH" \
            --case_segmentation "$CASE_SEGMENTATION_PATH" \
            --speaker_profiles "$SPEAKER_PROFILES_PATH"
    fi

    step "evidence"
    conda run -n mtb python src/task_generator/evidence.py --dir "$VIDEO_OUT"

    step "atomic QA"
    conda run -n mtb python src/task_generator/atomic_qa.py --dir "$VIDEO_OUT"

    step "conclusion"
    conda run -n mtb python src/task_generator/conclusion.py --dir "$VIDEO_OUT"

    step "sharegpt"
    conda run -n mtb python src/task_generator/convert_sharegpt.py --dir "$VIDEO_OUT"

    echo "=== Done: ${VIDEO_NAME} ==="
}

# ── Worker ────────────────────────────────────────────────────────────────────
run_worker() {
    local WID="$1"
    shift
    local VIDEOS=("$@")

    export CUDA_VISIBLE_DEVICES="$WID"
    cd "$PROJECT_ROOT"

    echo "[Worker ${WID}] Starting — ${#VIDEOS[@]} videos on GPU ${WID}"

    local VIDEO_NAME
    for VIDEO_NAME in "${VIDEOS[@]}"; do
        if process_video "$WID" "$VIDEO_NAME"; then
            printf '%s\n' "$VIDEO_NAME" >> "$PROGRESS_DONE"
        else
            printf '%s\n' "$VIDEO_NAME" >> "$PROGRESS_ERROR"
            echo "[Worker ${WID}] ERROR processing: ${VIDEO_NAME}"
        fi
    done

    printf 'idle\n' > "${LOG_DIR}/worker_${WID}.current"
    echo "[Worker ${WID}] All done."
}

# ── Launch ────────────────────────────────────────────────────────────────────
run_monitor "$TOTAL" "$START_TIME" &
MONITOR_PID=$!

run_worker 0 "${WORKER_VIDEOS_0[@]+"${WORKER_VIDEOS_0[@]}"}" \
    >> "${LOG_DIR}/worker_0.log" 2>&1 &
PID0=$!
run_worker 1 "${WORKER_VIDEOS_1[@]+"${WORKER_VIDEOS_1[@]}"}" \
    >> "${LOG_DIR}/worker_1.log" 2>&1 &
PID1=$!
run_worker 2 "${WORKER_VIDEOS_2[@]+"${WORKER_VIDEOS_2[@]}"}" \
    >> "${LOG_DIR}/worker_2.log" 2>&1 &
PID2=$!
run_worker 3 "${WORKER_VIDEOS_3[@]+"${WORKER_VIDEOS_3[@]}"}" \
    >> "${LOG_DIR}/worker_3.log" 2>&1 &
PID3=$!

echo "Workers launched: PIDs ${PID0} ${PID1} ${PID2} ${PID3}"
echo "Full logs : ${LOG_DIR}/worker_{0,1,2,3}.log"
echo "Failed    : ${PROGRESS_ERROR}  (populated on error)"
echo ""

# ── Wait and summarise ────────────────────────────────────────────────────────
EXIT=0
wait $PID0 || { echo "Worker 0 exited non-zero"; EXIT=1; }
wait $PID1 || { echo "Worker 1 exited non-zero"; EXIT=1; }
wait $PID2 || { echo "Worker 2 exited non-zero"; EXIT=1; }
wait $PID3 || { echo "Worker 3 exited non-zero"; EXIT=1; }

kill "$MONITOR_PID" 2>/dev/null || true

DONE_FINAL=$(wc -l < "$PROGRESS_DONE" | tr -d ' ')
ERROR_FINAL=$(wc -l < "$PROGRESS_ERROR" | tr -d ' ')
ELAPSED=$(( $(date +%s) - START_TIME ))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf '  Done   : %d / %d\n'  "$DONE_FINAL"  "$TOTAL"
printf '  Errors : %d\n'       "$ERROR_FINAL"
printf '  Elapsed: %s\n'       "$(fmt_dur $ELAPSED)"
if [[ $ERROR_FINAL -gt 0 ]]; then
    printf '  Failed videos → %s\n' "$PROGRESS_ERROR"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $EXIT
