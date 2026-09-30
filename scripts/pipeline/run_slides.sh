#!/usr/bin/env bash
# Extract the slides of one recording.
# Run from the project root: bash scripts/pipeline/run_slides.sh

set -euo pipefail

# ── Parse command-line flags (override env vars) ─────────────────────────────
# Accepted: -VIDEO <path>  -OUT <dir>  -CASE_SEG <path>
#           -DETECTOR <cv|sam2|gsam2>  -FPS <float>  -SIM <float>
while [[ $# -gt 0 ]]; do
  case "$1" in
    -VIDEO)    VIDEO="$2";       shift 2 ;;
    -OUT)      OUT="$2";         shift 2 ;;
    -CASE_SEG) CASE_SEG="$2";   shift 2 ;;
    -DETECTOR) DETECTOR="$2";   shift 2 ;;
    -FPS)      SAMPLE_FPS="$2"; shift 2 ;;
    -TRANSCRIPT) TRANSCRIPT="$2"; shift 2 ;;
    -SIM)      SIM_THRESHOLD="$2"; shift 2 ;;
    *) echo "Unknown flag: $1  (valid: -VIDEO -OUT -CASE_SEG -DETECTOR -FPS -SIM)"; exit 1 ;;
  esac
done

# ── Config — edit these ────────────────────────────────────────────────────────

# Which video to test against. Default: the local copy in the repo root.
# VIDEO_NAME="Tumor Board： Case Presentations.mp4"
VIDEO_NAME="PET Tumor Board – 72-Year-Old with Rising PSA and MRI Showing PI-RADSv2.1 Categories 4 and 5.mp4"
VIDEO="${VIDEO:-data/videos/$VIDEO_NAME}"
CASE_SEG="${CASE_SEG:-}"
TRANSCRIPT="${TRANSCRIPT:-}"

# Output directory for this test run
OUT="${OUT:-./processed/$VIDEO_NAME}"

# Conda environment name
CONDA_ENV="${CONDA_ENV:-mtb}"

# Sample rate: 0.2 = one frame every 5 s (fast), 0.5 = every 2 s (default)
SAMPLE_FPS="${SAMPLE_FPS:-0.2}"

# Dedup threshold: higher → fewer slides kept
SIM_THRESHOLD="${SIM_THRESHOLD:-0.85}"

# CLIP pre-filter on full frames before detection (skips slow detector on non-slide frames).
# Score = sim("presentation slide") − sim("audience/room") on the full frame.
# Full-frame scores are lower than crop scores, so use a slightly negative value:
#   -0.02  keeps ~27% of frames (recommended)
#    0.0   keeps ~5%  (strict, may miss some slides)
# Leave empty to disable.
CLIP_VERIFY="${CLIP_VERIFY:--0.02}"

# Detector: cv (no checkpoint needed) | sam2 | gsam2 (text-prompt, most accurate)
DETECTOR="${DETECTOR:-gsam2}"

# ── Checkpoint paths (pre-filled, no need to change) ──────────────────────────
_CKPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/checkpoints"

SAM_CHECKPOINT="${SAM_CHECKPOINT:-${_CKPT_DIR}/sam2.1_hiera_large.pt}"

GDINO_CONFIG="${GDINO_CONFIG:-path/to/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GDINO_CHECKPOINT="${GDINO_CHECKPOINT:-${_CKPT_DIR}/groundingdino_swint_ogc.pth}"
GDINO_PROMPT="${GDINO_PROMPT:-entire projection screen . entire presentation slide . entire powerpoint slide}"
GDINO_BOX_THRESHOLD="${GDINO_BOX_THRESHOLD:-0.3}"

# Minimum slide area as a fraction of the full frame.
# Detected regions smaller than this are discarded.
MIN_SLIDE_AREA="${MIN_SLIDE_AREA:-0.10}"

# GPT-5.4 final verification: set to "true" to enable, anything else to disable.
# When enabled, slides are sent to GPT vision and only clinically useful ones are kept.
GPT_VERIFY="${GPT_VERIFY:-true}"

# Number of slide images packed into each GPT API call (reduces round-trips).
GPT_BATCH_SIZE="${GPT_BATCH_SIZE:-1}"

# ── Helpers ───────────────────────────────────────────────────────────────────

RUN="conda run -n $CONDA_ENV --no-capture-output"
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
section() { echo -e "\n${BOLD}${CYAN}=== $* ===${RESET}"; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────

section "Pre-flight"

if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: video not found: $VIDEO"
  echo "Set VIDEO=/path/to/your.mp4 before running."
  exit 1
fi

echo "Video    : $VIDEO"
echo "Output   : $OUT"
echo "Detector : $DETECTOR"
echo "FPS      : $SAMPLE_FPS  (sim_threshold=$SIM_THRESHOLD)"
[[ -n "$CLIP_VERIFY" ]] && echo "CLIP verify threshold: $CLIP_VERIFY"
echo "GPT verify: $GPT_VERIFY  (batch_size=$GPT_BATCH_SIZE)"

mkdir -p "$OUT"

# ── Build argument list ───────────────────────────────────────────────────────

ARGS=(
  --video           "$VIDEO"
  --output-dir      "$OUT"
  --detector        "$DETECTOR"
  --sample-fps      "$SAMPLE_FPS"
  --sim-threshold   "$SIM_THRESHOLD"
  --min-slide-area  "$MIN_SLIDE_AREA"
  --debug
)

[[ "$GPT_VERIFY" == "true" ]] && ARGS+=(--gpt-verify --gpt-batch-size "$GPT_BATCH_SIZE")

[[ -n "$CLIP_VERIFY" ]] && ARGS+=(--clip-verify-threshold "$CLIP_VERIFY")

if [[ "$DETECTOR" == "gsam2" ]]; then
  for var in SAM_CHECKPOINT GDINO_CONFIG GDINO_CHECKPOINT; do
    [[ -z "${!var}" ]] && { echo "ERROR: $var must be set for --detector gsam2"; exit 1; }
  done
  ARGS+=(
    --sam-checkpoint   "$SAM_CHECKPOINT"
    --gdino-config     "$GDINO_CONFIG"
    --gdino-checkpoint "$GDINO_CHECKPOINT"
    --gdino-prompt     "$GDINO_PROMPT"
    --gdino-box-threshold "$GDINO_BOX_THRESHOLD"
  )
elif [[ "$DETECTOR" != "cv" ]]; then
  [[ -z "$SAM_CHECKPOINT" ]] && { echo "ERROR: SAM_CHECKPOINT must be set for --detector $DETECTOR"; exit 1; }
  ARGS+=(--sam-checkpoint "$SAM_CHECKPOINT")
fi

if [[ -n "$CASE_SEG" ]]; then
  ARGS+=(--case-seg "$CASE_SEG")
fi

if [[ -n "$TRANSCRIPT" ]]; then
  ARGS+=(--transcript "$TRANSCRIPT")
fi

# ── Run ───────────────────────────────────────────────────────────────────────

section "Running slides"
$RUN python src/slides "${ARGS[@]}"

# ── Results ───────────────────────────────────────────────────────────────────

section "Results"

N_SLIDES=$(SLIDES_JSON="$OUT/slides.raw.json" python -c "import json,os; d=json.load(open(os.environ['SLIDES_JSON'])); print(len(d['slides']))")
N_FRAMES=$(ls "$OUT/frames/" 2>/dev/null | wc -l | tr -d ' ')

echo "Unique slides extracted : $N_SLIDES"
echo "Frame images written    : $N_FRAMES"
echo "Output directory        : $OUT"
echo ""
echo "Slide JSON (first 3 entries):"
SLIDES_JSON="$OUT/slides.raw.json" python -c "
import json, os
data = json.load(open(os.environ['SLIDES_JSON']))
for s in data['slides'][:3]:
    print(f\"  {s['slide_id']}  {s['start_sec']:.1f}s – {s['end_sec']:.1f}s  {s['frame_path']}\")
print('  ...' if len(data['slides']) > 3 else '')
"

echo ""
echo "Cropped slide frames : $OUT/frames/"
echo "Debug bbox overlays  : $OUT/debug_frames/  (one per sampled frame)"
echo ""
echo "To browse slides quickly:"
echo "  ls $OUT/frames/"
echo ""
echo "Done."
