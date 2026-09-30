#!/usr/bin/env bash
# One-click QA-answer rephrase pass over every processed video directory.
#
# Rewrites each expert ANSWER into a clean, declarative clinical statement and
# drops "non-answers" (deferrals / "I don't know" / pure speculation), then
# rebuilds sharegpt from the rephrased pairs. Purely additive — the original
# task_case_*.json and sharegpt.json are NEVER modified, and no existing
# pipeline script is touched.
#
# Outputs per video dir:
#   task_case_XXX_rephrased.json   (rephrased + filtered QA pairs)
#   sharegpt_rephrased.json        (full sharegpt rebuilt from those pairs)
#
# The heavy lifting (multi-dir discovery, concurrency, live progress bar,
# skip-existing) is handled inside rephrase_qa.py; this wrapper just points it
# at the right root and conda env.
#
# Usage:
#   scripts/pipeline/pipeline_par_rephrase.sh [ROOT_DIR]
#   WORKERS=24 scripts/pipeline/pipeline_par_rephrase.sh
#   FRESH=1 scripts/pipeline/pipeline_par_rephrase.sh        # re-run dirs already done

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROOT_DIR="${1:-data/processed}"
WORKERS="${WORKERS:-16}"
CONDA_ENV="${CONDA_ENV:-mtb}"

# By default skip dirs that already have sharegpt_rephrased.json; FRESH=1 redoes.
SKIP_FLAG="--skip-existing"
[[ "${FRESH:-0}" == "1" ]] && SKIP_FLAG=""

cd "$PROJECT_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  QA rephrase pass"
echo "  Root    : ${ROOT_DIR}"
echo "  Workers : ${WORKERS}"
echo "  Env     : ${CONDA_ENV}"
[[ -n "$SKIP_FLAG" ]] && echo "  Resume  : skipping dirs with sharegpt_rephrased.json" \
                       || echo "  Resume  : OFF (re-running every dir)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run --no-capture-output -n "$CONDA_ENV" \
    python src/task_generator/rephrase_qa.py "$ROOT_DIR" \
        --workers "$WORKERS" $SKIP_FLAG

echo "Done. Rephrased outputs: task_case_*_rephrased.json + ${ROOT_DIR%/}/<dir>/sharegpt_rephrased.json"
