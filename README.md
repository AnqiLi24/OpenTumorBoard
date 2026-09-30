# OpenTumorBoard

**[Paper](https://arxiv.org/abs/2609.32810) · [Dataset](https://huggingface.co/datasets/al1219/OpenTumorBoard) · [Leaderboard](https://huggingface.co/spaces/al1219/OpenTumorBoard-Leaderboard)**

OpenTumorBoard is a benchmark of real multidisciplinary tumor board discussions,
built from 219 publicly recorded meetings: 611 patient cases, 19,157 specialist
turns, and 16,215 questions put to specialists during the meetings. It evaluates
LLMs in two settings:

- **Specialist Turn**: answer a question the board actually put to one
  specialist, compared against the specialist's recorded answer.
- **Board Simulation**: given the case summary and the slides, simulate the
  whole multidisciplinary discussion and reach a conclusion, compared against
  the board's recorded conclusion.

This repository contains the evaluation code, automated curation pipeline
and training code. Benchmark records are distributed through
the Hugging Face dataset linked above, not bundled in this repository. Run every
command from the repository root.

## Interactive leaderboard

[Open the interactive leaderboard](https://huggingface.co/spaces/al1219/OpenTumorBoard-Leaderboard) to explore the
aggregate results for the paper's 14 Board Simulation and 9 Specialist Turn
baseline model configurations.

The leaderboard is hosted on Hugging Face and is not bundled in this repository.

## Installation

```bash
pip install -r envs/evaluation.txt
```

## Data

Sign in to [the Hugging Face dataset](https://huggingface.co/datasets/al1219/OpenTumorBoard)
and accept its access conditions. Download its `data/` directory into the root
of this repository, preserving the directory structure. Access is automatically
approved; the dataset's access controls remain on Hugging Face.

The hosted release contains processed records, slide captions and source
references. It includes caption-input manifests, not image-input manifests or
the underlying media. This public code repository does not mirror those records.

| File | Contents |
|---|---|
| `data/task1_{train,validation,test}.jsonl` | Board Simulation cases (366 / 61 / 184) |
| `data/task2_{train,validation,test}.jsonl` | Specialist Turn questions (9,731 / 1,640 / 4,844) |
| `data/test_inputs/` | prepared caption-input test manifests |
| `data/split.json` | recording-level split, with the YouTube URL of every recording |
| `data/slides.jsonl`, `data/slides_index.jsonl` | source time of every slide, and the slides of every case |

Each case holds the case summary, the slide captions and the board's
conclusion; each question holds its type, the target specialty and the
specialist's answer. The files ship gzip-compressed and the code reads them as
they are; `gunzip -k data/*.jsonl.gz data/test_inputs/*.jsonl.gz` unpacks them
for reading by hand. Recordings, slide images and transcripts are not
redistributed. To obtain the slide images, run the curation pipeline below on
the recordings in `data/split.json` and link its output:

```bash
ln -s data/processed slides
```

## Evaluation

### Caption-input generation

The commands below use the caption manifests available in the hosted dataset.
They generate model responses; they do not run the image-based judge or reproduce
the paper's image-input settings.

Serve the model to evaluate and a judge model behind OpenAI-compatible
endpoints, for example with vLLM:

```bash
vllm serve <model> --served-model-name my_model --port 8000
vllm serve <judge-model> --served-model-name judge --port 8100 --max-model-len 65536 \
  --limit-mm-per-prompt '{"image":32}' \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
```

Point the commands below at those two endpoints:

```bash
export MODEL_URL=...   # for example http://127.0.0.1:8000/v1
export JUDGE_URL=...   # for example http://127.0.0.1:8100/v1
```

**Board Simulation**

```bash
python -m scripts.evaluation.generation.run_vllm_simulation_evaluation --model my_model --base-url $MODEL_URL --manifest data/test_inputs/board_simulation.caption.jsonl --prompt evaluation/prompts/board_simulation_caption.txt --output-dir runs/board
```

**Specialist Turn**

```bash
python -m scripts.evaluation.generation.run_vllm_qa_evaluation --model my_model --base-url $MODEL_URL --manifest data/test_inputs/specialist_turn.caption.jsonl --prompt evaluation/prompts/specialist_turn_caption.txt --output-dir runs/turn
```

### Image-based judging and multimodal evaluation

The paper's judge requires the slide images and an image-input manifest aligned
with the released case IDs. These are **not** included in the hosted caption-only
release. Prepare the images from the source recordings and the corresponding
manifest before running the following judging commands; caption downloads alone
are not sufficient. The default slide manifest path is
`data/test_inputs/board_simulation.image.jsonl`.

```bash
python -m scripts.evaluation.judge.build_llm_judge_batch --task1-run my_model=runs/board/responses.jsonl --slide-manifest /path/to/matching/board_simulation.image.jsonl --output-dir judge/board
python -m scripts.evaluation.judge.run_vllm_llm_judge_batch --batch judge/board/task1.batch.jsonl --base-url $JUDGE_URL
python -m scripts.evaluation.judge.summarize_llm_judge_batch --batch-dir judge/board

python -m scripts.evaluation.judge.build_llm_judge_batch --task2-run my_model=runs/turn/responses.jsonl --slide-manifest /path/to/matching/board_simulation.image.jsonl --output-dir judge/turn
python -m scripts.evaluation.judge.run_vllm_llm_judge_batch --batch judge/turn/task2.batch.jsonl --base-url $JUDGE_URL
python -m scripts.evaluation.judge.summarize_llm_judge_batch --batch-dir judge/turn
```

The scores are written to `judge/*/summary.json`: conclusion alignment (1–5)
for Board Simulation, and clinical equivalence (1–5) with critical-error and
unsupported-claim rates for Specialist Turn.

- **Text-only models** read the slide captions instead of the images: use
  `scripts.evaluation.generation.run_vllm_simulation_evaluation` for Board
  Simulation, and add `--manifest data/test_inputs/specialist_turn.caption.jsonl
  --prompt evaluation/prompts/specialist_turn_caption.txt` for Specialist Turn.
- **Hosted models**: `run_vllm_simulation_multimodal` takes
  `--provider openai|openrouter|anthropic --api-key-file <file>`; the other
  generation scripts take the `--base-url` of any OpenAI-compatible API with
  `--api-key-file <file>`.
- **Four decisions**: `--task1-rubric four_decisions` scores therapy, surgery,
  next action and clinical trial separately.
- **ROUGE-L and BERTScore**: `python -m scripts.evaluation.metrics.text_metrics
  --task board --run runs/board/responses.jsonl` (or `--task turn`).

The judge sees every slide as an image, so judging needs the slide images even
for a text-only model. `evaluation/paper_configurations.json` lists the input,
prompt and decoding settings of every model in the paper.

## Curation pipeline

Search and filtering outputs are generated locally and are not included in this
repository.

```bash
conda env create -f environment.yml && conda activate mtb
python src/search.py                           # search YouTube for tumor board recordings
bash scripts/pipeline/obtain_video.sh          # download videos and transcripts to data/videos
python src/filter_case_presentation.py         # keep recordings that discuss patient cases
bash scripts/pipeline/pipeline_par.sh          # run the pipeline, one folder per recording in data/processed
bash scripts/pipeline/pipeline_par_rephrase.sh data/processed   # rewrite the questions
python src/build_benchmark.py                  # benchmark files in the format of data/, in data/custom
```

To process a single recording, replace the first two steps with
`python src/download.py --url <URL> --output-dir data/videos`. The filter step
still has to run: the pipeline only processes recordings it marks as case
presentations.

The pipeline runs speech recognition, speaker diarization, case segmentation,
slide extraction, role inference, alignment of utterances to slides, evidence
extraction, question generation and conclusion generation. It needs the
slide-extraction weights `sam2.1_hiera_large.pt` and
`groundingdino_swint_ogc.pth` in `scripts/checkpoints/`, and these variables:

```bash
export HF_TOKEN=...                   # pyannote speaker diarization
export AZURE_OPENAI_ENDPOINT=...      # Azure OpenAI endpoint for the LLM stages
export AZURE_OPENAI_DEPLOYMENT=...    # chat deployment used by the LLM stages
export GDINO_CONFIG=/path/to/GroundingDINO_SwinT_OGC.py
```

`scripts/pipeline/setup_mtb_env.sh` installs the environment step by step.

`src/build_benchmark.py` writes `task1.jsonl`, `task2.jsonl` and
`test_inputs/` in the format of `data/`. Evaluate on them by passing
`--manifest` to the generation scripts and `--task1-manifest` or
`--task2-manifest` with `--slide-manifest` to the judge, and link
`data/processed` as `slides`. The released benchmark was de-leaked against each
decision point following the protocol in `evaluation/deleak/`; files built here
are not, so their scores are not comparable to the released test split.

## Training

The training target is the recorded discussion, which is read from the
pipeline output. Build the finetuning data from the released split:

```bash
for split in train validation; do
  python training/build_sft_data.py --split $split --records data/task1_$split.jsonl \
    --processed-root data/processed --image-root "$PWD" --out training/data/board_sft_$split.jsonl
done
```

**Supervised finetuning** uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
with `training/sft_qwen25vl3b.yaml` (environment `envs/sft_llamafactory.txt`).

**Reinforcement learning** uses [verl](https://github.com/verl-project/verl) at
commit `890dfc3e` with `training/verl_890dfc3e.patch` applied (environment
`envs/rl_verl.txt`). Copy `training/verl_otb/` into verl's `examples/` and run
its scripts there: `prepare_data.py` builds the training data, `start_judge.sh`
serves the reward judge, `precompute_reference_scope.py` fills its cache, and
`run_rl.sh` runs Dr.GRPO with the reward in `reward.py`. Each script lists its
settings at the top.
