"""Build benchmark files from the pipeline output, in the format of data/.

Reads every <recording>/task_case_*_rephrased.json under --processed-root and
writes task1.jsonl (Board Simulation cases), task2.jsonl (Specialist Turn
questions) and the four test_inputs files that the evaluation scripts read.
A case enters Board Simulation only if at least one of its questions survived
the rephrase pass, the rule used for the released benchmark. Image paths are
written as slides/<recording>/frames/..., so link the pipeline output as in the
README (ln -s data/processed slides).

The released benchmark was de-leaked against each decision point before
release (evaluation/deleak/); files built here are not, so their scores are
not comparable to the released test split.

usage: python src/build_benchmark.py --processed-root data/processed --output-dir data/custom
"""

import argparse
import json
from pathlib import Path

NO_SLIDES = "No slide descriptions are available."


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def case_bounds(recording):
    segmentation = json.loads((recording / "case_segmentation.json").read_text(encoding="utf-8"))
    return {c["case_id"]: (c.get("start_sec"), c.get("end_sec")) for c in segmentation.get("cases") or []}


def board_user(case, heading):
    return f"CASE SUMMARY:\n{case['case_summary']}\n\n{heading}:\n{case['slides'] or NO_SLIDES}"


def turn_user(qa, heading):
    return (f"TARGET SPECIALIST ROLE:\n{qa['target_specialist_role']}\n\n"
            f"CASE SUMMARY:\n{qa['case_summary']}\n\n{heading}:\n{qa['slides']}\n\n"
            f"QUESTION:\n{qa['question']}")


def sharegpt(record_id, prompt_version, prompts, metadata, user, images=None):
    row = {
        "schema_version": 1,
        "format": "sharegpt",
        "id": record_id,
        "prompt_version": prompt_version,
        "metadata": metadata,
        "conversations": [
            {"from": "system", "value": prompts[prompt_version]},
            {"from": "human", "value": user},
        ],
    }
    if images is not None:
        row["images"] = images
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/custom"))
    parser.add_argument("--prompt-dir", type=Path, default=Path("evaluation/prompts"))
    parser.add_argument("--benchmark-version", default="custom")
    args = parser.parse_args()

    prompts = {
        name: (args.prompt_dir / f"{name}.txt").read_text(encoding="utf-8").strip()
        for name in ("board_simulation_image", "board_simulation_caption",
                     "specialist_turn_image", "specialist_turn_caption")
    }
    task_paths = sorted(args.processed_root.glob("*/task_case_*_rephrased.json"))
    if not task_paths:
        raise SystemExit(f"no task_case_*_rephrased.json under {args.processed_root}; "
                         "run scripts/pipeline/pipeline_par_rephrase.sh first")

    video_uids = {}
    cases, questions, images = [], [], {}
    bounds = {}
    for task_path in task_paths:
        recording = task_path.parent
        case_id = task_path.name[len("task_"):-len("_rephrased.json")]
        task = json.loads(task_path.read_text(encoding="utf-8"))
        pairs = task.get("qa_pairs") or []
        if not pairs:
            continue
        if not task.get("case_summary"):
            raise SystemExit(f"{task_path} has no case summary")
        slides = task.get("slides") or ""
        if isinstance(slides, list):
            slides = "\n".join(slides)
        slide_images = [f"slides/{p}" for p in task.get("slide_paths") or []]
        if len(slide_images) != slides.count("<image>"):
            raise SystemExit(f"{task_path}: {len(slide_images)} slide images for "
                             f"{slides.count('<image>')} slide captions")
        if recording not in bounds:
            bounds[recording] = case_bounds(recording)
        start_sec, end_sec = bounds[recording].get(case_id, (None, None))
        source_task = f"{recording.name}/{task_path.name}"
        video_uid = video_uids.setdefault(recording.name, f"video_{len(video_uids) + 1:04d}")

        simulation_id = f"sim_{len(cases) + 1:06d}"
        images[simulation_id] = slide_images
        cases.append({
            "schema_version": 1,
            "benchmark_version": args.benchmark_version,
            "task_name": "tumor_board_simulation",
            "simulation_id": simulation_id,
            "video_uid": video_uid,
            "video_dir": recording.name,
            "case_id": case_id,
            "case_start_sec": start_sec,
            "case_end_sec": end_sec,
            "case_summary": task["case_summary"],
            "slides": slides,
            "reference_conclusion": (task.get("conclusion") or {}).get("text") or "",
            "source_task": source_task,
        })

        # The rephrase pass can keep the same question from one utterance twice;
        # it is one test item, with every source location recorded.
        seen = {}
        for qa_index, pair in enumerate(pairs, start=1):
            key = (pair.get("utterance_id"), pair["question"].strip())
            alias = {"source_task": source_task, "qa_index": qa_index, "qa_type": pair["type"]}
            if key in seen:
                seen[key]["source_aliases"].append(alias)
                seen[key]["qa_type_candidates"] = sorted(
                    set(seen[key]["qa_type_candidates"]) | {pair["type"]})
                continue
            qa_id = f"qa_{len(questions) + 1:06d}"
            images[qa_id] = slide_images
            seen[key] = {
                "schema_version": 1,
                "benchmark_version": args.benchmark_version,
                "qa_id": qa_id,
                "video_uid": video_uid,
                "video_dir": recording.name,
                "case_id": case_id,
                "qa_index": qa_index,
                "qa_type": pair["type"],
                "qa_type_candidates": [pair["type"]],
                "utterance_id": pair.get("utterance_id") or "",
                "target_specialist_role": pair["speaker_role"],
                "case_summary": task["case_summary"],
                "slides": slides,
                "question": pair["question"].strip(),
                "reference_answer": pair.get("answer") or "",
                "source_task": source_task,
                "source_aliases": [alias],
            }
            questions.append(seen[key])

    out = args.output_dir
    inputs = out / "test_inputs"
    board_meta = [{k: v for k, v in c.items() if k != "reference_conclusion"} for c in cases]
    counts = {
        "task1.jsonl": write_jsonl(out / "task1.jsonl", cases),
        "task2.jsonl": write_jsonl(out / "task2.jsonl", questions),
        "test_inputs/board_simulation.image.jsonl": write_jsonl(inputs / "board_simulation.image.jsonl", [
            sharegpt(c["simulation_id"], "board_simulation_image", prompts, m,
                     board_user(c, "SLIDES"), images[c["simulation_id"]])
            for c, m in zip(cases, board_meta)]),
        "test_inputs/board_simulation.caption.jsonl": write_jsonl(inputs / "board_simulation.caption.jsonl", [
            sharegpt(c["simulation_id"], "board_simulation_caption", prompts, m,
                     board_user(c, "SLIDE DESCRIPTIONS"))
            for c, m in zip(cases, board_meta)]),
        "test_inputs/specialist_turn.image.jsonl": write_jsonl(inputs / "specialist_turn.image.jsonl", [
            sharegpt(q["qa_id"], "specialist_turn_image", prompts, q,
                     turn_user(q, "SLIDES"), images[q["qa_id"]])
            for q in questions]),
        "test_inputs/specialist_turn.caption.jsonl": write_jsonl(inputs / "specialist_turn.caption.jsonl", [
            sharegpt(q["qa_id"], "specialist_turn_caption", prompts, q,
                     turn_user(q, "SLIDE DESCRIPTIONS"))
            for q in questions]),
    }
    print(f"{len(video_uids)} recordings, {len(cases)} cases, {len(questions)} questions")
    for name, n in counts.items():
        print(f"{n:7d}  {out / name}")


if __name__ == "__main__":
    main()
