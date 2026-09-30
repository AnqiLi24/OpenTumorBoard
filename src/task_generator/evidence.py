import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_client import client, DEPLOYMENT

_SYSTEM_PROMPT = (
    "You are preparing a structured clinical case summary for a multidisciplinary tumor board meeting. "
    "This document will be read by specialists who have NOT seen the patient and have NOT attended any prior discussion. "
    "Write as a formal clinical case brief — the same register as a hospital referral letter or case note. "
    "Include ONLY objective, verifiable clinical facts: demographics, presenting complaint, diagnosis, "
    "histopathology, imaging, molecular/genomic findings, laboratory results, prior treatments and outcomes, "
    "and current disease status. "
    "STRICTLY FORBIDDEN: any reference to a transcript, discussion, meeting, speaker, or source of information "
    "(never write 'as mentioned', 'later in the transcript', 'the presenter stated', 'it was noted', etc.). "
    "STRICTLY FORBIDDEN: expert opinions, treatment recommendations, panel decisions, or speculative claims. "
    "Write in past tense, impersonal clinical prose, as if reporting from the medical record. "
    "Omit any section for which no factual data are available.\n\n"
    "Use this structure:\n\n"
    "[ patient demographics ]: ...\n\n"
    "[ chief complaint ]: ...\n\n"
    "[ diagnosis ]: ...\n\n"
    "[ histopathology results ]: ...\n\n"
    "[ imaging findings ]: ...\n\n"
    "[ molecular/genomic findings ]: ...\n\n"
    "[ laboratory results ]: ...\n\n"
    "[ prior treatments and outcomes ]: ...\n\n"
    "[ current disease status ]: ...\n\n"
    "[ other clinical data ]: ...\n\n"
)

_USER_PROMPT_TEMPLATE = (
    "The following source material contains clinical information about a patient being referred to a tumor board. "
    "Extract all objective clinical facts and present them as a structured case brief for the attending specialists.\n\n"
    "SOURCE:\n{transcript}\n\n"
    "Write the case brief following the system instructions. "
    "Do not reference the source material itself — write only the clinical facts as they would appear in a medical record."
)


def _build_transcript(utterances: list) -> str:
    lines = []
    for u in utterances:
        role = u.get("speaker_id", "speaker")
        text = u.get("text", "").strip()
        if text:
            lines.append(f"[{role}]: {text}")
    return "\n\n".join(lines)


def _generate_case_summary(utterances: list, model: str) -> str:
    transcript = _build_transcript(utterances)
    if not transcript:
        return ""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(transcript=transcript)},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def _collect_slide_ids(utterances: list) -> list:
    seen = set()
    ordered = []
    for u in utterances:
        for sid in u.get("slide_ids", []):
            if sid not in seen:
                seen.add(sid)
                ordered.append(sid)
    return ordered


def process_case_file(case_path: str, slides_by_id: dict, output_dir: str, model: str) -> None:
    with open(case_path, encoding="utf-8") as f:
        case_data = json.load(f)

    case_id = case_data.get("case_id", os.path.basename(case_path))
    utterances = case_data.get("aligned_utterances", [])

    print(f"  Generating case summary for {case_id}...")
    case_summary = _generate_case_summary(utterances, model)

    slide_ids = _collect_slide_ids(utterances)

    slides_text_parts = []
    slide_paths = []
    for sid in slide_ids:
        slide = slides_by_id.get(sid)
        if slide is None:
            continue
        caption = slide.get("caption", "").strip()
        frame_path = slide.get("frame_path", "")
        abs_path = os.path.join(os.path.basename(output_dir), frame_path)
        slides_text_parts.append(f"<image> {caption}" if caption else "<image>")
        slide_paths.append(abs_path)

    evidence = {
        "case_summary": case_summary,
        "slides": "\n".join(slides_text_parts),
        "slide_paths": slide_paths,
    }

    print(case_summary)

    save_evidence_path = case_path.replace("align_", "task_")
    with open(save_evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)

    print(f"  Saved evidence to {save_evidence_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evidence (clinical facts + slides) for each align_case_*.json."
    )
    parser.add_argument(
        "--dir", required=True,
        help="Processed video directory containing align_case_*.json and slides.raw.json",
    )
    parser.add_argument(
        "--model", default=DEPLOYMENT,
        help="Azure OpenAI deployment to use",
    )
    args = parser.parse_args()

    output_dir = args.dir.rstrip("/")

    slides_path = os.path.join(output_dir, "slides.raw.json")
    slides_by_id: dict = {}
    if os.path.exists(slides_path):
        with open(slides_path, encoding="utf-8") as f:
            slides_data = json.load(f)
        slides_by_id = {s["slide_id"]: s for s in slides_data.get("slides", [])}
        print(f"Loaded {len(slides_by_id)} slides from {slides_path}")
    else:
        print(f"Warning: {slides_path} not found — slide fields will be empty")

    case_files = sorted(glob.glob(os.path.join(output_dir, "align_case_*.json")))
    if not case_files:
        print(f"No align_case_*.json files found in {output_dir}")
        return

    print(f"Found {len(case_files)} case file(s).")
    for case_path in case_files:
        print(f"\nProcessing {os.path.basename(case_path)}...")
        try:
            process_case_file(case_path, slides_by_id, output_dir, args.model)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
