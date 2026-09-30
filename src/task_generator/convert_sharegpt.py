"""
Convert task_case_*.json files to ShareGPT instruction-tuning format.

Setting 1 — Tumor Board Simulation:
  Input : case_summary + slides
  Output: simulated discussion + conclusion

Setting 2 — Expert QA:
  Input : case_summary + slides + question  (one sample per QA pair)
  Output: expert answer
"""

import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── System prompts ──────────────────────────────────────────────────────────

_SYSTEM_SETTING1 = """\
You are a multidisciplinary tumor board AI. Given a patient case summary and relevant presentation slides, simulate a full tumor board discussion in which specialists from different disciplines review the case, exchange clinical perspectives, and reach a consensus treatment plan.

Format your response exactly as:

<tumor board discussion>
| [role] | [response] |

| [role] | [response] |

...
</tumor board discussion>

CONCLUSION: [2-4 sentence objective treatment plan in formal clinical prose]

Guidelines:
- Roles: moderator, medical oncologist, radiation oncologist, surgeon, radiologist, pathologist, molecular pathologist, genetic counselor, clinical trial specialist. Use only roles relevant to this case.
- Each turn is one specialist's contribution. Responses must reflect genuine domain expertise.
- The discussion should progress naturally: case presentation → specialist input → cross-disciplinary debate → consensus.
- The CONCLUSION must be written in impersonal third-person clinical language. Do not use "I think", "I would", "we", or any first-person phrasing.\
"""

_SYSTEM_SETTING2 = """\
You are a {role} participating in a multidisciplinary tumor board meeting. You will be given a patient case summary, relevant presentation slides, and a question from another participant. Answer as you would in a real tumor board — directly, concisely, and in first person. Ground your answer strictly in the clinical facts presented. Do not introduce information not in the case.\
"""

# ── Formatting helpers ──────────────────────────────────────────────────────

def _format_case_input(case_summary: str, slides: str) -> str:
    parts = ["PATIENT CASE SUMMARY:", case_summary.strip()]
    if slides and slides.strip():
        parts += ["", "PRESENTATION SLIDES:", slides.strip()]
    return "\n".join(parts)


def _build_discussion_answer(utterances: list, conclusion: dict | str) -> str:
    rows = []
    for u in utterances:
        role = u.get("inferred_role", "specialist")
        text = u.get("text", "").strip()
        if text:
            rows.append(f"| {role} | {text} |")

    discussion_block = (
        "<tumor board discussion>\n"
        + "\n\n".join(rows)
        + "\n</tumor board discussion>"
    )

    if isinstance(conclusion, dict):
        conclusion_text = conclusion.get("text", "").strip()
    else:
        conclusion_text = str(conclusion).strip()

    return f"{discussion_block}\n\nCONCLUSION: {conclusion_text}"


# ── Per-setting sample builders ─────────────────────────────────────────────

def _build_setting1_sample(
    task_data: dict,
    align_data: dict,
    video_id: str,
    case_id: str,
) -> dict | None:
    conclusion = task_data.get("conclusion")
    if not conclusion:
        return None

    utterances = align_data.get("aligned_utterances", [])
    if not utterances:
        return None

    human_text = (
        _format_case_input(task_data.get("case_summary", ""), task_data.get("slides", ""))
        + "\n\nConduct the tumor board review for this patient."
    )
    gpt_text = _build_discussion_answer(utterances, conclusion)

    return {
        "id": f"s1__{video_id}__{case_id}",
        "setting": "tumor_board_simulation",
        "conversations": [
            {"from": "system", "value": _SYSTEM_SETTING1},
            {"from": "human",  "value": human_text},
            {"from": "gpt",    "value": gpt_text},
        ],
    }


def _build_setting2_samples(
    task_data: dict,
    video_id: str,
    case_id: str,
) -> list[dict]:
    qa_pairs = task_data.get("qa_pairs", [])
    case_input = _format_case_input(
        task_data.get("case_summary", ""), task_data.get("slides", "")
    )

    samples = []
    for i, qa in enumerate(qa_pairs):
        question = qa.get("question", "").strip()
        answer   = qa.get("answer",   "").strip()
        role     = qa.get("speaker_role", "specialist")
        uid      = qa.get("utterance_id", f"qa{i}")

        if not question or not answer:
            continue

        samples.append({
            "id": f"s2__{video_id}__{case_id}__{uid}_{i}",
            "setting": "expert_qa",
            "speaker_role": role,
            "qa_type": qa.get("type", ""),
            "conversations": [
                {"from": "system", "value": _SYSTEM_SETTING2.format(role=role)},
                {"from": "human",  "value": f"{case_input}\n\nQUESTION: {question}"},
                {"from": "gpt",    "value": answer},
            ],
        })

    return samples


# ── Main ────────────────────────────────────────────────────────────────────

def convert(input_dir: str, output_path: str, settings: list[int]) -> None:
    input_dir = input_dir.rstrip("/")
    video_id  = os.path.basename(input_dir)

    # Exclude QC-filtered variants (task_case_*_qc*.json) so the canonical
    # sharegpt is built only from the base task files.
    task_files = sorted(
        p for p in glob.glob(os.path.join(input_dir, "task_case_*.json"))
        if "_qc" not in os.path.basename(p)
    )
    if not task_files:
        print(f"No task_case_*.json files found in {input_dir}")
        return

    all_samples: list[dict] = []

    for task_path in task_files:
        case_id    = os.path.basename(task_path).replace("task_", "").replace(".json", "")
        align_path = task_path.replace("task_", "align_")

        with open(task_path, encoding="utf-8") as f:
            task_data = json.load(f)

        align_data: dict = {}
        if os.path.exists(align_path):
            with open(align_path, encoding="utf-8") as f:
                align_data = json.load(f)

        if 1 in settings:
            s1 = _build_setting1_sample(task_data, align_data, video_id, case_id)
            if s1:
                all_samples.append(s1)
                print(f"  [S1] {case_id}: 1 sample")
            else:
                print(f"  [S1] {case_id}: skipped (missing conclusion or utterances)")

        if 2 in settings:
            s2_list = _build_setting2_samples(task_data, video_id, case_id)
            all_samples.extend(s2_list)
            print(f"  [S2] {case_id}: {len(s2_list)} samples")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    s1_count = sum(1 for s in all_samples if s["setting"] == "tumor_board_simulation")
    s2_count = sum(1 for s in all_samples if s["setting"] == "expert_qa")
    print(f"\nTotal: {len(all_samples)} samples  (S1={s1_count}, S2={s2_count})")
    print(f"Saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert task_case_*.json to ShareGPT instruction-tuning format."
    )
    parser.add_argument(
        "--dir", required=True,
        help="Processed video directory containing task_case_*.json and align_case_*.json",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: <dir>/sharegpt.json)",
    )
    parser.add_argument(
        "--settings", nargs="+", type=int, choices=[1, 2], default=[1, 2],
        help="Settings to generate: 1=tumor board simulation, 2=expert QA (default: both)",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.dir.rstrip("/"), "sharegpt.json")
    print(f"Input   : {args.dir}")
    print(f"Output  : {output_path}")
    print(f"Settings: {args.settings}\n")

    convert(args.dir, output_path, args.settings)


if __name__ == "__main__":
    main()
