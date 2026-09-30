import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_client import client, DEPLOYMENT

_MIN_TEXT_LEN = 60  # skip utterances too short to contain meaningful clinical content

_SYSTEM_PROMPT = """\
You are building a clinical benchmark that tests whether an LLM can participate as an expert in a virtual tumor board — not whether it can recall or summarize a transcript.

BENCHMARK FRAMING:
- The LLM being tested will be given the patient case summary and asked to answer AS a specific clinical expert (oncologist, radiologist, molecular specialist, etc.).
- Each QUESTION is what another tumor board participant would naturally direct at that expert during the live discussion.
- Each ANSWER is the expert's actual response — their clinical reasoning and judgment as expressed in the utterance.

═══ SOURCE 1 — TRANSCRIPT QUESTIONS (highest priority) ═══
If a PRECEDING UTTERANCE is provided and contains a direct clinical question addressed to the current speaker, use that question as-is (verbatim or very lightly cleaned for grammar). This is always preferred because it is a real, naturally occurring tumor board question.
- Keep the question close to the original wording.
- The answer must be grounded in the CURRENT UTTERANCE only.

═══ SOURCE 2 — GENERATED QUESTIONS ═══
For clinical content in the current utterance not already covered by a transcript question, generate additional questions that:
- Are grounded in this patient's specific clinical details (name the relevant findings, values, grades, stages).
- Require domain expertise to answer correctly — a non-clinician with only the case summary could NOT answer them.
- Sound like real questions colleagues ask each other in tumor boards: about rationale, trade-offs, eligibility, evidence, or edge cases.
- Are NOT "What did the speaker say/mention/note about X?" — the question must stand alone as a clinical question.

═══ GROUNDING RULES (MANDATORY) ═══
- The ANSWER must be the expert's actual words from the utterance — verbatim or with only the lightest editing (remove filler words like "um/uh", fix false starts, trim to the relevant clause). Do NOT rephrase, summarize, or infer. The expert's voice must be preserved.
  WRONG: "I'd favor dose-escalated radiation because the disease appears high-grade and high-volume." (paraphrase — not their words)
  CORRECT: "It seems like he's got high-grade disease. It seems like he has high volume, so I'd want to do dose-escalated radiation." (their actual words, minimally trimmed)
- Your job is SELECTION, not generation: find the sentence(s) in the utterance that directly answer the question, and use them as the answer.
- The QUOTE field must be the exact verbatim excerpt — no editing at all. This should match or be a substring of the answer before any light cleanup.
- If no sentence in the utterance directly addresses the question, do NOT generate that QA pair.

STRICTLY FORBIDDEN:
- Questions asking what the speaker "said", "mentioned", "noted", "discussed", or "described".
- Questions whose correct answers can be directly read from the patient case summary without domain knowledge.
- For findings_interpretation: do NOT ask what a test showed — ask what it IMPLIES clinically.
- Answers that reference ANY participant by role label — including yourself or others. Never use "the speaker", "the moderator", "the oncologist", "the radiologist", "the expert", or any similar label in the answer. In a live meeting, participants speak directly: "I'd do X", "a colleague mentioned Y", "we agreed that Z".
- Any specific personal names (of clinicians, researchers, or anyone else) in either the question or the answer. Replace with a neutral role descriptor: "a colleague", "another panelist", "a radiation oncologist on the panel", etc.
- Any personal relationship references: "my partner", "my friend", "my colleague Dr. X", "someone I know". Replace with "a colleague" or omit entirely.

QA TYPE DEFINITIONS:
- findings_interpretation  : clinical significance or management implication of a finding (not what it showed, but what it means)
- treatment_recommendation : rationale for a proposed drug/regimen/intervention or the trade-off it addresses
- clinical_trial_suggestion: why a trial category is relevant, what eligibility criterion applies, or what the scientific basis is
- uncertainty              : what specific information is missing and how knowing it would change the decision
- eligibility_assessment   : whether and why this patient meets a specific criterion for a therapy, trial, or guideline
- evidence_discussion      : how published data, guidelines, or prior experience applies to this patient's situation
- agreement_or_support     : substantive endorsement of a clinical point with the reasoning behind it
- clarification_question   : a question seeking specific clinical information that would materially affect management
- next_action_suggestion   : what specific test, referral, procedure, or follow-up step should be taken next for this patient, and the clinical rationale

SKIP the utterance (return empty list) if it contains only social filler, logistics, or no clinical substance.

OUTPUT FORMAT:
Return ONLY a valid JSON object with key "qa_pairs" whose value is a list.
Each element must have exactly four string keys: "type", "question", "answer", "quote".
"""

_USER_PROMPT_TEMPLATE = """\
PATIENT CASE SUMMARY (the LLM being tested will have access to this):
{case_summary}

---
{preceding_block}
CURRENT SPEAKER ROLE: {role}
UTTERANCE CLASSIFICATION: {response_type}

CURRENT UTTERANCE:
{text}

---
Generate atomic QA pairs. All answers and quotes must be grounded exclusively in the CURRENT UTTERANCE.
If a PRECEDING UTTERANCE is present and contains a clinical question, use it as your first question source.\
"""


def _generate_qa_pairs(
    utterance: dict,
    case_summary: str,
    model: str,
    prev_utterance: dict | None = None,
) -> list[dict]:
    text = utterance.get("text", "").strip()
    if len(text) < _MIN_TEXT_LEN:
        return []

    role = utterance.get("inferred_role", "clinician")
    response_type = utterance.get("response_type", "unknown")

    if prev_utterance:
        prev_role = prev_utterance.get("inferred_role", "participant")
        prev_text = prev_utterance.get("text", "").strip()
        preceding_block = (
            f"PRECEDING UTTERANCE ({prev_role}):\n{prev_text}\n\n"
        )
    else:
        preceding_block = ""

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        case_summary=case_summary,
        preceding_block=preceding_block,
        role=role,
        response_type=response_type,
        text=text,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)
    pairs = parsed.get("qa_pairs", [])

    for pair in pairs:
        pair["utterance_id"] = utterance.get("utterance_id", "")
        pair["speaker_role"] = role

    return pairs


def process_case(
    align_path: str,
    output_dir: str,
    model: str,
) -> None:
    with open(align_path, encoding="utf-8") as f:
        align_data = json.load(f)

    case_id = align_data.get("case_id", os.path.basename(align_path))
    utterances = align_data.get("aligned_utterances", [])

    # Load case_summary from task_case_*.json if available
    task_path = align_path.replace("align_", "task_")
    task_data: dict = {}
    if os.path.exists(task_path):
        with open(task_path, encoding="utf-8") as f:
            task_data = json.load(f)
    case_summary = task_data.get("case_summary", "")

    print(f"  {case_id}: {len(utterances)} utterances, case_summary={'yes' if case_summary else 'missing'}")

    all_qa_pairs: list[dict] = []
    for i, utt in enumerate(utterances):
        uid = utt.get("utterance_id", "?")
        role = utt.get("inferred_role", "")

        if role == "moderator":
            print(f"    {uid} (moderator): skipped")
            continue

        prev_utt = utterances[i - 1] if i > 0 else None
        try:
            pairs = _generate_qa_pairs(utt, case_summary, model, prev_utterance=prev_utt)
            print(f"    {uid} ({role}/{utt.get('response_type','?')}): {len(pairs)} QA pair(s)")
            all_qa_pairs.extend(pairs)
        except Exception as e:
            print(f"    {uid}: ERROR — {e}")

    task_data["qa_pairs"] = all_qa_pairs

    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(all_qa_pairs)} QA pairs → {task_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract atomic QA pairs from each align_case_*.json and write to task_case_*.json."
    )
    parser.add_argument(
        "--dir", required=True,
        help="Processed video directory containing align_case_*.json and task_case_*.json files",
    )
    parser.add_argument(
        "--model", default=DEPLOYMENT,
        help="Azure OpenAI deployment to use",
    )
    args = parser.parse_args()

    output_dir = args.dir.rstrip("/")

    align_files = sorted(glob.glob(os.path.join(output_dir, "align_case_*.json")))
    if not align_files:
        print(f"No align_case_*.json files found in {output_dir}")
        return

    print(f"Found {len(align_files)} case file(s) in {output_dir}\n")
    for align_path in align_files:
        print(f"Processing {os.path.basename(align_path)}...")
        try:
            process_case(align_path, output_dir, args.model)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
