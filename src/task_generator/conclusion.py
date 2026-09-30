import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_client import client, DEPLOYMENT

# Per NCI Dictionary of Cancer Terms, a tumor board review is:
# "A treatment planning process in which a group of cancer doctors and other health care
# specialists meet regularly to review and discuss new and complex cancer cases."
# The board examines diagnostic results, develops a treatment strategy across all modalities
# (surgery, chemotherapy, immunotherapy, radiation, clinical trials), and accounts for
# patient-specific factors such as comorbidities, performance status, and genetic findings.
_PROMPT_TEMPLATE = """\
You are a senior oncologist summarising the treatment plan agreed upon in a tumor board review session.

Per the NCI definition, a tumor board review is a treatment planning process in which a multidisciplinary group of cancer specialists collectively develops a treatment strategy or next steps for a patient.

Below is the full annotated discussion transcript for one patient case.

Your task:
1. Identify the sentences in the transcript that together constitute the agreed treatment plan or next steps.
2. Produce a JSON object with exactly two fields:
   - "text": a 2–4 sentence objective, professional treatment plan summary written in formal clinical language. Rewrite the panel's conclusions in impersonal third-person prose — remove all first-person expressions ("I would", "I think", "we", "my"), hedges, filler, and colloquialisms. Retain only the clinical substance and terminology from the panel's discussion. Do NOT introduce recommendations not present in the transcript.
   - "quote": the verbatim excerpt(s) from the transcript that directly support the treatment plan or next steps — copy exact words, no paraphrasing. If multiple passages are needed, separate them with " [...] ".

Do not include patient-identifiable information.

Return ONLY valid JSON: {{"text": "...", "quote": "..."}}

Discussion transcript:

{discussion_text}
"""


def _build_discussion_text(utterances: list) -> str:
    lines = []
    for u in utterances:
        role = u.get("inferred_role", "unknown")
        rtype = u.get("response_type", "")
        tag = f"[{role}]" + (f"[{rtype}]" if rtype else "")
        lines.append(f"{tag} {u['text'].strip()}")
    return "\n\n".join(lines)


def generate_conclusion(align_path: str, model_name: str = DEPLOYMENT) -> None:
    with open(align_path, encoding="utf-8") as f:
        align_data = json.load(f)

    utterances = align_data.get("aligned_utterances", [])
    if not utterances:
        print(f"  No utterances in {os.path.basename(align_path)}, skipping.")
        return

    discussion_text = _build_discussion_text(utterances)
    prompt = _PROMPT_TEMPLATE.format(discussion_text=discussion_text)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a senior oncologist writing formal tumor board treatment plan summaries."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    conclusion = json.loads(response.choices[0].message.content)

    # Write conclusion into task_case_*.json alongside evidence and qa_pairs
    task_path = align_path.replace("align_", "task_")
    task_data: dict = {}
    if os.path.exists(task_path):
        with open(task_path, encoding="utf-8") as f:
            task_data = json.load(f)

    task_data["conclusion"] = conclusion

    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task_data, f, indent=2, ensure_ascii=False)

    print(f"  Conclusion written → {os.path.basename(task_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NCI-aligned treatment plan conclusions for each align_case_*.json."
    )
    parser.add_argument("--dir", required=True, help="Directory containing align_case_*.json and task_case_*.json files")
    parser.add_argument("--model", default=DEPLOYMENT, help="Azure OpenAI deployment to use")
    args = parser.parse_args()

    align_files = sorted(glob.glob(os.path.join(args.dir, "align_case_*.json")))
    if not align_files:
        raise FileNotFoundError(f"No align_case_*.json files found in: {args.dir}")

    for align_path in align_files:
        case_id = os.path.basename(align_path).replace("align_", "").replace(".json", "")
        print(f"Processing {case_id}...")
        try:
            generate_conclusion(align_path, args.model)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
