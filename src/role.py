import argparse
import json
import os
from pydantic import BaseModel
from gpt_client import client, DEPLOYMENT


VALID_ROLES = [
    "moderator",
    "pathologist",
    "radiologist",
    "medical oncologist",
    "radiation oncologist",
    "surgeon",
    "molecular pathologist",
    "genetic counselor",
    "clinical trial specialist",
    "nurse navigator",
    "other",
]


class SpeakerRole(BaseModel):
    speaker_id: str
    inferred_role: str  # one of VALID_ROLES
    role_description: str  # professional profile, no conversation content or decisions


class SpeakerRoles(BaseModel):
    speakers: list[SpeakerRole]


def infer_roles(transcript_path: str, output_path: str, model_name: str = DEPLOYMENT):
    """
    Reads a transcript with speaker IDs, groups utterances per speaker,
    calls an LLM to infer each speaker's role, and writes speaker_profiles.json.
    """
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    video_id = transcript_data.get("video_id", "")
    utterances = transcript_data.get("utterances", [])
    if not utterances:
        raise ValueError("No utterances found in the transcript.")

    # Group utterances by speaker
    speaker_map: dict[str, list[dict]] = {}
    for utt in utterances:
        sid = utt["speaker_id"]
        speaker_map.setdefault(sid, []).append(utt)

    # Build per-speaker data: full_text, stats, utterance_ids
    speaker_data = {}
    for sid, utts in speaker_map.items():
        utts_sorted = sorted(utts, key=lambda u: u["start_sec"])
        utterance_ids = [u["utterance_id"] for u in utts_sorted]
        full_text = " ".join(u["text"].strip() for u in utts_sorted)
        total_duration = sum(u["end_sec"] - u["start_sec"] for u in utts_sorted)
        speaker_data[sid] = {
            "utterance_ids": utterance_ids,
            "full_text": full_text,
            "stats": {
                "utterance_count": len(utts_sorted),
                "total_duration_sec": round(total_duration, 3),
                "first_start_sec": utts_sorted[0]["start_sec"],
                "last_end_sec": utts_sorted[-1]["end_sec"],
            },
        }

    # Build prompt: give GPT each speaker's full_text and ask for role
    speaker_blocks = []
    for sid, data in speaker_data.items():
        speaker_blocks.append(f"Speaker {sid}:\n{data['full_text']}")
    speakers_text = "\n\n---\n\n".join(speaker_blocks)

    prompt = (
        "You are an expert in oncology tumor board meetings. "
        "Below are transcripts of what each speaker said during a virtual tumor board session.\n\n"
        "For each speaker, do two things:\n\n"
        "1. Infer their most likely clinical role from the following options:\n"
        f"   {VALID_ROLES}\n"
        "   Base your inference on vocabulary, topics discussed, and how they interact with others:\n"
        "   - moderator: introduces cases, facilitates discussion, asks follow-up questions.\n"
        "   - pathologist: discusses biopsy, histology, tissue diagnosis, and gross/micro pathology.\n"
        "   - radiologist: interprets imaging studies (CT, MRI, PET, ultrasound).\n"
        "   - medical oncologist: discusses systemic therapy, chemotherapy, immunotherapy, targeted agents, and prognosis.\n"
        "   - radiation oncologist: discusses radiation planning, dose, fractionation, and field design.\n"
        "   - surgeon: discusses surgical technique, resection, lymph node dissection, margins, and operative planning.\n"
        "   - molecular pathologist: discusses genomic/molecular profiling, NGS results, biomarkers, and variant interpretation.\n"
        "   - genetic counselor: discusses germline implications, hereditary risk, variant allele frequency, and genetic testing.\n"
        "   - clinical trial specialist: discusses trial eligibility, NCT numbers, enrollment criteria, and trial design.\n"
        "   - nurse navigator: discusses care coordination, patient support, logistics, and follow-up scheduling.\n"
        "   - other: role is unclear or does not fit any category above.\n\n"
        "2. Write a concise professional profile (2-4 sentences) for the speaker. "
        "The profile should describe their clinical specialty, institutional background if inferable, "
        "and general expertise relevant to tumor boards. "
        "You MAY reference any names, affiliations, or specialties mentioned in the transcript. "
        "Do NOT reveal, paraphrase, or allude to any specific case discussion, treatment recommendation, "
        "clinical decision, or patient information from the transcript.\n\n"
        "Return JSON exactly in this format:\n"
        "{\n"
        "  \"speakers\": [\n"
        "    {\n"
        "      \"speaker_id\": \"spk_01\",\n"
        "      \"inferred_role\": \"oncologist\",\n"
        "      \"role_description\": \"Dr. ... is an oncologist at ... specializing in ...\"\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Transcripts:\n\n"
        f"{speakers_text}"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies tumor board speakers into JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        result_content = response.choices[0].message.content
        parsed = json.loads(result_content)
        validated = SpeakerRoles(**parsed)

    except Exception as e:
        print(f"Error during API call or validation: {e}")
        raise

    # Build final output
    profile_lookup = {s.speaker_id: s for s in validated.speakers}

    speaker_profiles = []
    for sid, data in speaker_data.items():
        sp = profile_lookup.get(sid)
        inferred_role = sp.inferred_role if sp else "other"
        if inferred_role not in VALID_ROLES:
            inferred_role = "other"
        role_description = sp.role_description if sp else ""
        speaker_profiles.append({
            "speaker_id": sid,
            "utterance_ids": data["utterance_ids"],
            "inferred_role": inferred_role,
            "role_description": role_description,
            "full_text": data["full_text"],
            "stats": data["stats"],
        })

    # Sort by first appearance
    speaker_profiles.sort(key=lambda p: p["stats"]["first_start_sec"])

    output = {
        "video_id": video_id,
        "speaker_profiles": speaker_profiles,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Inferred roles for {len(speaker_profiles)} speakers:")
    for p in speaker_profiles:
        print(f"  {p['speaker_id']}: {p['inferred_role']}  ({p['stats']['utterance_count']} utterances)")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Infer speaker roles from a tumor board transcript.")
    parser.add_argument("--transcript", type=str, required=True, help="Path to transcript.with_speakers.json")
    parser.add_argument("--output", type=str, required=True, help="Path to output speaker_profiles.json")
    parser.add_argument("--model", type=str, default=DEPLOYMENT, help="Azure OpenAI deployment to use")
    args = parser.parse_args()

    infer_roles(args.transcript, args.output, args.model)


if __name__ == "__main__":
    main()
