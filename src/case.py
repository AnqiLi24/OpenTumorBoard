import argparse
import json
import os
from pydantic import BaseModel
from gpt_client import client, DEPLOYMENT

# Define expected output format using pydantic for clarity
class CaseSegment(BaseModel):
    case_id: str
    start_sec: float
    end_sec: float

class CaseSegmentation(BaseModel):
    cases: list[CaseSegment]

def segment_cases(transcript_path: str, output_path: str, model_name: str = DEPLOYMENT):
    """
    Reads a transcript file, uses an LLM to segment the discussion into patient cases,
    and writes the results to an output JSON file.
    """
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)

    utterances = transcript_data.get("utterances", [])
    if not utterances:
        raise ValueError("No utterances found in the transcript.")

    # Merge consecutive utterances from the same speaker
    merged = []
    for utt in utterances:
        if merged and merged[-1].get("speaker_id") == utt.get("speaker_id"):
            merged[-1]["end_sec"] = utt.get("end_sec")
            merged[-1]["text"] = merged[-1]["text"].rstrip() + " " + utt.get("text", "").strip()
        else:
            merged.append({**utt})
    utterances = merged

    # Format transcript for the LLM
    # To save tokens and make it easier to read, we use a simple format:
    # [start_sec - end_sec] text
    formatted_transcript = []
    for utt in utterances:
        start = utt.get("start_sec")
        end = utt.get("end_sec")
        text = utt.get("text", "").strip()
        formatted_transcript.append(f"[Start Time: {start} - End Time: {end}] {text}")

    transcript_text = "\n".join(formatted_transcript)

    prompt = (
        "You are an expert medical virtual tumor board assistant. "
        "Below is a transcript of a panel discussion, where experts discuss several patient clinical cases sequentially.\n\n"
        "Your task is to identify and segment the transcript into distinct patient cases. "
        "Make sure to incorporate all the discussion around each case afterwards."
        "Look for transitions like 'Let's move to the next patient/case', 'Patient/case two', or 'First patient/case'.\n\n"
        "Return the case segments in JSON format exactly as follows:\n"
        "{\n"
        "  \"cases\": [\n"
        "    {\n"
        "      \"case_id\": \"case_001\",\n"
        "      \"start_sec\": 120.0,\n"
        "      \"end_sec\": 540.0\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "The start_sec should be the start time of the first utterance of the case discussion.\n"
        "The end_sec should be the end time of the last utterance of the case discussion.\n"
        "Only segment actual designated medical cases discussed.\n"
        "Be completely comprehensive and output valid JSON only.\n\n"
        "Transcript:\n"
        f"{transcript_text}"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that segments medical transcripts into JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        result_content = response.choices[0].message.content
        parsed_result = json.loads(result_content)

        # Basic validation
        CaseSegmentation(**parsed_result)

        # Save to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_result, f, indent=2)

        print(f"Successfully segmented cases and saved to {output_path}")

    except Exception as e:
        print(f"An error occurred during API call or validation: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a tumor board transcript into cases using LLM.")
    parser.add_argument("--transcript", type=str, default='./processed/Penile Cancer Management： Insights & Case Studies ｜ BackTable Tumor Board Ep. 17.mp4/transcript.with_speakers.json', help="Path to transcript.raw.json")
    parser.add_argument("--output", type=str, default='./processed/Penile Cancer Management： Insights & Case Studies ｜ BackTable Tumor Board Ep. 17.mp4/case_segmentation.json', help="Path to output case_segmentation.json")
    parser.add_argument("--model", type=str, default=DEPLOYMENT, help="Azure OpenAI deployment to use")
    
    args = parser.parse_args()
    segment_cases(args.transcript, args.output, args.model)
