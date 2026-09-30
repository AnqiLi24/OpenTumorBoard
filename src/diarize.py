import torch

# Monkey-patch torch.load to bypass weights_only=True which became default in PyTorch 2.6
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx
from whisperx.diarize import DiarizationPipeline
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Perform speaker diarization and assign to transcripts.")
    parser.add_argument("--video", type=str, required=True, help="Path to the mp4 file (for audio extraction).")
    parser.add_argument("--transcript", type=str, required=True, help="Path to the input transcript.raw.json.")
    parser.add_argument("--hf-token", type=str, required=True, help="HuggingFace token for pyannote (required for diarization).")
    parser.add_argument("--output", type=str, default="transcript.with_speakers.json", help="Output JSON path.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device to use.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    if not os.path.exists(args.transcript):
        raise FileNotFoundError(f"Transcript file not found: {args.transcript}")

    print(f"Loading raw transcript from {args.transcript}...")
    with open(args.transcript, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert our minimal utterance format into WhisperX 'segments' format
    # whisperx.assign_word_speakers expects segments to have at least 'start', 'end', 'text'
    whisperx_result = {"segments": []}
    for utt in data.get("utterances", []):
        whisperx_result["segments"].append({
            "start": utt["start_sec"],
            "end": utt["end_sec"],
            "text": utt["text"]
        })

    print(f"Loading audio from {args.video}...")
    audio = whisperx.load_audio(args.video)

    print("Running Speaker Diarization...")
    diarize_model = DiarizationPipeline(use_auth_token=args.hf_token, device=args.device)
    # the output is a pandas DataFrame with columns 'segment', 'label', 'speaker', 'start', 'end'
    diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=10)

    print("Assigning speakers to text...")
    whisperx_result = whisperx.assign_word_speakers(diarize_segments, whisperx_result)

    print("Formatting output into transcript.with_speakers.json format...")
    
    # 1. Map generic speaker labels ("SPEAKER_00") to custom IDs ("spk_01")
    # Using the exact unique speakers returned by diarization model
    unique_speakers_raw = sorted(diarize_segments["speaker"].unique().tolist())
    speaker_map = {}
    for i, spk_label in enumerate(unique_speakers_raw):
        speaker_map[spk_label] = f"spk_{i+1:02d}"

    # 2. Re-build utterances list with assigned speaker_id attached
    output_utterances = []
    for i, seg in enumerate(whisperx_result["segments"]):
        original_utt = data["utterances"][i]
        
        # WhisperX might leave speaker missing if no overlap was found, default to 'UNKNOWN'
        raw_spk = seg.get("speaker", "UNKNOWN")
        spk_id = speaker_map.get(raw_spk, raw_spk)
        
        output_utterances.append({
            "utterance_id": original_utt["utterance_id"],
            "start_sec": original_utt["start_sec"],
            "end_sec": original_utt["end_sec"],
            "speaker_id": spk_id,
            "text": original_utt["text"]
        })
        
    # 3. Construct speakers array representing their exact boundary segments
    speakers_list = []
    for spk_raw in unique_speakers_raw:
        spk_id = speaker_map[spk_raw]
        
        # Filter dataframe for this speaker
        spk_df = diarize_segments[diarize_segments["speaker"] == spk_raw]
        
        segments_list = []
        for _, row in spk_df.iterrows():
            segments_list.append({
                "start_sec": round(float(row["start"]), 3),
                "end_sec": round(float(row["end"]), 3)
            })
            
        speakers_list.append({
            "speaker_id": spk_id,
            "segments": segments_list
        })
        
    final_output = {
        "video_id": data.get("video_id", "unknown_video"),
        "utterances": output_utterances,
        "speakers": speakers_list
    }
    
    # make directory if not exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully finished diarization! Output saved to: {args.output}")

if __name__ == "__main__":
    main()
