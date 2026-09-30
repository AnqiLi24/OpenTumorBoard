import torch

# Monkey-patch torch.load to bypass weights_only=True which became default in PyTorch 2.6
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Extract raw ASR transcript using WhisperX.")
    parser.add_argument("--video", type=str, required=True, help="Path to the mp4 file.")
    parser.add_argument("--output", type=str, default="transcript.raw.json", help="Output raw JSON path.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device to use.")
    
    args = parser.parse_args()
    
    video_path = args.video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Set inference parameters
    batch_size = 4
    # Use float16 for CUDA, int8 for low memory, fp32 for CPU
    compute_type = "float16" if args.device == "cuda" else "int8"
    
    print(f"Loading audio from {video_path}...")
    audio = whisperx.load_audio(video_path)

    print("Transcribing audio (this will take a while)...")
    # Using 'large-v2' model for better transcription accuracy
    model = whisperx.load_model("large-v2", args.device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size)
    
    print("Aligning transcript...")
    language = result["language"]
    model_a, metadata = whisperx.load_align_model(language_code=language, device=args.device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, args.device, return_char_alignments=False)

    print("Formatting output into transcript.raw.json format...")
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    utterances = []
    for i, segment in enumerate(result["segments"]):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        text = segment.get("text", "").strip()
        
        utterances.append({
            "utterance_id": f"utt_{i+1:06d}",
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "text": text
        })
        
    raw_transcript_format = {
        "video_id": video_id,
        "asr_model": "whisperx",
        "language": language,
        "utterances": utterances
    }

    # make directory if not exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(raw_transcript_format, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully processed video and saved raw transcript (without speakers) to: {args.output}")

if __name__ == "__main__":
    main()
