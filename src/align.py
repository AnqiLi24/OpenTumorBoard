import argparse
import json
import os


def find_slide_for_utterance(utt_start, utt_end, slides):
    """Return the slide_id with the most overlap with [utt_start, utt_end], or None."""
    best_slide_id = None
    best_overlap = 0.0

    for slide in slides:
        overlap_start = max(utt_start, slide["start_sec"])
        overlap_end = min(utt_end, slide["end_sec"])
        overlap = overlap_end - overlap_start
        if overlap > best_overlap:
            best_overlap = overlap
            best_slide_id = slide["slide_id"]

    return best_slide_id


def merge_consecutive_utterances(aligned_utterances):
    """Merge consecutive utterances by the same speaker.
    slide_ids is a deduplicated list of non-null slide_ids in order, or [] if all null.
    """
    if not aligned_utterances:
        return []

    def make_group(utt):
        group = {k: v for k, v in utt.items() if k != "slide_id"}
        group["slide_ids"] = [utt["slide_id"]] if utt["slide_id"] is not None else []
        return group

    merged = []
    current = make_group(aligned_utterances[0])

    for utt in aligned_utterances[1:]:
        if utt["speaker_id"] == current["speaker_id"]:
            current["end_sec"] = utt["end_sec"]
            current["text"] = current["text"] + " " + utt["text"]
            if utt["slide_id"] is not None and utt["slide_id"] not in current["slide_ids"]:
                current["slide_ids"].append(utt["slide_id"])
        else:
            merged.append(current)
            current = make_group(utt)

    merged.append(current)
    return merged


def align(slides_path, transcript_path, output_path, case_segmentation_path=None, speaker_profiles_path=None):
    with open(slides_path, "r", encoding="utf-8") as f:
        slides_data = json.load(f)

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    video_id = transcript_data.get("video_id", slides_data.get("video_id", ""))
    slides = slides_data.get("slides", [])
    utterances = transcript_data.get("utterances", [])

    aligned_utterances = []
    for utt in utterances:
        slide_id = find_slide_for_utterance(utt["start_sec"], utt["end_sec"], slides)
        aligned_utterances.append({
            "utterance_id": utt["utterance_id"],
            "speaker_id": utt["speaker_id"],
            "start_sec": utt["start_sec"],
            "end_sec": utt["end_sec"],
            "text": utt["text"],
            "slide_id": slide_id,
        })

    role_lookup = {}
    if speaker_profiles_path is not None:
        with open(speaker_profiles_path, "r", encoding="utf-8") as f:
            profiles_data = json.load(f)
        for p in profiles_data.get("speaker_profiles", []):
            role_lookup[p["speaker_id"]] = p["inferred_role"]

    if case_segmentation_path is not None:
        with open(case_segmentation_path, "r", encoding="utf-8") as f:
            case_data = json.load(f)

        output_dir = os.path.dirname(os.path.abspath(output_path))
        cases = case_data.get("cases", [])

        for case in cases:
            case_id = case["case_id"]
            case_start = case["start_sec"]
            case_end = case["end_sec"]

            case_utterances = [
                u for u in aligned_utterances
                if u["start_sec"] >= case_start and u["end_sec"] <= case_end
            ]

            merged = merge_consecutive_utterances(case_utterances)
            for utt in merged:
                utt["inferred_role"] = role_lookup.get(utt["speaker_id"], "other")

            output = {
                "video_id": video_id,
                "case_id": case_id,
                "aligned_utterances": merged,
            }

            case_output_path = os.path.join(output_dir, f"align_{case_id}.json")
            with open(case_output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            print(f"Case {case_id}: {len(merged)} merged utterances -> {case_output_path}")

    else:
        output = {
            "video_id": video_id,
            "aligned_utterances": aligned_utterances,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        matched = sum(1 for u in aligned_utterances if u["slide_id"] is not None)
        print(f"Aligned {matched}/{len(aligned_utterances)} utterances to slides.")
        print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Align transcript utterances to slides.")
    parser.add_argument("--slides", type=str, required=True, help="Path to slides.raw.json")
    parser.add_argument("--transcript", type=str, required=True, help="Path to transcript.with_speakers.json")
    parser.add_argument("--output", type=str, required=True, help="Path to output align.json (or output dir base when --case_segmentation is used)")
    parser.add_argument("--case_segmentation", type=str, default=None, help="Path to case_segmentation.json; generates per-case align_case_xxx.json files")
    parser.add_argument("--speaker_profiles", type=str, default=None, help="Path to speaker_profiles.json; adds inferred_role to each utterance")
    args = parser.parse_args()

    for path in (args.slides, args.transcript):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    if args.case_segmentation and not os.path.exists(args.case_segmentation):
        raise FileNotFoundError(f"File not found: {args.case_segmentation}")

    align(args.slides, args.transcript, args.output, args.case_segmentation, args.speaker_profiles)


if __name__ == "__main__":
    main()
