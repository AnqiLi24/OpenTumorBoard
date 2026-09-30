"""Repair, rephrase and filter already-generated QA pairs — WITHOUT touching the
original task files, the canonical sharegpt, or any existing script.

This is a purely additive, opt-in pass. It reads the base ``task_case_*.json``
files (which already contain ``qa_pairs`` with question/answer/quote/type) under
a processed video directory and, for every QA pair, the model returns one of:

  KEEP  -> the pair is sound. Two edits are applied:
    * QUESTION REPAIR — resolve in-room deixis ("this", "that study", "the rib")
      and complete truncated questions into a self-contained clinical question,
      using only the case context. Guardrails: never leak the answer into the
      question, never change its difficulty, and never invent an entity (trial /
      drug / lesion) that is not recoverable from the evidence — if a referent
      cannot be resolved, the pair is DROPPED as ``unanswerable`` instead.
    * ANSWER REWRITE — rewrite the spoken answer into a clean, declarative
      clinical statement that is strictly entailed by the verbatim ``quote``.

  DROP  -> the pair is intrinsically broken. One ``drop_reason`` is recorded:
    * non_answer  : speaker only defers / "I don't know" / pure speculation.
    * tangential  : answer does not address the question (only restates a
                    condition or premise from it).
    * circular    : answer merely restates the question's premise or a fact
                    already in the case summary — no interpretation / reasoning.
    * trivial     : a non-expert could answer from the case summary, or it is a
                    yes/no with no reasoning, or generically true for any patient.
    * open_ended  : so open that many different answers are equally acceptable,
                    so the single reference answer is not authoritative.
    * unanswerable: answering needs facts not in the case summary, slides, or
                    quote, or the question references an entity absent from the
                    evidence — the test-taker could neither answer nor be graded.
    * non_clinical: the question/answer turns on operational, logistical, or
                    pandemic-era resource constraints rather than transferable
                    clinical reasoning — time-bound, not durable knowledge.

The verbatim ``quote`` is preserved unchanged as the evidence anchor; the
original spoken answer and original question are kept under ``answer_verbatim``
and ``question_verbatim``.

Outputs (originals are never modified):
  - task_case_XXX_rephrased.json          kept pairs, question + answer repaired
  - task_case_XXX_rephrased_dropped.json  dropped pairs + drop_reason (audit)
  - sharegpt_rephrased.json               full sharegpt rebuilt from kept pairs

The directory argument may be a single processed video directory (containing
task_case_*.json) or a ROOT holding many such video subdirectories.

Usage:
    python src/task_generator/rephrase_qa.py [--workers N] [--skip-existing] \
        ["<root or video dir>"]

Set DRY_RUN=1 to process and report without writing any files.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # src/            -> gpt_client
sys.path.insert(0, str(_HERE))         # task_generator/ -> convert_sharegpt
from gpt_client import client, DEPLOYMENT  # noqa: E402
# Reuse the canonical sharegpt builders so the rephrased sharegpt is byte-format
# identical to the original — convert_sharegpt.py itself is NOT modified.
from convert_sharegpt import (  # noqa: E402
    _build_setting1_sample,
    _build_setting2_samples,
)

DEFAULT_DIR = "data/processed"

MAX_WORKERS = 16  # concurrent per-case requests (LLM-bound, no GPU)
DRY_RUN = os.environ.get("DRY_RUN") == "1"

TASK_SUFFIX = "_rephrased"                       # task_case_XXX_rephrased.json
DROPPED_SUFFIX = "_rephrased_dropped"            # ..._rephrased_dropped.json
SHAREGPT_OUT = "sharegpt_rephrased.json"

DROP_REASONS = (
    "non_answer", "tangential", "circular",
    "trivial", "open_ended", "unanswerable", "non_clinical",
)

SYSTEM_PROMPT = (
    "You are a meticulous clinical editor curating a molecular tumor board QA "
    "benchmark. The model being tested will see ONLY the patient case summary "
    "and slides, and must answer a single self-contained question as the expert "
    "would. For each QA pair you either KEEP it — repairing the question and "
    "rewriting the answer — or DROP it with a reason. You respond only with JSON."
)

INSTRUCTIONS = """\
Each QA pair gives: qa_type, the QUESTION (often a lightly cleaned transcript
question that may contain in-room deixis), the spoken ANSWER (raw transcript),
and the verbatim evidence QUOTE. Decide KEEP or DROP for each, by index.

═══ DROP the pair (set "verdict":"drop" and one "drop_reason") if ANY holds ═══
  non_answer  : the answer's core is that the speaker does not know, has not
                checked, will find out later, defers to someone else, or is pure
                speculation with no clinical substance ("I don't know", "I'd have
                to check", "what do you think?").
  tangential  : the answer does not actually address what the question asks — it
                only restates a condition or premise from the question.
                e.g. Q "if sent to you for radiation, what would you do?" /
                A "he has already received 71.8 Gy" (states a constraint, never
                says what you'd do).
  circular    : the answer merely restates the question's premise, or a fact
                already in the case summary, WITHOUT the interpretation,
                reasoning, or decision the question asks for.
                e.g. Q "how do you interpret the negative 9-11 gene panel?" /
                A "the 9-11 gene panel was negative" (no interpretation).
  trivial     : a non-clinician could answer it directly from the case summary;
                OR it is a yes/no with no reasoning; OR the answer is generically
                true for any patient and not specific to this one; OR the
                question looks specific but its real intention is clinical common
                sense once the answer is read.
                e.g. "how does finding an actionable alteration change treatment
                timing?" / "targeted treatment can start immediately"; or "when
                would you NOT enroll a patient despite meeting the growth
                requirement?" / "if it felt too risky to give them placebo".
  non_clinical: the question (or the answer's substance) turns on operational,
                logistical, administrative, scheduling, institutional, or
                pandemic-era RESOURCE constraints rather than transferable
                clinical reasoning about THIS patient. Such items are time-bound
                and not durable clinical knowledge, so they do not belong in the
                benchmark.
                e.g. "how did COVID-19 resource constraints alter management?" /
                "the CT-guided biopsy was deemed nonessential and radiology would
                not perform it; the infusion unit closed for bed space". A purely
                clinical decision that merely MENTIONS the pandemic in passing is
                NOT non_clinical — drop only when the constraint IS the substance.
  open_ended  : so open-ended that many different answers would be equally
                acceptable, so the single reference answer is not authoritative
                and the item cannot be graded.
                e.g. "what is your routine next step regarding molecular testing
                for a new patient?".
  unanswerable: answering requires facts NOT present in the case summary, slides,
                or quote; OR the question references a finding/entity absent from
                the provided evidence; OR a needed referent cannot be resolved
                from the evidence. The test-taker could neither answer nor be
                graded.
                e.g. "why wasn't the kinase domain duplication seen on the
                earlier 11-gene panel?" when no such finding is in the evidence;
                a general tracer-physics digression (Ga-68 vs F-18) not tied to
                this patient.
When uncertain between keep and drop, prefer to KEEP (be conservative). A
substantive statement about a clinical UNKNOWN and how it changes management
(the `uncertainty` type) is a real answer — keep it.

═══ Otherwise KEEP (set "verdict":"keep") and produce a repaired pair ═══
"question": repair the question into a SELF-CONTAINED, complete clinical question.
  - Resolve every in-room / slide referent using the case context: "this" / "this
    lesion" -> name it ("the indeterminate rib focus on PSMA PET"); "that study"
    / "the study" -> the specific trial if recoverable from the evidence; "the
    rib though?" -> complete it ("What about radiating the rib?").
  - GUARDRAILS: do NOT leak or hint at the answer; do NOT make it easier or
    harder; do NOT invent a trial/drug/entity name that is not in the evidence
    (if a referent truly cannot be resolved, DROP as unanswerable instead).
  - If the question is already self-contained, return it unchanged.
"answer": rewrite the ANSWER into one or a few complete, DECLARATIVE sentences
  that directly answer the (repaired) question, in clean written clinical prose.
  - Remove filler, false starts, spoken tics ("um", "you know", "sort of", "we
    can squint and"), and slide-/room-dependent deixis ("you can see here").
  - First-person clinical voice is fine; never name or label a participant
    ("the radiologist", "Dr. X").
  - STRICT FIDELITY: include ONLY clinical content present in the QUOTE. Do NOT
    add facts, numbers, certainty, or recommendations the quote lacks; do NOT
    strip genuine clinical uncertainty that is part of the content.

Respond with JSON of the form
{"results": [
  {"index": 0, "verdict": "keep", "question": "<repaired>", "answer": "<rewritten>"},
  {"index": 1, "verdict": "drop", "drop_reason": "circular"}
]}
with exactly one entry per input index.\
"""


def build_prompt(case_summary: str, slides: str, qa_pairs: list[dict]) -> str:
    items = [
        {
            "index": i,
            "qa_type": qa.get("type", ""),
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "evidence_quote": qa.get("quote", ""),
        }
        for i, qa in enumerate(qa_pairs)
    ]
    ctx = f"PATIENT CASE SUMMARY (shared context for all pairs below):\n{case_summary.strip()}\n"
    if slides and slides.strip():
        ctx += f"\nPRESENTATION SLIDES:\n{slides.strip()}\n"
    return (
        f"{INSTRUCTIONS}\n\n{ctx}\n"
        "Process every QA pair in this JSON array:\n"
        f"{json.dumps(items, ensure_ascii=False, indent=1)}"
    )


def _norm_reason(reason) -> str:
    if isinstance(reason, str) and reason.strip().lower() in DROP_REASONS:
        return reason.strip().lower()
    return "non_answer"  # default bucket for an unlabeled drop


def rephrase_case(case_summary: str, slides: str, qa_pairs: list[dict]) -> list[dict]:
    """Return one decision dict per qa pair, aligned by index:

        {"keep": bool, "question": str, "answer": str, "drop_reason": str|None}

    On any failure the pair is kept with its original question/answer (fail-safe:
    never silently lose data to a model/parse error).
    """
    results = [
        {
            "keep": True,
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "drop_reason": None,
        }
        for qa in qa_pairs
    ]
    if not qa_pairs:
        return results

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(case_summary, slides, qa_pairs)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    parsed = json.loads(resp.choices[0].message.content)
    for r in parsed.get("results", []):
        idx = r.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(qa_pairs)):
            continue
        orig = qa_pairs[idx]
        if str(r.get("verdict", "keep")).lower() == "drop":
            results[idx] = {
                "keep": False,
                "question": orig.get("question", ""),
                "answer": orig.get("answer", ""),
                "drop_reason": _norm_reason(r.get("drop_reason")),
            }
        else:
            q = r.get("question", "")
            a = r.get("answer", "")
            results[idx] = {
                "keep": True,
                "question": q if isinstance(q, str) and q.strip()
                else orig.get("question", ""),
                "answer": a if isinstance(a, str) and a.strip()
                else orig.get("answer", ""),
                "drop_reason": None,
            }
    return results


def _case_id(task_path: str) -> str:
    return os.path.basename(task_path).replace("task_", "").replace(".json", "")


def process_case(task_path: str, video_id: str):
    """Process one task_case file. Returns
    (case_id, new_task, dropped_pairs, total, kept, reason_counter, task_path)."""
    case_id = _case_id(task_path)
    with open(task_path, encoding="utf-8") as f:
        task_data = json.load(f)

    qa_pairs = task_data.get("qa_pairs", [])
    decisions = rephrase_case(
        task_data.get("case_summary", ""),
        task_data.get("slides", ""),
        qa_pairs,
    )

    kept_pairs: list[dict] = []
    dropped_pairs: list[dict] = []
    reasons: Counter = Counter()

    for qa, d in zip(qa_pairs, decisions):
        if not d["keep"]:
            reasons[d["drop_reason"]] += 1
            dropped = dict(qa)
            dropped["drop_reason"] = d["drop_reason"]
            dropped_pairs.append(dropped)
            continue
        new = dict(qa)
        new["question_verbatim"] = qa.get("question", "")  # original question
        new["question"] = d["question"]                    # repaired, self-contained
        new["answer_verbatim"] = qa.get("answer", "")      # original spoken answer
        new["answer"] = d["answer"]                        # rewritten, declarative
        # `quote` is left exactly as-is — it stays the verbatim evidence anchor.
        kept_pairs.append(new)

    new_task = dict(task_data)
    new_task["qa_pairs"] = kept_pairs
    new_task["rephrase_summary"] = {
        "total": len(qa_pairs),
        "kept": len(kept_pairs),
        "dropped": len(dropped_pairs),
        "dropped_by_reason": dict(reasons),
    }
    return (case_id, new_task, dropped_pairs, len(qa_pairs),
            len(kept_pairs), reasons, task_path)


def render_progress(done: int, total: int, errors: int, start: float) -> None:
    """Draw a single-line, self-updating progress bar on stderr."""
    elapsed = time.time() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    width = 28
    filled = int(width * done / total) if total else width
    bar = "█" * filled + "░" * (width - filled)
    pct = (done * 100 // total) if total else 100
    msg = (f"\r  [{bar}] {done}/{total} cases {pct:>3}%  "
           f"{rate:4.1f}/s  ETA {eta:4.0f}s")
    if errors:
        msg += f"  {errors} err"
    sys.stderr.write(msg)
    sys.stderr.flush()


def base_task_files(d: Path) -> list[str]:
    """Base task_case_*.json files in a dir (excluding _qc / _rephrased variants)."""
    return sorted(
        p for p in glob.glob(str(d / "task_case_*.json"))
        if "_qc" not in os.path.basename(p)
        and "_rephrased" not in os.path.basename(p)
    )


def discover_video_dirs(root: Path) -> list[Path]:
    """A single video dir if `root` holds task_case_*.json, else its subdirs."""
    if base_task_files(root):
        return [root]
    return sorted(
        sub for sub in root.iterdir()
        if sub.is_dir() and base_task_files(sub)
    )


def build_sharegpt(video_id: str, cases: list) -> list[dict]:
    """Rebuild the full sharegpt (S1 simulation + S2 repaired QA) for one dir.

    `cases` is a list of (case_id, new_task, task_path). Setting-1 simulation
    samples come from the untouched align files (unaffected by this pass);
    Setting-2 samples come from the kept, repaired qa_pairs.
    """
    samples: list[dict] = []
    for case_id, new_task, task_path in sorted(cases):
        align_path = task_path.replace("task_", "align_")
        align_data: dict = {}
        if os.path.exists(align_path):
            with open(align_path, encoding="utf-8") as f:
                align_data = json.load(f)
        s1 = _build_setting1_sample(new_task, align_data, video_id, case_id)
        if s1:
            samples.append(s1)
        samples.extend(_build_setting2_samples(new_task, video_id, case_id))
    return samples


def write_dir_outputs(target: Path, results: list, verbose: bool):
    """Write per-case _rephrased + _rephrased_dropped files + sharegpt for one dir.

    `results` is a list of process_case() return tuples. Returns
    (total, kept, dropped, reason_counter)."""
    total = total_kept = total_dropped = 0
    reasons: Counter = Counter()
    cases_for_sharegpt = []

    for (case_id, new_task, dropped_pairs, total_c,
         kept_c, reasons_c, task_path) in sorted(results, key=lambda r: r[0]):
        total += total_c
        total_kept += kept_c
        total_dropped += len(dropped_pairs)
        reasons.update(reasons_c)
        cases_for_sharegpt.append((case_id, new_task, task_path))

        if not DRY_RUN:
            out_path = task_path.replace(".json", f"{TASK_SUFFIX}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(new_task, f, ensure_ascii=False, indent=2)
            # Audit sidecar — only when something was dropped.
            if dropped_pairs:
                drop_path = task_path.replace(".json", f"{DROPPED_SUFFIX}.json")
                with open(drop_path, "w", encoding="utf-8") as f:
                    json.dump(dropped_pairs, f, ensure_ascii=False, indent=2)

        if verbose:
            rc = ",".join(f"{k}:{v}" for k, v in sorted(reasons_c.items()))
            print(f"  {case_id:<14}{total_c:>5}{kept_c:>7}{len(dropped_pairs):>10}"
                  f"   {rc}")

    if not DRY_RUN:
        samples = build_sharegpt(target.name, cases_for_sharegpt)
        with open(target / SHAREGPT_OUT, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    return total, total_kept, total_dropped, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "dir", nargs="?", default=DEFAULT_DIR,
        help="A processed video dir, or a root holding many of them.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help=f"Skip video dirs that already have {SHAREGPT_OUT}.",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Concurrent requests (default {MAX_WORKERS}).",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        sys.exit(f"Directory not found: {root}")

    video_dirs = discover_video_dirs(root)
    if args.skip_existing:
        kept_dirs = [d for d in video_dirs if not (d / SHAREGPT_OUT).exists()]
        skipped = len(video_dirs) - len(kept_dirs)
        video_dirs = kept_dirs
        if skipped:
            print(f"Skipping {skipped} dir(s) with existing {SHAREGPT_OUT}.")
    if not video_dirs:
        sys.exit(f"No video dirs with task_case_*.json under {root}")

    single = len(video_dirs) == 1
    print(f"Root  : {root}")
    print(f"Mode  : repair question + rewrite answer + drop broken pairs (originals untouched)")
    print(f"Dirs  : {len(video_dirs)} video dir(s){'   [DRY RUN]' if DRY_RUN else ''}\n")

    # One shared pool over all (dir, case) jobs across every video dir.
    jobs = [(d, tp) for d in video_dirs for tp in base_task_files(d)]
    results_by_dir: dict[Path, list] = {d: [] for d in video_dirs}
    errors = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(process_case, tp, d.name): (d, tp)
            for d, tp in jobs
        }
        done = 0
        render_progress(0, len(jobs), 0, start)
        for fut in as_completed(futures):
            d, tp = futures[fut]
            done += 1
            try:
                results_by_dir[d].append(fut.result())
            except Exception as e:
                errors += 1
                sys.stderr.write("\n")  # don't let the bar clobber the error
                print(f"  ! {os.path.basename(tp)} in {d.name}: {e}",
                      file=sys.stderr)
            render_progress(done, len(jobs), errors, start)
    sys.stderr.write("\n")  # finish the progress line

    g_total = g_kept = g_dropped = 0
    g_reasons: Counter = Counter()
    if single:
        print(f"  {'case':<14}{'total':>5}{'kept':>7}{'dropped':>10}   reasons")
        print("  " + "-" * 50)
    for i, d in enumerate(video_dirs, 1):
        if not results_by_dir[d]:
            continue
        t, k, dr, rc = write_dir_outputs(d, results_by_dir[d], verbose=single)
        g_total += t
        g_kept += k
        g_dropped += dr
        g_reasons.update(rc)
        if not single:
            print(f"[{i}/{len(video_dirs)}] {d.name[:46]:<46} "
                  f"total={t:>4} kept={k:>4} dropped={dr:>4}")

    print("\n" + "=" * 50)
    print(f"Summary: {len(video_dirs)} dir(s), {g_dropped} dropped, "
          f"{g_kept} kept (of {g_total} QA pairs)."
          + (f"  {errors} case error(s)." if errors else ""))
    if g_reasons:
        print("Dropped by reason: "
              + ", ".join(f"{r}={g_reasons.get(r, 0)}" for r in DROP_REASONS
                          if g_reasons.get(r)))
    if DRY_RUN:
        print("[DRY RUN] no files written.")


if __name__ == "__main__":
    main()
