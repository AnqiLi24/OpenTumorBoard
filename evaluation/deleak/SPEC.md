# De-leaking tumor board case material

The judgment spec. An agent given a batch of cases follows this and nothing
else; `scripts/evaluation/deleak/verify_batch.py` enforces the mechanical rules
at the end.

Work by clinical judgment, case by case. Do not write or run a script that
pattern-matches phrases — read each case and decide. A script may APPLY
decisions already made per case (exact string replacements authored one at a
time); it may not make them.

Each input record has: `simulation_id`, `split`, `case_summary`,
`slide_captions` (a list), `reference_conclusion`.

## What you are removing

These case summaries and slide captions were generated from recordings that
often continue past the board meeting. They therefore sometimes state facts that
did not exist when the board convened: the pathology found at an operation the
board was authorizing, the response to a therapy it was deciding to start,
follow-up imaging after that decision, survival, death.

**The test is causal, not textual.** Read `reference_conclusion` only to learn
WHAT the board was deciding — the clinical question in front of it and the plan
it chose. Then ask of each statement: could this have been known at the moment
the board made that recommendation? **If the board selected plan X, the result
of X is post-decision leakage, whether or not the reference conclusion mentions
that result.**

This is the trap, and it is the main one. `reference_conclusion` is generated
from the SAME recording by the same model, and it is already known to narrate
events that happened after the meeting. Its coverage is NOT evidence that a fact
was available. Eight agents working independently all made this mistake before
the rule was stated this way, so check yourself against it explicitly. All of
these are leakage even when the conclusion narrates them:

- conclusion says the team selected chemoradiation then surgery; summary reports
  the surgical pathology or a pathologic complete response
- conclusion says neoadjuvant immunotherapy was chosen over upfront surgery;
  summary reports the excision showing no residual disease
- conclusion says a drug was started; summary reports the response to it
- conclusion says surveillance was chosen; summary reports the follow-up scan
- conclusion says a trial was proposed; summary reports enrolment and outcome
- conclusion endorses a plan already delivered; the result of that plan is still
  post-decision

## Finding the decision point

The slide captions are usually a more reliable marker than the conclusion. Decks
of this kind carry a discussion or poll slide stating the question the panel was
asked — "comparing endoscopic versus open approaches", "surgery / RT / both /
RAI", "response assessment and timing of possible retreatment". When you find
one, that question IS the decision point, and it tells you directly which
statements are inputs the panel reasoned from and which are results of the plan
it chose.

Treat an event as an input only when the board demonstrably reasoned from it —
never merely because the conclusion narrates it.

## The plan stated as already done

Two different things get confused here, so separate them.

**The result of the plan always goes.** "The mass responded", "lymphocytosis
resolved over a year", "0 of 15 nodes contained tumor" — remove, every time.

**The plan reported as already enacted usually goes too, for a different
reason.** If the summary says "Finasteride was initiated" and the board's
recommendation was to start finasteride, the input hands the model the answer it
is being asked to produce. That is leakage of a second kind, and just as
damaging here as leaked follow-up. Remove it.

The exception is a treatment that demonstrably started before this meeting and
formed part of the history the board reasoned from — a drug begun days before
the referral, a course completed elsewhere and now under review. That is
legitimate history and stays. Decide from the evidence, not from the wording.

Where naming the treatment is needed for the surrounding text to make sense,
prefer trimming to the clinical facts over deleting the whole passage.

A related distinction worth checking before you delete: a result can be the
*premise* of the question rather than the *outcome* of the answer. If a deck
shows post-treatment imaging and then asks "is this patient now a transplant
candidate?", the response shown is what the panel reasoned from, and the plan it
chose is the transplant. Keep the premise; there is no leakage unless the
transplant outcome is also reported.

## Longitudinal cases with several decisions

Some decks are full-course teaching presentations where the board decides more
than once. Apply the causal test to the **last** decision, and treat the earlier
lines of therapy and their outcomes as the history that motivated it. Testing
against the first decision would delete most of the disease course and leave the
later part of the conclusion unanswerable.

## Be conservative about everything else

The goal is to keep as much clinical information as possible. Do NOT remove:

- Prior treatments and their outcomes that happened BEFORE this meeting. A
  patient treated years ago whose recurrence brought them to this board — all of
  that is legitimate history and must stay, including how those treatments went.
- Disease progression that motivated the referral.
- Any test, biopsy, imaging or molecular result the board had in hand.
- Statements that something was pending, unavailable, or not yet done.
- Past tense by itself. The whole summary is past tense by design; tense is not
  evidence of leakage.
- Captions belonging to a different patient in the same session deck, and
  captions showing published trial data rather than this patient's course.

When you genuinely cannot tell whether a statement predates the decision, KEEP
IT. A false negative costs a little residual leakage; a false positive destroys
information the benchmark needs.

## How to edit

`case_summary` keeps a `[ section name ]:` structure. Delete only the leaking
sentence or clause, not the surrounding text. Then:

- If a section still has content, leave the rest exactly as it was, adjusting
  only a connective word if the deletion left the prose ungrammatical.
- If a section becomes empty, delete the whole section including its `[ ... ]:`
  header. **NEVER replace it with a statement of absence** such as "No prior
  treatment was documented" or "Systemic therapy had not yet been started" —
  that asserts something the source never said. This rule was violated in 19 of
  the first pass's 184 cases and every one had to be repaired; see
  `README.md` for what that cost.
- Every content word in your output must appear in the original. You may delete
  and you may fix a connective. You may not introduce a new clinical fact,
  hedge, or characterisation.

`slide_captions` are decided one at a time, by index, as keep / trim / drop:

- **keep** — nothing in it postdates the decision. This is the default.
- **trim** — it mixes leaked and legitimate content. Delete the leaking clause,
  keep the rest. PREFER THIS. A caption that reports an outcome but also states
  the question the board was deliberating gets trimmed to the question, not
  dropped.
- **drop** — the whole caption is post-decision content and nothing survives.

A trimmed caption must be a subsequence of the original words, keeping
`<image> ` if present.

## Output format

Write exactly this JSON to your output path, no prose around it:

```json
{"cases":[{"simulation_id":"...","summary_verdict":"clean","removed_sentences":[],"case_summary_final":"...","captions":[{"index":0,"verdict":"keep","final":"..."},{"index":1,"verdict":"trim","final":"...","reason":"short"}]}]}
```

`summary_verdict` is `"clean"` or `"edited"`. These rules are checked
mechanically, so satisfy them exactly:

- Every input case appears exactly once, in the input order.
- `"clean"` means `case_summary_final` is byte-identical to the input
  `case_summary` and `removed_sentences` is `[]`.
- Each entry in `removed_sentences` is verbatim from the input `case_summary`.
- `captions` has one entry per input caption, same indices, in order.
- `"keep"` means `final` is byte-identical to the input caption.
- `"drop"` means `final` is `""`.

Cover every case in the batch before writing. Two earlier attempts stopped
partway and left apply scripts covering a third of their cases; the uncovered
cases were silently emitted as `clean`, which the verifier cannot detect.

Report at the end how many cases you edited and how many captions you trimmed or
dropped, with one line of reasoning per edited case.
