# Screening Task 2 questions against a de-leaked case input

The judgment spec for stage 2 of the Task 2 redo. An agent given a batch of
cases follows this and nothing else; `scripts/evaluation/deleak/verify_qa_batch.py`
enforces the mechanical rules at the end.

Work by clinical judgment, question by question. Do not write or run a script
that pattern-matches phrases — read each question and decide. A script may APPLY
decisions already made (exact string replacements authored one at a time); it
may not make them.

## Why this stage exists

Task 1's de-leak removed post-decision facts from `case_summary` and `slides`.
Task 2 shows a model those same two fields plus a `question` — and the questions
were written from the **canonical** summaries by `atomic_qa.py`, then made
self-contained by `rephrase_qa.py`, whose instruction was to rewrite each one
"using only the case context". That instruction copies facts out of the summary
and into the question. So a sentence deleted from the input can still be sitting
verbatim in the question printed underneath it.

Removing it from one field and leaving it in the other de-leaks nothing.

## What you are given

Per case: the de-leaked `case_summary` and `slides` **exactly as a model will
see them**, the spans the de-leak removed, the reference conclusion, and every
question on that case with its reference answer.

`removed_spans` is the evidence, not the answer. It tells you what stopped being
visible. Whether a particular question is damaged by that is the thing you are
deciding.

You are not re-deciding Task 1. Do not argue that a removed span should have
been kept; work from the input as it now stands.

## The two conditions

A question survives only if both hold.

**1 — It does not state a removed fact.** If the input no longer says the
follow-up MRI showed no regrowth, the question may not say it either. The leak
does not care which field it is printed in.

**2 — It is answerable in principle from what remains.** Not "the model can
reproduce the reference answer verbatim" — that is never the standard. The test
is whether a board sitting at the decision point, holding the de-leaked input,
could reason its way to the reference answer as a clinical judgment.

Condition 2 fails two ways, and they need different verdicts:

- **The question asks for a post-decision observation.** A follow-up scan, the
  pathology of an operation the board was authorizing, the response to a therapy
  it was deciding to start, survival, death. No reasoning reaches these; they
  were observed, not inferred. The question is asking the model to recite a
  result. Nothing rescues it — **drop**.
- **The question depends on a premise that was removed.** "Given the pathology
  showing X, what next?" with X gone. If the rest of the question stands on its
  own, cut the premise. If it does not, drop it.

## The distinction that decides most cases

**A question that STATES a removed fact leaks. A question that ASKS FOR one
usually does not — it is the benchmark working.**

Real example, kept:

> Given that she was not considered a candidate for chemotherapy, what was the
> initial systemic treatment plan?
> *removed span:* "Afatinib was being pursued as initial systemic therapy."
> *reference answer:* "The initial plan was to obtain afatinib for this patient."

The de-leak removed the plan because the plan-as-enacted hands the model its own
answer. The question does not name afatinib. After the de-leak it asks a model
to produce a recommendation from the clinical picture, graded against what the
board chose. That is exactly what this benchmark is for. **Keep.**

Real example, dropped:

> What has the one-year follow-up shown regarding regrowth after surgery for
> this giant pituitary macroadenoma?
> *removed span:* "Follow-up MRI was reported to show no regrowth over 1 year."

Also asks for a removed fact — but the fact is an observation made a year after
the meeting. No board could answer it, and no reasoning from the de-leaked input
gets there. **Drop.**

The line between them is not "was the fact removed". It is **could a competent
board at the decision point have reasoned to this answer**.

## Be conservative — the flags are noisy by design

Stage 1 narrowed for recall and its signal is weak. A word that vanished from
one sentence often survives the case elsewhere, and questions get flagged for
nothing. Two patterns to expect:

**Operative detail removed, planning question kept.** One case lost a long
operative note ("total parotidectomy, facial nerve mobilization, transparotid
approach…"), which took `facial`, `nerve` and `surgical` out of the visible
vocabulary. Six questions were flagged. All six are pre-operative planning —
"how would you manage the facial nerve during resection?", "how would you obtain
carotid control?" — which is precisely what the board was deliberating, from
anatomy that is still in the input. **Keep all of them.** The operation's
conduct is post-decision; the plan for it is the decision.

**Literature questions.** "How do outcomes with single-agent HER2-directed
therapy compare with carboplatin/paclitaxel plus trastuzumab?" is about
published evidence, not this patient. Nothing about it changes when a patient
fact is removed. **Keep.**

When you genuinely cannot tell, KEEP. A residual leak in one question dilutes a
score slightly; a wrongly dropped question destroys an evaluation item, and 2,494
of them are being read here.

## Verdicts, in order of preference

Reach for the first one that works. Do not skip to a stronger verdict because it
is easier.

**keep** — the question states nothing removed and remains answerable. The
default, and it should be the large majority.

**trim** — the question carries a removed fact in a clause the rest does not
need. Delete that clause; keep the rest untouched. A trimmed question must be a
subsequence of the original words and must still end in `?`.

> Given that follow-up MRI showed no regrowth, how should surveillance be
> structured for this patient?

trims to "How should surveillance be structured for this patient?" — every word
kept is a word the original had, in its original order.

**Check what a surviving "the …" now points at.** This is the one way a valid
trim silently changes the question, and the subsequence rule cannot see it. Cut
`use of osimertinib-based` from

> Do you support the current use of osimertinib-based treatment for this
> patient with EGFR exon 19 deletion and acquired T790M?

and you are left with "the current **treatment**" — which, in a de-leaked input
where the only named treatment is the afatinib the patient progressed on, now
asks about afatinib. The reference answer still endorses osimertinib, so the
item grades an answer to a question nobody asked. The fix is to cut a different
span (`the current use of`), not to drop the item.

After any trim, read the residue **against the de-leaked input alone** and check
that every definite reference — "the regimen", "the operation", "this therapy" —
still binds to what it bound to before. If it re-binds, cut elsewhere.

**And check what still places the question in time.** The same fault has a
second form, seen twice: the leading clause carries both an assertion and the
item's only temporal anchor, and cutting the whole clause moves the question to
a different moment in the case.

> When the ulcerative lesion became foul-smelling and pain was unbearable **at
> the January 2025 visit**, what symptom-directed measures were recommended?

trimmed to "What symptom-directed measures were recommended?" is a legal
subsequence that removes the leaked premise — and the case documents
symptom-directed measures at four other timepoints, all still visible, so the
residue is answered by the input while the reference answer grades only the
January ones. Same shape as "After **this salvage** resection **with negative
margins**, what next?" cut to "What next?", which re-seats the question before
the operation, where the answer is "operate" rather than the reference's "refer
for systemic therapy".

Cut the assertions *inside* the clause and keep the anchor. "After **resection**
for recurrent papillary RCC, …" — no determiner, so it asserts no particular
operation, but the question still happens after one.

**paraphrase** — reach for this ONLY when trimming cannot produce a grammatical
question that means what the original meant. This is rare; the subsequence rule
already permits dropping whole leading clauses, mid-sentence appositives and
trailing qualifiers. If you find yourself paraphrasing often, you are probably
rewriting questions that a trim would have handled, and the count is reported so
that this is visible.

A paraphrase may use only words that are already visible: every content word
must appear in the de-leaked `case_summary`/`slides`, or in the original
question minus the words the de-leak removed. This is checked mechanically. It
exists so that no paraphrase can quietly import a fact from
`reference_discussion` or `reference_conclusion` — those are not de-leaked and
narrate the whole meeting.

The gate is lexical, not semantic. It cannot stop you restating a removed fact
in surviving words. That is your job.

**drop** — the question asks for a post-decision observation, or its premise was
removed and nothing coherent survives, or trimming would leave a question the
reference answer no longer answers. Dropped questions leave the benchmark.

Do not drop merely because a question became harder. Harder is the point.

## What you may not touch

`reference_answer`, `reference_quote`, `qa_type`, `target_specialist_role` and
`qa_id` are not yours to edit, and neither is the case input — Task 1 settled
that. You are editing question text and nothing else. The output format has no
field for anything else, and the gate rejects a batch that tries.

`reference_answer` being a post-decision fact is not itself a problem — the
answer is the grading target, never model input. It is a *diagnostic*: an answer
that is purely an observation from after the meeting usually means the question
should drop under condition 2.

## Output format

Write exactly this JSON to your output path, no prose around it:

```json
{"cases":[{"video_uid":"...","case_id":"...","questions":[{"qa_id":"...","verdict":"keep"},{"qa_id":"...","verdict":"trim","final":"...","reason":"short"},{"qa_id":"...","verdict":"drop","final":"","reason":"short"}]}]}
```

Checked mechanically, so satisfy these exactly:

- Every input case appears exactly once, in the input order, and every question
  under it exactly once, in the input order.
- `"keep"` takes **no `final` and no `reason`** — just the id and the verdict.
  Most of these questions are keeps, and re-typing each one to say "unchanged"
  would make copying the main activity of the batch, which is how a keep turns
  into an accidental reword. If you do supply `final` it must be byte-identical.
- `"trim"` means `final` is shorter than the original, is a word subsequence of
  it, is non-empty and ends with `?`.
- `"paraphrase"` means `final` differs from the original, ends with `?`, and
  introduces no content word outside the allowed vocabulary above.
- `"drop"` means `final` is `""`.
- `trim`, `paraphrase` and `drop` each need a one-line `reason`.

One further rule the gate enforces on trims: you may not produce a near-copy of
the original that has lost a negation. Deleting `not` from a question is a valid
subsequence and inverts its meaning, and no other rule here would catch it.
Deleting a whole negated clause is fine; deleting the negator and keeping the
clause is not.

Cover every question in the batch before writing. An earlier stage of this
pipeline had two agents stop partway and emit their uncovered cases as unchanged
— well-formed, and wrong in a way no gate can see.

Report at the end: questions read, and counts of keep / trim / paraphrase / drop
broken down by `qa_type`, with one line of reasoning per non-keep verdict.
