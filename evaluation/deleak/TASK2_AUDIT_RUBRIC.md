# Auditing the Task 2 question screening

Stage 3. An auditor re-reads stage 2's verdicts against the same evidence the
screener had and classifies each one. This is the same shape as
`AUDIT_RUBRIC.md`, applied to a different kind of fault.

`AUDIT_RUBRIC.md` audits **insertions** — the de-leak was allowed to delete and
some of it wrote instead, so the question there is "does the record support this
sentence?". Stage 2 is allowed to delete *and* to reword, and it decides what
leaves the benchmark. So the faults are different: a wrong verdict here either
leaves a leak in place, changes what was asked, or destroys an evaluation item.
The letters below are not the same letters.

## What the auditor is given

Per question: the de-leaked `case_summary` and `slides`, the removed spans, the
original question, the reference answer, and stage 2's verdict, edited text and
reason.

The auditor is **not** given stage 2's reasoning beyond the one-line reason, and
does not consult the screener. Agreement is only worth measuring between two
readings that were made separately.

## What to audit

Not a sample of everything. The two populations fail differently and need
different coverage:

- **Every non-keep verdict.** These are the visible changes, they are few, and
  each one either removed an item from the benchmark or altered its text.
- **A stratified sample of keeps.** A wrong keep is the quiet failure — the
  question is unchanged, nothing in the artifact looks edited, and the leak
  survives into published numbers. Sample by `qa_type`, and oversample the types
  whose questions most often narrate the case (`findings_interpretation`,
  `agreement_or_support`) over those that ask for a judgment
  (`treatment_recommendation`, `next_action_suggestion`).

Report the keep-sample size and the disagreement rate found in it. A rate
measured on 200 sampled keeps does not license a claim about the other ~2,000,
and the report should say which number is which.

## The classes

**A — correct and minimal.** The verdict is right, and no lighter verdict would
have served. Includes a keep that is genuinely leak-free and answerable.

**B — correct but heavier than needed.** The direction is right and the outcome
is not wrong, but a lighter verdict would have preserved more: a question
dropped where a trim would have left a usable item, or paraphrased where a trim
would have done the job. Nothing is corrupted; an evaluation item was spent that
did not have to be. Acceptable, and counted, because a screening that drifts
towards `drop` shrinks the benchmark without anyone deciding to.

**C — residual leakage.** The question still states a fact that is no longer in
the input. Kept when it should have been trimmed, or trimmed in the wrong place.
This is the failure the stage exists to prevent.

**D — the edit changed what was asked.** A trim or paraphrase that inverted a
negation, narrowed or widened the scope, or turned one clinical question into a
different one. The item still looks well-formed, so nothing downstream will
notice; the reference answer now grades a question nobody asked.

**E — the item no longer works.** Kept or trimmed, but the premise it needed is
gone and the reference answer is no longer derivable from the input — or the
answer never was, and the question asks for a post-decision observation that
should have been dropped. The item is unfair rather than leaky: every model
fails it, and the failure carries no information.

A and B are accepted. C, D and E are repaired.

## Calling the boundaries

**C against A on a kept question.** The test is not "does the question mention
something that appears in a removed span" — words survive elsewhere and the
spans are long. It is: does the question *assert*, as given fact, something a
reader of the de-leaked input could not otherwise know? A question that asks for
that fact is not C.

**E against A.** "Harder without the removed text" is A. "Not reachable by
clinical reasoning from what remains" is E. The reference answer is the guide: if
it is a judgment a board could argue for, the item works; if it is an
observation someone recorded after the meeting, it does not.

**B against A on a drop.** Ask what a trim would have left. If the residue would
have been a real question with a real answer, the drop is B. If it would have
been a stub — "What is the next step?" with no case anchoring — the drop is A.

**D against B.** D is a change in meaning; B is a change in weight. If the
audited text asks the same clinical question as the original, it is at worst B.

## Output format

```json
{"audited":[{"qa_id":"...","population":"edit","stage2_verdict":"drop","class":"B","note":"a trim to 'What is the role of consolidation?' would have left a usable item","proposed_verdict":"trim","proposed_final":"..."}]}
```

- `population` is `"edit"` or `"keep_sample"`.
- `class` is one of `A B C D E`.
- `note` is one line, and for C/D/E must name the specific fact, word or premise
  at issue rather than restating the class.
- `proposed_verdict` and `proposed_final` are required for C, D and E, and for B
  only when you are proposing the lighter verdict be adopted. Anything proposed
  must satisfy `TASK2_QA_SPEC.md` — the repair is re-checked by the same gate.

Do not edit stage 2's output file. The audit is a separate artifact so that the
disagreement between the two readings stays measurable.
