# Repairing a rewritten case summary

A de-leak pass was meant to delete post-decision statements. In these cases it
instead rewrote a section, and the rewrite asserts something the source record
does not say. An audit classified each offending clause. You are repairing them.

The repair is DELETION ONLY. You may not write a replacement clause, a hedge, or
a statement of absence. That prohibition is the whole point: the previous pass
broke this benchmark's inputs by writing "systemic therapy had not yet been
started" where it should have deleted a section, and asserting a nodal station
the imaging never showed.

## What you are given

Each record has `simulation_id`, `original_case_summary` (the source, before any
de-leak), `deleaked_case_summary` (the current, defective text), and `findings`
— the audited clauses, each with `new_text`, its class and a note.

## What to do

Produce `repaired_case_summary` from `deleaked_case_summary` by deleting the
offending clauses.

- **C — inferred negative.** Delete the sentence. It asserts that something had
  not happened; the source never says so.
- **D — unsupported new fact.** Delete the unsupported element. If it is a
  modifier inside an otherwise supported sentence ("Newly diagnosed metastatic
  breast carcinoma...", "suspicious central and right lateral neck adenopathy"),
  delete only that modifier and keep the rest. Do not delete a whole sentence
  when cutting two words suffices.
- **E — altered fact.** Delete the altered statement. Do NOT restore the
  original wording: the audit found it altered precisely because the underlying
  fact was post-decision content the de-leak was right to remove. Removing the
  statement leaves the case silent on it, which is correct.

Then repair the surroundings, by deletion only:

- If a section is left with no content, delete the whole section including its
  `[ ... ]:` header. Never leave a bare header, and never write a sentence
  saying the section is empty.
- If a deletion leaves the prose ungrammatical, you may adjust or remove a
  connective word (`,` to `and`, dropping a stranded "and"). Nothing else.

## The hard constraint

**Every content word in `repaired_case_summary` must already appear in
`deleaked_case_summary`.** You are only ever removing. This is checked
mechanically and a violation fails the batch.

Note that the repaired text is checked against the DE-LEAKED summary, not the
original. Words the rewrite introduced legitimately (class B restatements) are
allowed to stay — you are not reverting the rewrite, only cutting what the audit
flagged.

## Output

Write exactly this JSON to your output path, no prose around it:

```json
{"cases":[{"simulation_id":"...","repaired_case_summary":"...","deleted":["exact span removed","..."],"note":"one sentence"}]}
```

- Every input case appears exactly once, in input order.
- Each entry in `deleted` is verbatim from `deleaked_case_summary`.
- `repaired_case_summary` must be shorter than `deleaked_case_summary`.

Report at the end: cases repaired, and for each one the clause you cut.
