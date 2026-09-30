# Auditing rewritten case summaries

A de-leak pass was supposed to REMOVE post-decision statements from tumor board
case summaries. Its spec said: you may delete, and you may fix a connective word
left ungrammatical by a deletion. You may not introduce a new clinical fact,
hedge, or characterisation. If a section is emptied, delete the section and its
`[ ... ]:` header — never replace it with a statement of absence.

Some of its output does not satisfy that. Your job is to say exactly how far it
departed, per case, with evidence. You are NOT re-doing the de-leak and you are
NOT judging whether the right things were removed. Only: did the rewrite change
or invent information?

Each record gives you `original_case_summary`, `deleaked_case_summary`,
`introduced_words` (content words in the new text that are absent from the
original summary), plus `slide_captions`, `reference_discussion` and
`reference_conclusion` — the rest of the source record.

## Classify every case into exactly one class — the most severe that applies

- **A — deletion only.** Text removed; anything remaining is verbatim. At most a
  connective word changed (`,` → `and`). No new information.
- **B — restatement.** New wording, but every clinical fact it asserts is stated
  somewhere in the source record (summary, captions, discussion or conclusion).
  A synonym or a grammatical variant of a source word counts as supported.
  Nothing new is claimed.
- **C — inferred negative.** Asserts that something had NOT happened yet
  ("systemic therapy had not yet been started", "definitive surgery had not yet
  been performed"). The source never states it; it is an inference from the
  timeline. Consistent with the record but unsupported by it.
- **D — unsupported new fact.** Asserts a clinical fact, characterisation or
  qualifier that appears nowhere in the record and does not follow from it.
  Examples of the kind of thing that qualifies: a performance status, a
  functional level, a disease descriptor such as "residual", "recurrent",
  "newly diagnosed" or "untreated", a severity qualifier such as "mild" — when
  the record does not state it.
- **E — altered fact.** The new text contradicts the record or changes a value,
  laterality, stage, count, date or clinical meaning.

Judge by meaning, not by string matching. `introduced_words` is a starting
pointer only: a word can be new while the fact is supported (class B), and a
fact can be invented using only words that appear elsewhere (class D or E).
Negation matters — read the whole sentence before deciding a fact was inverted.

Be accurate rather than harsh, and accurate rather than lenient. If a case is
genuinely borderline between two classes, choose the LOWER severity and say why
in `note`.

## Output

Write to your output path, this JSON and nothing else:

```json
{"cases":[{"simulation_id":"...","class":"A","note":"one sentence","evidence":[{"new_text":"the exact new sentence or clause","status":"supported|inferred|unsupported|contradicted","where_supported":"quote from the record, or null"}]}]}
```

Rules checked mechanically: every input case appears exactly once in input
order; `class` is one of A B C D E; every `new_text` is a verbatim substring of
`deleaked_case_summary`; every non-null `where_supported` is a verbatim
substring of the source record. For class A, `evidence` may be `[]`.

For C, D and E, quote every offending clause — those are the ones that will be
repaired, so a missed clause stays in the benchmark.

End your reply with a count per class and the ids in each of C, D and E.
