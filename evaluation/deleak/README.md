# De-leaking protocol

The case summary and slide captions of a case are generated from the whole
recorded discussion, and recordings often continue past the board's decision.
De-leaking removes every fact that was not available when the board made its
decision, so that no input reveals or implies the conclusion a model is scored
against. This directory specifies the protocol; the tooling that runs it is not
distributed.

## What is edited

Only the inputs are edited: `case_summary` and `slides` for both settings, and
the question for Specialist Turn. The reference discussion and the reference
conclusion are never edited, since they are what the board actually said.

Every edit is a deletion. An output passes only if

- an unchanged field is byte-identical to its input,
- every removed span is verbatim from the input,
- a trimmed caption is a subsequence of the original words,
- an emptied section also loses its header, and
- no content word appears that was not in the input.

## Board Simulation inputs

| Step | Specification |
|---|---|
| 1. Split the cases into work batches | — |
| 2. De-leak each case summary and its slide captions | `SPEC.md` |
| 3. Rebuild the input files from the de-leaked cases | — |
| 4. Audit the edited cases | `AUDIT_RUBRIC.md` |
| 5. Repair what the audit flags | `REPAIR_SPEC.md` |

## Specialist Turn questions

Questions were written from the case summary, so a fact deleted from the
summary can remain in a question. Every question on an edited case is
therefore screened as well.

| Step | Specification |
|---|---|
| 1. Take the de-leaked case inputs | — |
| 2. Select the questions on cases whose inputs were edited | — |
| 3. Screen each selected question: keep, trim or drop | `TASK2_QA_SPEC.md` |
| 4. Audit the screening and apply its repairs | `TASK2_AUDIT_RUBRIC.md` |
| 5. Rebuild the question files | — |

The same protocol is applied to the training, validation and test splits.
