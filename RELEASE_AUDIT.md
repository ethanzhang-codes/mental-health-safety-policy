# Public-release audit

Audit performed on October 30, 2026.

## Included

- Research Python source files
- Aggregate CSV, Markdown, JSON, and text results
- Publication tables and figures
- Presentation and reproduction documentation
- Aggregate logistic-regression versus random-forest reassessment

## Deliberately excluded

- Raw or derived conversation datasets
- Case-level false-negative reports
- Prediction exports and policy-output rows
- Model checkpoints and serialized policy models
- Embeddings and NumPy arrays
- TensorBoard event files and runtime logs
- Cached dependencies and Python bytecode
- Backup copies of superseded scripts

## Automated checks

- No copied CSV has a raw `text`, `dialogue_text`, `conversation_text`, `utterance`, or `message` column.
- No copied text file contains the original `/data/ezhang2` or `/home/ezhang2` workspace path.
- No common API-key, access-token, password-assignment, or private-key pattern was detected.
- The largest included file is below 1 MB.
- Raw dialogue text is excluded from the logistic-regression prediction artifacts; only aggregate summaries are included.

## Remaining owner decisions

- Choose and add a software license.
- Add an owner contact method for private security/privacy reports.
- Verify third-party dataset and pretrained-model terms before linking or distributing any additional artifacts.
- Review the README and presentation for claims and authorship before making the repository public.

Automated scanning reduces risk but cannot prove that every artifact is safe. The repository owner should still perform a final visual review before publication.
