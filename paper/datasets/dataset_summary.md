# Dataset Summary

## Verified roles

AnnoMI was used to prototype an end-to-end conversational pipeline. Its repository evaluation explicitly states that clinical risk labels are absent and that prediction-against-pseudo-label metrics are sanity checks. CRADLE-Dialogue is the main crisis-emergence evaluation source and supplies exact Alert/Confirm annotations and Confirm-linked categories. The two result families are not directly comparable.

CRADLE train/validation are described by the preprocessing audit as synthetic/structured conversational data; test is the clinician-annotated benchmark. Dialogue IDs were inferred at `turn_id == 0` resets because the loaded release lacked dialogue IDs. This inference and the exact release provenance require author/dataset confirmation.

| dataset | split | split_role | dialogues | total_turns | user_turns | listener_turns | unknown_turns | label_source | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRADLE-Dialogue | train | train | 3058 | 48557 | 25662 | 22830 | 65.0 | synthetic/structured release | cradle_dialogue/data/cradle_train_targets.csv |
| CRADLE-Dialogue | validation | validation | 420 | 6679 | 3529 | 3142 | 8.0 | synthetic/structured release | cradle_dialogue/data/cradle_validation_targets.csv |
| CRADLE-Dialogue | test | clinician test | 600 | 8975 | 4527 | 4448 | 0.0 | clinician-annotated benchmark | cradle_dialogue/data/cradle_test_targets.csv |
| AnnoMI | prototype input | prototyping only | 133 | 9699 | 0 | 0 | NA | no clinical escalation ground truth used; severity outputs are constructed/pseudo-label pipeline | data/annomi_turns.csv; results/framework_evaluation_annomi.txt |

## Labels and splitting

`target_alert` is one only for an exact `alert_*` current-turn label. `target_confirm` is one for any exact `confirm_*` current-turn label. Category targets require exact `confirm_<category>_ongoing` or `confirm_<category>_past` labels. Models are evaluated on user turns using masks. CRADLE ships separate train/validation/test files; no dialogue ID overlaps occur after ID inference, but the inferred IDs are split-local and cannot by themselves prove source-level identity separation.

## Processing, licensing, and privacy

The repository records speaker normalization, empty-text replacement, exact label parsing, and dialogue-ID inference in `prepare_cradle_dialogue.py`; model tokenization uses the selected Hugging Face tokenizer, truncation/padding, and maximum lengths specified in training scripts. No dataset license, redistribution permission, download date, or authoritative dataset citation is stored in the audited repository. Raw dialogue text should therefore not be redistributed until terms are verified. Sensitive mental-health content requires access control, minimization, and non-identifying reporting.
