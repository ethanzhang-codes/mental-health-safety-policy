# Table 1. Dataset statistics

Counts are reconstructed from repository CSVs. CRADLE test is clinician annotated; AnnoMI is prototype-only.

| dataset | split | split_role | dialogues | total_turns | user_turns | listener_turns | unknown_turns | label_source | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRADLE-Dialogue | train | train | 3058 | 48557 | 25662 | 22830 | 65.0 | synthetic/structured release | cradle_dialogue/data/cradle_train_targets.csv |
| CRADLE-Dialogue | validation | validation | 420 | 6679 | 3529 | 3142 | 8.0 | synthetic/structured release | cradle_dialogue/data/cradle_validation_targets.csv |
| CRADLE-Dialogue | test | clinician test | 600 | 8975 | 4527 | 4448 | 0.0 | clinician-annotated benchmark | cradle_dialogue/data/cradle_test_targets.csv |
| AnnoMI | prototype input | prototyping only | 133 | 9699 | 0 | 0 | NA | no clinical escalation ground truth used; severity outputs are constructed/pseudo-label pipeline | data/annomi_turns.csv; results/framework_evaluation_annomi.txt |
