# Table 5. Alert and Confirm classification performance

Validation and clinician-test rows are explicitly separated. CRADLE components use seed 1.

| model | split | auroc | auprc | tuned_recall | tuned_f1 | source | task | precision | recall | f1 | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single-turn Alert | validation | 0.607675 | 0.090167 | NA | NA | single_task_vs temporal report | Alert | NA | NA | NA | NA |
| Temporal Alert hw8 | validation | 0.859042 | 0.238532 | NA | 0.105634 | single_task_vs temporal report | Alert | NA | NA | NA | NA |
| Single-turn Alert | clinician test | 0.536585 | 0.055381 | NA | NA | single_task_vs temporal report | Alert | NA | NA | NA | NA |
| Temporal Alert hw8 | clinician test | 0.73571 | 0.086796 | 0.25641 | 0.105634 | single_task_vs temporal report | Alert | NA | NA | NA | NA |
| Original Confirm | validation | NA | NA | NA | NA | cradle_dialogue/results/single_task_confirm/seed_1.txt | Confirm | NA | NA | NA | NA |
| Original Confirm | clinician test | NA | NA | NA | NA | cradle_dialogue/results/single_task_confirm/seed_1.txt | Confirm | NA | NA | NA | NA |
| B_pos_weight_x2 | validation | 0.9927719520688993 | 0.934170332882162 | NA | NA | cradle_dialogue/results/single_task_confirm_high_recall/B_pos_weight_x2/seed_1.txt | Confirm | 0.81194 | 0.951049 | 0.876006 | 0.011 |
| B_pos_weight_x2 | clinician test | 0.8839562825481568 | 0.6518664327800678 | NA | NA | cradle_dialogue/results/single_task_confirm_high_recall/B_pos_weight_x2/seed_1.txt | Confirm | 0.761755 | 0.517021 | 0.61597 | 0.011 |
