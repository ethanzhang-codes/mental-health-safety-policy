# Reconstructed Run Commands

Run from `bert-pretrained/`. These commands are reconstructed from argparse definitions and are not a substitute for the missing pinned environment.

```bash
python prepare_cradle_dialogue.py
python build_cradle_model_targets.py
python train_cradle_single_task.py --task alert --seed 1 --batch_size 32 --epochs 5
python train_cradle_single_task.py --task confirm --seed 1 --batch_size 32 --epochs 5
python precompute_cradle_turn_embeddings.py
python train_cradle_temporal_alert_from_embeddings.py --seed 1 --history_window 8
python train_cradle_single_task_confirm_high_recall.py --only_variant B_pos_weight_x2 --seed 1 --batch_size 32 --epochs 5
python train_cradle_single_turn_category.py --seed 1 --batch_size 32 --epochs 5
python build_cradle_progression_features_high_recall_confirm.py
python run_policies_with_high_recall_confirm.py
python search_safety_guardrails_v3_high_recall_confirm.py
python align_pipeline_with_abstract.py
python run_uncertainty_ablation_from_abstract_features.py
```

The original severity prototype uses `ensemble_uncertainty.py` and `temperature_scaling.py`; their defaults point to the five seed checkpoint directories. Do not rerun training merely to reproduce this package: frozen CSVs support all derived audit statistics.
