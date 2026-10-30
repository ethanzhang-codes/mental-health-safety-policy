# Master Paper Facts

## 1. Candidate titles

1. **Safety-Constrained Escalation for Multi-Turn Mental-Health Dialogue with Alert–Confirm Progression Signals**
2. **Balancing Unsafe Continuation and Over-Escalation in an Offline Mental-Health Chatbot Safety Framework**
3. **From Early Alert to Confirmed Risk: Context-Sensitive Escalation Policies for Mental-Health Dialogue**

## 2–6. Abstract, introduction, gap, questions, and contributions

The project asks whether current-turn crisis evidence and causal conversation-history summaries can support a graded offline escalation policy that reduces unsafe continuation without excessive escalation. The unit for Alert, Confirm, category, and policy evaluation is primarily the user turn; temporal features group user turns within dialogue. Secondary questions test whether history improves Alert, whether higher Confirm recall improves downstream safety, and whether guardrails/uncertainty improve the safety-burden tradeoff.

Supported contributions are: an audited modular Alert–Confirm–category/progression pipeline; causal CRADLE progression features; comparison of rule-based, learned, and safety-constrained four-level policies; validation-only high-recall Confirm selection; and safety-centered evaluation. Pretrained Transformers, weighted BCE, temperature scaling, ensembles, random forests, rolling features, and template rationales are adaptations of established techniques, not novel algorithms.

Too-strong claims include clinical validation, diagnosis, real-time deployment, real-world crisis prediction/prevention, elimination of unsafe continuation, a final uncertainty-aware policy, or a custom cross-attention architecture.

## 7. Related-work categories

Mental-health chatbot safety; crisis-language detection; longitudinal/contextual NLP; imbalanced classification; calibration and uncertainty; selective/human-defer policies; interpretable safety decisions; AnnoMI and CRADLE. Bibliographic details are unresolved.

## 8–14. Data and methodology

AnnoMI supports preliminary pipeline prototyping and has no usable clinical escalation ground truth in this repository. CRADLE-Dialogue supports exact current-turn Alert/Confirm/category targets, causal progression, and clinician-test evaluation. Train/validation are described as synthetic/structured and test as clinician annotated. The final selected components are temporal Alert (history window 8), current-turn high-recall Confirm B, current-turn seven-category probabilities, causal progression features, a random-forest policy, and a `light_safety` guardrail. Four levels are implemented: continue, resources, human deferral, and urgent escalation.

The earlier severity prototype implements five-seed averaging and temperature scaling. Final CRADLE calibration is absent. Probability-derived CRADLE uncertainty was evaluated but not selected. No cross-attention exists.

## 15–17. Baselines, metrics, and main results

Baselines include single-turn/temporal Alert, original/high-recall/temporal Confirm, rule-based/unconstrained/safety-constrained policies, v3 guardrails, and uncertainty features. Primary safety metrics are Level-0 continuation among high-risk turns and Level-2/3 escalation among no-risk turns.

On the 4,527-user-turn CRADLE clinician-test split, replacing original Confirm with validation-selected `B_pos_weight_x2` in the same safety-constrained policy reduced high-risk unsafe continuation from **140/470 = 0.2978723404255319 (29.79%)** to **108/470 = 0.2297872340425532 (22.98%)**, a **−6.81 percentage-point** change (−22.86% relative). Over-escalation changed from **455/3,824 = 0.11898535564853556 (11.90%)** to **468/3,824 = 0.12238493723849372 (12.24%)**, a **+0.34 percentage-point** change (+2.86% relative). These are single-seed, turn-level test results, not clinical outcomes.

## 18–19. Ablations and errors

Temporal Alert history window 8 improved validation and clinician-test ranking metrics versus single-turn Alert. Temporal Confirm was worse and rejected. v3 and probability-uncertainty policy ablations did not improve validation unsafe continuation and were rejected. The final 108 unsafe cases generally had extremely low Confirm probabilities; 63/108 were annotated as likely history-dependent. Error categories overlap and are not independent clinical adjudications.

## 20–25. Discussion, limitations, ethics, reproducibility, conclusion, future work

The evidence supports an offline safety-burden tradeoff and an upstream-recall bottleneck. It does not support deployment. Major limitations are single-seed CRADLE results, domain shift, proxy policy targets, weak category generalization, incomplete calibration, repeated exploratory test reporting, absent subgroup evaluation, and unresolved dataset/IRB/licensing provenance. The repository contains frozen outputs sufficient to verify core numbers, but lacks a pinned environment and complete acquisition documentation.

The defensible conclusion is that validation-selected higher-recall Confirm predictions improved the downstream test safety tradeoff of one frozen policy configuration, while substantial unsafe continuation remained. Future work should test contextual Confirm, validation-fitted calibration, multi-seed final pipelines, richer uncertainty with selective-risk evaluation, external/clinical review, subgroup robustness, and prospective human-factors studies.

Every numeric source is available in `results/main_results_table.csv`, `results/policy_results.csv`, and `source_inventory.csv`.
