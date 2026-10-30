# Presentation Blueprint: Safety-Constrained Escalation for Multi-Turn Mental-Health Dialogue

## Purpose of this document

This is a copy-ready blueprint for recreating a presentation about the current project. It is tailored to the repository's audited evidence rather than copied from the structure of the earlier presentation.

The deck should tell one clear story:

> A mental-health chatbot that evaluates only the latest message can miss risk that develops across a conversation. This project combines early-warning and confirmation signals, causal conversation history, a learned four-level policy, and explicit guardrails. On the clinician-test split, a validation-selected higher-recall Confirm model reduced unsafe continuation, but substantial misses remained and the system is not ready for clinical deployment.

The intended audience is technically literate but does not need prior familiarity with the project. The recommended main deck is 24 slides plus references and optional backup slides. At roughly 45–75 seconds per substantive slide, it fits a 20–25 minute talk.

---

# 1. Presentation identity

## Recommended title

**From Early Alert to Confirmed Risk**  
**Safety-Constrained Escalation for Multi-Turn Mental-Health Dialogue**

Alternative titles:

- **Balancing Unsafe Continuation and Over-Escalation in Mental-Health Chatbot Safety**
- **A Multi-Turn Safety Layer for Mental-Health Chatbots**
- **Context-Sensitive Escalation for Mental-Health Dialogue**

Recommended subtitle:

> An offline study of current-turn evidence, conversation history, and guardrails

Add:

- Presenter name
- Department or program
- Institution
- Email
- Presentation date

## Visual style

Use a restrained research style rather than a consumer-chatbot aesthetic.

- Background: warm white or very light gray
- Primary color: deep navy
- Alert color: amber
- Confirm/high-risk color: muted red
- Safe/continue color: teal
- Neutral/context color: slate
- Use red only for safety-critical signals and errors.
- Use large diagrams and short claims; move technical qualifications into a bottom caption or speaker notes.
- Never use photographs of distressed people. Prefer abstract conversation, sequence, shield, and decision-path visuals.

Suggested level colors:

| Level | Meaning | Color |
| --- | --- | --- |
| 0 | Continue | Teal |
| 1 | Offer resources | Amber |
| 2 | Human deferral | Orange |
| 3 | Urgent deferral | Muted red |

## Language rules for the entire deck

Use:

- “offline safety framework”
- “dataset-defined risk evidence”
- “proxy escalation levels”
- “clinician-test split”
- “unsafe continuation”
- “human deferral”
- “research candidate”

Do not use:

- “clinical diagnosis”
- “prevents suicide”
- “clinically validated”
- “real-time emergency response”
- “deployment-ready”
- “proven safe”
- “autonomously contacts emergency services”

The output levels are labels in saved offline results. They do not trigger a real intervention.

---

# 2. Narrative architecture

The deck follows this progression:

1. Define the narrow problem.
2. Show why current-turn classification is insufficient.
3. State the research question and safety tradeoff.
4. Introduce the data and Alert–Confirm formulation.
5. Explain temporal evidence, categories, policy levels, and guardrails.
6. Show the frozen evaluation workflow.
7. Present component and policy results.
8. Analyze remaining errors.
9. define the evidence boundary.
10. End with a prioritized research path.

The audience should leave with four ideas:

1. The project is a safety layer, not a therapist or diagnostic model.
2. Alert, Confirm, category, and temporal progression answer different questions.
3. Raising Confirm recall improved one downstream safety tradeoff, but did not solve the problem.
4. The remaining failures point toward contextual Confirm modeling, calibration, stronger validation, and human oversight.

---

# 3. Slide-by-slide presentation

## Slide 1 — Title

### On-slide text

**From Early Alert to Confirmed Risk**

**Safety-Constrained Escalation for Multi-Turn Mental-Health Dialogue**

An offline study of current-turn evidence, conversation history, and guardrails

Presenter · Institution · Date

### Visual

A minimal left-to-right sequence of three conversation bubbles becoming progressively more specific, ending in a shield/decision icon. Avoid showing sample crisis text.

### Speaker note

> This project asks how a mental-health chatbot could recognize risk that develops across a conversation and choose a proportionate response level. It is an offline research framework—not a therapist, diagnosis system, or autonomous emergency service.

---

## Slide 2 — The problem in one sentence

### Headline

**The latest message may not contain the whole safety signal.**

### On-slide content

Use a simple conceptual sequence:

| Earlier turn | Middle turn | Current turn |
| --- | --- | --- |
| vague distress | increasing specificity | explicit or ambiguous evidence |

Under the sequence, place:

> A single-turn model asks, “What does this message say?”  
> A safety layer must also ask, “What has been building across the conversation?”

### Visual

Three neutral, paraphrased conversation cards with an upward trend line. Do not use raw dataset dialogue.

### Speaker note

Explain that the same current-turn score can mean something different after a stable sequence versus a steadily escalating sequence. This is the motivation for combining local evidence with causal history.

---

## Slide 3 — Scope and boundary

### Headline

**A narrow safety decision layer—not a clinical agent**

### On-slide content

**The layer does**

- detect early Alert and explicit Confirm evidence;
- track categories and change across prior user turns;
- select one of four offline response levels;
- produce a short, trigger-based rationale.

**The layer does not**

- diagnose a condition;
- replace clinicians or emergency services;
- prove real-world crisis prediction;
- initiate an actual intervention.

### Visual

Place the safety layer between “conversation” and “response policy.” Show therapist/diagnosis/emergency-service icons outside the boundary with a thin “not included” line.

### Speaker note

Set this boundary early. “Urgent escalation” is an output label in a CSV, not a call or message to an external service.

---

## Slide 4 — Research question

### Headline

**Can richer evidence reduce dangerous misses without escalating ordinary conversations too often?**

### On-slide content

Primary question:

> Can current-turn crisis evidence and causal conversation-history summaries support a graded offline escalation policy that reduces unsafe continuation without excessive escalation?

Secondary questions:

- Does conversation history improve early Alert detection?
- Does higher Confirm recall improve downstream safety?
- Do guardrails improve the safety–burden tradeoff?

### Visual

A balance scale:

- left: **Unsafe continuation** — high-risk turn receives Level 0
- right: **Over-escalation** — no-risk turn receives Level 2 or 3

### Speaker note

Accuracy alone hides the fact that these two errors have different consequences. The project therefore treats the outcome as a tradeoff, not a single score to maximize.

---

## Slide 5 — Contribution summary

### Headline

**A modular pipeline for evidence, progression, and constrained action**

### On-slide content

1. Audited Alert, Confirm, and seven-category prediction components
2. Causal multi-turn progression features with no future-turn inputs
3. Four-level learned policy with explicit minimum-level guardrails
4. Validation-only selection of a higher-recall Confirm candidate
5. Safety-centered evaluation of continuation and escalation burden

Bottom caption:

> The contribution is the assembled and evaluated framework; the underlying Transformers, random forests, weighted loss, and guardrail concepts are established techniques.

### Visual

Five numbered modules in a horizontal ribbon.

---

## Slide 6 — Data roles

### Headline

**One dataset built the pipeline; another supports the safety evaluation**

### On-slide table

| Dataset | Role | Dialogues | User turns | Ground-truth boundary |
| --- | --- | ---: | ---: | --- |
| AnnoMI | End-to-end prototyping | 133 | not mapped to CRADLE user-turn convention | No usable clinical escalation ground truth; pseudo-label metrics are sanity checks |
| CRADLE train | Model training | 3,058 | 25,662 | Synthetic/structured release |
| CRADLE validation | Threshold and candidate selection | 420 | 3,529 | Synthetic/structured release |
| CRADLE clinician test | Held-out reporting | 600 | 4,527 | Clinician-annotated benchmark |

### Visual

Two-lane data flow:

- AnnoMI → engineering prototype
- CRADLE → training → validation selection → clinician-test reporting

### Speaker note

Do not compare AnnoMI prototype metrics with CRADLE performance. The presentation’s main empirical claim comes from the 4,527-user-turn CRADLE clinician-test split.

### Source

`paper_package/tables/table1_dataset_statistics.md`

---

## Slide 7 — The Alert–Confirm formulation

### Headline

**Alert and Confirm represent different stages of evidence**

### On-slide content

**Alert**

- earlier warning signal;
- may be vague or indirect;
- benefits from conversation history.

**Confirm**

- more explicit crisis-linked evidence;
- drives stronger escalation;
- false negatives are especially safety-critical.

**Seven Confirm-linked categories**

- modeled as independent binary outputs;
- distinguish types of crisis evidence;
- category generalization remains weak for some rare labels.

### Visual

A two-stage funnel or timeline: **Alert → Confirm**, with seven category branches emerging from Confirm. Do not invent category names beyond those verified for the talk; “suicide/self-harm” and “interpersonal harm” are safe examples documented in the repository.

### Speaker note

These are dataset labels and probabilities, not diagnoses. Alert is not simply “low Confirm”; it represents a distinct earlier-stage annotation.

---

## Slide 8 — Why time matters

### Headline

**Risk is a trajectory, not only a point estimate**

### On-slide content

Example probability paths:

- stable: 0.10 → 0.12 → 0.11
- rising: 0.10 → 0.35 → 0.70
- isolated spike: 0.08 → 0.09 → 0.70

Derived causal features include:

- prior score and change from previous user turn;
- rolling mean, recent maximum, and highest-so-far;
- rising, accumulation, escalation, and volatility signals;
- whether Alert occurred before Confirm;
- category changes.

Bottom caption:

> Features use only the current and previous turns—never future dialogue.

### Visual

Plot the three illustrative paths on the same 0–1 line chart. Mark them as conceptual, not observed cases.

---

## Slide 9 — System architecture

### Headline

**Current-turn evidence and causal history converge at the policy**

### On-slide diagram

Use the existing architecture figure:

`paper_package/figures/fig13_final_model_architecture.svg`

Or rebuild this flow:

```text
Current user turn + prior user-turn history
        |
        +--> Temporal Alert (history window 8)
        +--> High-recall Confirm (current turn)
        +--> Seven category probabilities (current turn)
                         |
                         v
              Causal progression features
                         |
                         v
                 Random-forest policy
                         |
                         v
              light_safety guardrail
                         |
                         v
        Level 0 / Level 1 / Level 2 / Level 3
                + template rationale
```

### Speaker note

There is no custom cross-attention architecture. Temporal Alert uses frozen DeBERTa turn embeddings, while final Confirm remains current-turn-only. The policy consumes probabilities and summary features, not the raw dialogue history.

---

## Slide 10 — Component models

### Headline

**The final candidate is modular rather than end-to-end**

### On-slide table

| Component | Model | Context | Status |
| --- | --- | --- | --- |
| Temporal Alert | DeBERTa-v3-large embeddings + temporal encoder | Current and prior turns; window 8 | Selected candidate |
| Confirm | DeBERTa-v3-large binary classifier | Current turn | `B_pos_weight_x2` selected |
| Category | DeBERTa-v3-large, seven binary outputs | Current turn | Working component |
| Policy | Random forest | Predictions + causal progression features | Selected learned policy |
| Guardrail | `light_safety` rules | Minimum escalation level | Final defense layer |

### Speaker note

The system does not jointly train severity, category, progression, and policy. They remain separate modules combined at the feature and policy level.

### Source

`paper_package/models/model_architecture.md`

---

## Slide 11 — Improving Confirm recall

### Headline

**Missed Confirm evidence was treated as an upstream safety bottleneck**

### On-slide content

Selected candidate: **`B_pos_weight_x2`**

- doubles the standard negative-to-positive training weight;
- applied positive weight: **23.1588**;
- checkpoint and threshold selected on validation only;
- selected threshold: **0.011**.

Performance:

| Split | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.812 | 0.951 | 0.876 | 0.993 | 0.934 |
| Clinician test | 0.762 | 0.517 | 0.616 | 0.884 | 0.652 |

### Visual

Use `paper_package/figures/fig5_confirm_pr_curve.png` if its labels are legible. Otherwise use two recall bars, one for validation and one for clinician test, to emphasize the distribution gap.

### Speaker note

The low threshold is intentional because false-negative Confirm cases are costly. The large validation-to-test recall drop is also important evidence of domain shift and limits generalization claims.

### Source

`paper_package/tables/table5_alert_confirm_performance.md`

---

## Slide 12 — Temporal Alert result

### Headline

**History improved Alert ranking, but absolute performance remained limited**

### On-slide chart data

| Split | Single-turn Alert AUROC | Temporal Alert, window 8 AUROC | Change |
| --- | ---: | ---: | ---: |
| Validation | 0.608 | 0.859 | +0.251 |
| Clinician test | 0.537 | 0.736 | +0.199 |

Additional clinician-test values for temporal Alert:

- AUPRC: **0.0868**
- tuned recall: **0.256**
- tuned F1: **0.106**

### Visual

Grouped bars for single-turn versus temporal Alert on validation and clinician test. Add a small note: “AUROC improvement does not imply deployment-ready detection.”

### Speaker note

This supports the premise that earlier turns contain useful information. It does not mean Alert is solved; the low AUPRC and F1 show how difficult and imbalanced the task remains.

### Source

`paper_package/tables/table7_temporal_ablation.md`

---

## Slide 13 — From evidence to action

### Headline

**Four levels avoid a false choice between “continue” and “emergency”**

### On-slide table

| Level | Offline action label | Plain-language meaning |
| --- | --- | --- |
| 0 | Continue conversation | Signals are low or stable |
| 1 | Offer supportive resources | Concern is elevated; add support or a light check |
| 2 | Defer to human support | Stronger evidence warrants human involvement |
| 3 | Urgent deferral | Strong Confirm/category evidence warrants an urgent minimum level |

### Visual

A four-step vertical ladder using the level colors. Keep Level 3’s label as “urgent deferral” or “urgent escalation (offline label).”

### Speaker note

The supervised policy target is constructed from CRADLE Alert, Confirm, and category labels. It is a proxy target, not an independently validated clinical action label.

---

## Slide 14 — Learned policy plus guardrails

### Headline

**The learned action can be raised—but never lowered—by a guardrail**

### On-slide equation

\[
\text{final level} =
\max(\text{learned policy level},\ \text{guardrail minimum level})
\]

Guardrails react to patterns such as:

- high Confirm probability;
- elevated suicide/self-harm category evidence;
- accumulated Alert evidence;
- combinations of moderate evidence and escalation.

### Visual

Use `paper_package/figures/fig14_policy_decision_compact.svg` or show two inputs entering a `MAX` gate and one final level exiting.

### Speaker note

This structure is intentionally asymmetric. A safety rule may override a low learned prediction, but cannot reduce the learned level. Guardrails still depend on upstream predictors and cannot catch evidence that all models miss.

---

## Slide 15 — Evaluation design

### Headline

**Selection on validation; reporting on the clinician-test split**

### On-slide flow

1. Train component models on CRADLE train
2. Tune thresholds and select candidates on validation
3. Construct causal progression features
4. Train/select the policy and guardrail under validation constraints
5. Freeze the comparison
6. Report on 600 clinician-test dialogues / 4,527 user turns

### Visual

Use `paper_package/figures/fig15_experimental_workflow_compact.svg`.

### Bottom caution

> The broader project was exploratory and includes repeated test reporting; the full research history cannot be characterized as a pristine one-shot confirmatory test.

### Speaker note

This qualification matters. The isolated original-to-high-recall Confirm comparison uses the same downstream policy configuration, but the repository as a whole includes exploratory test reuse.

---

## Slide 16 — Safety-centered metrics

### Headline

**The main outcomes measure two different failure costs**

### On-slide definitions

**Unsafe continuation**

\[
\frac{\text{high-risk turns assigned Level 0}}
{\text{all eligible high-risk turns}}
\]

Lower is better.

**Over-escalation burden**

\[
\frac{\text{no-risk turns assigned Level 2 or 3}}
{\text{all eligible no-risk turns}}
\]

Lower is better.

Also report:

- Confirm precision, recall, F1, AUROC, and AUPRC;
- Alert ranking and temporal ablations;
- per-category performance;
- timing, calibration, and uncertainty quality when available.

### Speaker note

Always state these denominators. Older prototype reports sometimes use different “insufficient response” groupings, so results are only comparable when the metric definition is fixed.

---

## Slide 17 — Primary result

### Headline

**Higher-recall Confirm reduced unsafe continuation by 6.81 percentage points**

### On-slide chart

Use `paper_package/figures/fig4_main_safety_results.png`.

If rebuilding:

| Metric | Original Confirm | High-recall Confirm | Change |
| --- | ---: | ---: | ---: |
| Unsafe continuation among 470 high-risk turns | 140 / 470 = **29.79%** | 108 / 470 = **22.98%** | **−6.81 pp** |
| Over-escalation among 3,824 no-risk turns | 455 / 3,824 = **11.90%** | 468 / 3,824 = **12.24%** | **+0.34 pp** |

Add 95% dialogue-cluster bootstrap intervals:

- original unsafe continuation: **25.70%–33.90%**
- new unsafe continuation: **19.24%–26.71%**
- original over-escalation: **11.18%–12.63%**
- new over-escalation: **11.53%–12.98%**

### Takeaway box

> 32 high-risk turns moved out of Level 0, while over-escalation increased modestly.

### Speaker note

This is the central result. It is a single-seed, turn-level, held-out dataset result—not a clinical outcome. Phrase it as an improved offline safety–burden tradeoff for one frozen policy configuration.

### Source

`paper_package/tables/table8_escalation_policy_results.md`

---

## Slide 18 — Interpreting the tradeoff

### Headline

**The intervention helped, but did not solve unsafe continuation**

### On-slide content

**What improved**

- unsafe continuation fell from 29.79% to 22.98%;
- 32 high-risk cases improved and none worsened on that paired outcome;
- relative reduction: 22.86%.

**What it cost**

- 13 additional no-risk turns were escalated overall;
- over-escalation rose by 0.34 percentage points;
- 108 high-risk turns still received Level 0.

### Visual

Use a 100-turn icon array twice:

- before: about 30 red Level-0 misses
- after: about 23 red Level-0 misses

Beside it, show a very small increase in orange no-risk escalations from about 12 to about 12.2 per 100.

### Speaker note

Avoid leading with the post hoc paired p-values. The practical effect sizes, counts, and remaining errors are more informative and less likely to imply confirmatory statistical proof.

---

## Slide 19 — Why the remaining misses occurred

### Headline

**Many remaining errors were indirect, contextual, or gradual**

### On-slide chart

Use `paper_package/figures/fig10_error_categories.png`.

Data for 108 remaining unsafe-continuation cases:

| Overlapping error category | Count | Percent |
| --- | ---: | ---: |
| Indirect or ambiguous | 78 | 72.2% |
| Important evidence earlier | 63 | 58.3% |
| Gradual progression | 53 | 49.1% |
| Unusual wording | 50 | 46.3% |
| Calibration or confidence | 38 | 35.2% |
| Label ambiguity | 15 | 13.9% |

Bottom caption:

> Categories overlap and are post hoc research annotations, not independent clinical adjudications.

### Speaker note

The error pattern supports a specific next step: improve contextual Confirm modeling. A current-turn Confirm classifier cannot use earlier explicit evidence if the present turn is indirect.

---

## Slide 20 — Category and generalization limits

### Headline

**Category predictions were useful inputs, but not consistently reliable**

### On-slide content

Clinician-test seven-category summary:

- macro AUPRC: **0.332**
- best reported category AUPRC: **0.447**
- worst reported category AUPRC: **0.182**

Key implication:

> The policy should not treat weak category scores as definitive evidence, especially for rare categories.

Additional generalization evidence:

- Confirm recall fell from 0.951 on validation to 0.517 on clinician test.
- Temporal Alert AUROC fell from 0.859 to 0.736.
- Train/validation and clinician-test distributions differ.

### Visual

A “validation → clinician test” arrow with visibly shrinking bars. Keep the message focused on distribution shift, not leaderboard comparison.

### Source

`paper_package/tables/table11_per_risk_category.md`

---

## Slide 21 — Calibration and uncertainty: implemented versus final

### Headline

**Uncertainty is part of the research goal, but not yet part of the selected CRADLE policy**

### On-slide content

**Implemented in the earlier severity prototype**

- five-seed ensemble averaging;
- temperature scaling;
- ensemble disagreement and predictive entropy.

**Not established in the final CRADLE candidate**

- no final CRADLE temperature calibration;
- probability-derived uncertainty ablation was rejected;
- no held-out selective-risk or out-of-distribution validation;
- final policy must not be called ensemble-uncertainty-aware.

### Visual

A two-column “available prototype” versus “selected final candidate” comparison. Use a dashed arrow labeled “future integration.”

### Speaker note

Computing entropy is not the same as proving uncertainty quality. A useful uncertainty signal should identify errors, unfamiliar inputs, and distribution shifts on held-out evaluation.

---

## Slide 22 — What the evidence supports

### Headline

**Supported result, bounded claim**

### On-slide content

**Supported**

> In one single-seed CRADLE clinician-test evaluation, replacing the original Confirm predictor with a validation-selected higher-recall candidate reduced Level-0 continuation among dataset-defined high-risk turns from 29.79% to 22.98%, while Level-2/3 escalation among no-risk turns increased from 11.90% to 12.24%.

**Not supported**

- clinical effectiveness;
- diagnosis or real-world crisis prediction;
- robust fairness or external generalization;
- autonomous deployment;
- elimination of unsafe continuation;
- a final uncertainty-aware system.

### Visual

A green “evidence supports” box and a gray/red “evidence does not support” box.

### Speaker note

This slide gives the exact language that should also appear in an abstract, conclusion, poster, or defense response.

---

## Slide 23 — Limitations and ethical risks

### Headline

**The central risks are technical, human, and institutional**

### On-slide matrix

| Technical | Human | Governance |
| --- | --- | --- |
| Single-seed final pipeline | False negatives may leave risk unsupported | Dataset licensing and provenance unresolved |
| Current-turn-only Confirm | False positives may cause distress or loss of trust | IRB/consent/data-use basis undocumented |
| Weak rare-category generalization | Fluent rationales may create automation bias | No subgroup/fairness evaluation |
| Final probabilities uncalibrated | Proxy actions are not clinical decisions | No prospective or external validation |
| Exploratory test reuse | Guardrails can encode brittle thresholds | No production monitoring/localization |

### Speaker note

Template rationales explain active triggers; they are not causal explanations of a person’s mental state. Sensitive raw text should be minimized and access controlled.

---

## Slide 24 — Prioritized next steps

### Headline

**The next milestone is a stronger evaluation pipeline—not deployment**

### On-slide roadmap

**1. Fix the upstream bottleneck**

- context-concatenated or temporal Confirm baseline;
- category-specific temporal modeling;
- structured review of the 108 remaining misses.

**2. Strengthen statistical evidence**

- multi-seed CRADLE training;
- confidence intervals across training runs;
- external and out-of-distribution evaluation;
- subgroup robustness where data permit.

**3. Validate confidence and deferral**

- fit calibration on validation only;
- reliability and class-specific calibration;
- selective-risk/risk–coverage testing;
- held-out uncertainty-quality evaluation.

**4. Add human and governance evidence**

- clinician and human-factors review;
- rationale and escalation-workflow evaluation;
- license, provenance, IRB, privacy, and security documentation.

### Visual

A four-stage staircase ending in “prospective evaluation,” not “deployment.”

---

## Slide 25 — Conclusion

### Headline

**Context helped. Recall mattered. Safety remains unresolved.**

### On-slide content

1. Temporal history improved early Alert ranking.
2. A validation-selected higher-recall Confirm model reduced unsafe continuation in the frozen policy comparison.
3. The gain came with a small increase in over-escalation.
4. Nearly one quarter of eligible high-risk turns still received Level 0.
5. The system is an offline research candidate, not a clinically validated chatbot.

Final line:

> The value of the framework is not that it declares the system safe; it makes the safety tradeoff measurable and exposes where the next failures must be addressed.

### Visual

Return to the opening trajectory graphic, now overlaid with the four-level decision ladder and a visible “human oversight” endpoint.

---

## Slide 26 — Questions

### On-slide text

**Questions**

Small footer:

> Offline research framework · Dataset-defined targets · No autonomous intervention

### Visual

Use a simplified version of the system architecture as a background watermark so discussion can refer back to it.

---

# 4. References slide

The audited repository does not contain complete authoritative bibliographic entries. Do not invent them. Build the final references slide only after verifying:

- the authoritative CRADLE-Dialogue paper/release;
- the authoritative AnnoMI paper/release;
- DeBERTa-v3;
- temperature scaling;
- random forests;
- relevant crisis-language detection and mental-health chatbot safety literature;
- institutional or regulatory guidance used in the final talk.

Until verified, use a placeholder slide:

**References requiring authoritative completion**

- CRADLE-Dialogue dataset and benchmark
- AnnoMI dataset
- DeBERTa-v3 encoder
- Calibration and uncertainty methods
- Mental-health chatbot safety and crisis-language detection

Do not include repository filenames as formal academic citations. They are evidence anchors for building the deck, not publication references.

---

# 5. Optional backup slides

## Backup A — Full dataset statistics

Copy the complete table from:

`paper_package/tables/table1_dataset_statistics.md`

Include total turns, user turns, listener turns, and label source.

## Backup B — Encoder benchmark

Use only if asked how the base representation was selected.

Key severity-prototype averages across five seeds:

| Encoder | Mean test macro F1 | Mean high-risk recall | Mean calibrated ECE |
| --- | ---: | ---: | ---: |
| DeBERTa-v3-large | 0.9670 | 0.9994 | 0.0214 |
| ModernBERT-large | 0.9624 | 0.9915 | 0.0242 |
| Mental-RoBERTa-large | 0.9579 | 0.9977 | 0.0322 |

Prominent caption:

> These are severity-benchmark results, not CRADLE Alert/Confirm results.

Source: `paper_package/tables/table4_encoder_comparison.md`

## Backup C — Guardrail search

Explain:

- `light_safety` remained the selected guardrail;
- later v3 rule searches were validation-only;
- no v3 candidate improved validation unsafe continuation under the over-escalation constraint of 0.20;
- test outputs therefore remained unchanged.

Source: `paper_package/tables/table9_guardrail_comparison.md`

## Backup D — Main-result exact counts and paired audit

| Outcome | Improved cases | Worsened cases | Post hoc exact paired p |
| --- | ---: | ---: | ---: |
| Unsafe continuation | 32 | 0 | 4.66 × 10⁻¹⁰ |
| Over-escalation | 8 | 21 | 0.0241 |

Caption:

> Post hoc descriptive paired analysis; do not present as a preregistered confirmatory test.

Source: `paper_package/tables/table10_main_safety_results.md`

## Backup E — Formal policy specification

Show:

\[
\text{policy level}=\max(\text{random-forest prediction},\text{guardrail minimum})
\]

Then list the four levels and representative triggers. Source:

`paper_package/methods/final_policy_specification.md`

## Backup F — Complete pipeline status

**Implemented and exercised**

- CRADLE preprocessing, targets, fixed splits, and leakage audits;
- Alert and Confirm models;
- seven-label category model;
- temporal Alert;
- causal progression features;
- rule-based, learned, and safety-constrained policies;
- validation-selected high-recall Confirm;
- false-negative audits and safety reports.

**Still under development**

- unified severity/category/progression model;
- calibrated final CRADLE probabilities;
- validated uncertainty-aware final policy;
- multi-seed and external validation;
- prospective clinical and human-factors evaluation;
- production governance and monitoring.

---

# 6. Chart-building specifications

## Chart 1 — Main safety result

Use two small panels rather than one shared axis.

Panel A: unsafe continuation

- Original Confirm: 29.79%
- High-recall Confirm: 22.98%
- Direction annotation: −6.81 percentage points
- Color: muted red → amber

Panel B: over-escalation

- Original Confirm: 11.90%
- High-recall Confirm: 12.24%
- Direction annotation: +0.34 percentage points
- Color: slate → orange

Add counts directly on bars. If error bars are shown, use the dialogue-cluster bootstrap intervals from Slide 17.

## Chart 2 — Temporal Alert ablation

Grouped bars:

| Split | Single turn | History window 8 |
| --- | ---: | ---: |
| Validation | 0.607675 | 0.859042 |
| Clinician test | 0.536585 | 0.735710 |

Y-axis: AUROC, 0.5–1.0. Do not truncate without making the 0.5 baseline obvious.

## Chart 3 — Remaining error categories

Horizontal bars in descending order. Add a subtitle:

> Overlapping categories among 108 remaining unsafe-continuation cases

Do not sum percentages; they exceed 100% because categories overlap.

## Chart 4 — Validation-to-test gap

Two dumbbell plots:

- Confirm recall: 0.951 → 0.517
- Temporal Alert AUROC: 0.859 → 0.736

Label the right-hand values “clinician test.” The intended conclusion is domain shift/generalization concern.

---

# 7. Recommended existing visual assets

Use the repository assets before redrawing:

| Purpose | Preferred asset |
| --- | --- |
| Full system | `paper_package/figures/fig13_final_model_architecture.svg` |
| Compact policy decision | `paper_package/figures/fig14_policy_decision_compact.svg` |
| Experimental workflow | `paper_package/figures/fig15_experimental_workflow_compact.svg` |
| Main result | `paper_package/figures/fig4_main_safety_results.png` |
| Confirm precision–recall | `paper_package/figures/fig5_confirm_pr_curve.png` |
| Safety tradeoff | `paper_package/figures/fig7_safety_tradeoff.png` |
| Temporal example | `paper_package/figures/fig9_temporal_case_study.png` |
| Error analysis | `paper_package/figures/fig10_error_categories.png` |
| Ablations | `paper_package/figures/fig11_ablation_results.png` |
| Policy logic | `paper_package/figures/fig12_policy_decision_diagram.png` |

SVG is preferred for architecture diagrams because it remains sharp when resized. PNG is appropriate for generated plots. Check every figure’s title, axis labels, and legend against the current terminology before placing it in the final deck.

---

# 8. Presenter’s short scripts

## 30-second version

> This project builds an offline safety layer for a mental-health chatbot. It detects early Alert and more explicit Confirm evidence, tracks how those signals change across the conversation, and uses a constrained four-level policy to continue, offer resources, or defer to human support. On the CRADLE clinician-test split, a validation-selected higher-recall Confirm model reduced unsafe continuation from 29.79% to 22.98%, while over-escalation rose from 11.90% to 12.24%. The result shows a measurable safety tradeoff, not clinical readiness.

## 90-second version

> A chatbot that looks only at the latest message can miss a crisis that develops gradually. This project therefore separates early Alert evidence from more explicit Confirm evidence and combines them with category probabilities and causal summaries of previous user turns. A random-forest policy selects one of four offline levels, and explicit guardrails can raise the minimum response when warning patterns are strong.
>
> The main comparison replaced the original Confirm predictor with a validation-selected higher-recall candidate while keeping the downstream policy configuration fixed. On 4,527 user turns from the CRADLE clinician-test split, unsafe continuation among 470 high-risk turns fell from 140 cases, or 29.79%, to 108 cases, or 22.98%. Over-escalation among 3,824 no-risk turns rose slightly from 11.90% to 12.24%.
>
> The remaining 108 misses were often indirect, dependent on earlier evidence, or gradual. That makes contextual Confirm modeling, calibration, multi-seed evaluation, and human review the next priorities. The framework is an offline research candidate, not a clinically validated or autonomous safety system.

---

# 9. Likely audience questions and defensible answers

## “Is the model diagnosing a crisis?”

No. It predicts dataset-defined Alert, Confirm, and category labels and assigns proxy offline escalation levels. These are not diagnoses.

## “Does Level 3 call emergency services?”

No. Level 3 is an offline output label. The repository does not contact a person, clinician, or emergency service.

## “Why is the Confirm threshold only 0.011?”

It was selected on validation data for a high-recall objective in an imbalanced task. The absolute probability is not calibrated confidence, and the final CRADLE system still needs validation-fitted calibration.

## “Why not escalate every concerning case?”

Excessive escalation can create distress, loss of trust, unnecessary intervention, and unequal burden. The research question explicitly balances unsafe continuation against over-escalation.

## “Why is temporal context used for Alert but not Confirm?”

Temporal Alert improved ranking and was selected. Temporal Confirm experiments performed worse, so the selected Confirm model remained current-turn-only. Error analysis now suggests that stronger contextual Confirm methods should be revisited.

## “Are the results statistically robust?”

The main result has dialogue-cluster bootstrap intervals and a post hoc paired audit, but the final CRADLE pipeline is single-seed and the broader research history is exploratory. Multi-seed and external evaluation are needed for robust claims.

## “Is the system uncertainty-aware?”

The earlier severity prototype includes ensemble and calibration methods. Probability-derived uncertainty was tested in the CRADLE policy but not selected. The final candidate should not be described as a validated uncertainty-aware system.

## “What is the most important remaining failure?”

The 108 high-risk turns that still received Level 0. Many involved indirect language, earlier important evidence, or gradual progression, pointing to an upstream context bottleneck.

---

# 10. Final build checklist

Before presenting:

- Replace all presenter and institution placeholders.
- Verify authoritative dataset and method citations.
- Confirm the exact seven category names before listing all of them.
- Keep raw mental-health dialogue out of the slides unless data-use terms and ethics approval permit it.
- Ensure the words “offline,” “research,” and “not clinically validated” appear in the scope, results, and conclusion sections.
- Put numerators and denominators beside the two primary rates.
- Keep validation and clinician-test results visually distinct.
- Do not mix severity-prototype results with CRADLE Alert/Confirm results.
- Do not describe the final CRADLE policy as calibrated, ensemble-based, or uncertainty-aware.
- Do not imply that Level 3 initiates real-world intervention.
- Explain that error-analysis categories overlap.
- End with prospective evaluation and governance, not deployment.

---

# 11. Repository evidence map

Use these files to verify or update the deck:

| Topic | Source |
| --- | --- |
| Plain-language project explanation | `bert-pretrained/docs/project_walkthrough.md` |
| Master claims and results | `paper_package/paper_facts.md` |
| Verification boundary | `paper_package/audit_summary.md` |
| Dataset counts and roles | `paper_package/datasets/dataset_summary.md` |
| Component architecture | `paper_package/models/model_architecture.md` |
| Final policy and guardrails | `paper_package/methods/final_policy_specification.md` |
| Alert/Confirm results | `paper_package/tables/table5_alert_confirm_performance.md` |
| Temporal ablation | `paper_package/tables/table7_temporal_ablation.md` |
| Primary safety results and CIs | `paper_package/tables/table8_escalation_policy_results.md` |
| Paired main result | `paper_package/tables/table10_main_safety_results.md` |
| Category summary | `paper_package/tables/table11_per_risk_category.md` |
| Error categories | `paper_package/tables/table12_error_categories.md` |
| Limitations | `paper_package/ethics/limitations.md` |
| Ethical boundary | `paper_package/ethics/ethics_and_safety.md` |
| Missing documentation and experiments | `paper_package/unresolved_questions.md` |

