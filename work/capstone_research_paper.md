# Machine Learning for Content Refresh Opportunity Scoring: A Leakage-Aware, Client-Held-Out Approach to Prioritizing Search Content for Review

**Lane:** Refresh / Content Opportunity Scoring
**Program:** FlyRank AI Fluency / Machine Learning Internship — Capstone

---

## 1. Abstract

Content teams often manage far more pages than they can manually review, making it necessary to decide which pages deserve attention first. This study asks whether historical visibility, freshness, content, and engagement signals can be used to rank pages by their likelihood of being in an observed state of search-performance decline, so that a review queue can be prioritized. The dataset is a 30,000-row anonymized internship release in which each row is one content item with search, traffic, engagement, position, content, and freshness fields, plus an observed trend outcome. The target, `is_declining_label`, is defined as `trend_direction == "down"`, giving a declining-label rate of 54.2% across the modeling population. A transparent rule-based baseline (staleness combined with visibility and freshness percentiles) is compared against three learned classifiers — Logistic Regression, Decision Tree, and Random Forest — trained on a leakage-screened feature set. All models are evaluated on an identical 80/20 **client-held-out** split (`random_state=42`), so that no client appears in both the training and test partitions. Logistic Regression produced the best held-out ranking performance on the primary decision metric, reaching a Precision@50 of 0.82 versus 0.26 for the baseline, with a ROC-AUC of 0.731. These are observed associations within this dataset and validation design, not evidence that the identified signals cause decline or that acting on them will improve search rankings. The practical implication is a repeatable, human-reviewed workflow for prioritizing which pages an editorial team should examine first, rather than an automated content-change or algorithm-prediction system.

---

## 2. Introduction

A content team responsible for thousands of published pages cannot manually re-review all of them on a regular basis. Some pages are losing visibility, engagement, or traffic; others are stable or improving. The practical problem is not predicting a single yes/no outcome for each page — it is deciding, given limited editorial capacity, **which pages should be reviewed first**.

A ranked review queue is more useful for this problem than a binary classifier alone. A binary "declining / not declining" prediction gives no guidance about where to start when hundreds or thousands of pages are flagged; a ranked queue lets an analyst work down a prioritized list until their available review capacity is exhausted, and lets the organization tune queue size (top 10, top 20, top 50) to the resources actually available.

This leads to the research question addressed in this paper:

> **Can historical visibility, freshness, content, and engagement signals be used to identify content that is more likely to be declining, so that teams can prioritize which pages should be reviewed or refreshed first?**

The system described here is intended as **decision support**, not decision automation. Its output is a ranked list with reason codes that a content analyst or SEO specialist reviews before taking any action. It does not publish, edit, or approve changes to content on its own, and it makes no claim about how Google's ranking systems work.

---

## 3. Problem Formulation

- **Unit of analysis:** one content item (page), represented as a single row after deduplication by `content_id`.
- **Prediction target:** `is_declining_label`, a binary indicator derived directly from the dataset's `trend_direction` field:

  ```python
  is_declining_label = (trend_direction == "down")
  ```

- **Positive class:** pages observed with `trend_direction == "down"` (54.2% of the modeling population).
- **Intended decision:** given limited review capacity, which pages should a content team examine first?
- **Ranking objective:** maximize the concentration of true declining pages near the top of the ranked queue, evaluated primarily via Precision@K for small queue sizes (K = 10, 20, 50).

`is_declining_label` is a proxy for an **observed outcome already present in the dataset**, not a universal or externally validated definition of "content quality" or "needs refresh." A page can be labeled declining for reasons unrelated to content quality (e.g., seasonal demand, a competitor entering a space, a SERP feature change), and a page not labeled declining may still be a reasonable candidate for improvement. The label describes what happened in the observed window, not what should be done about it.

---

## 4. Data

The dataset is the repository's anonymized starter release, `content_refresh_anonymized.csv`, containing **30,000 rows and 44 columns**, one row per content item.

After applying the modeling population filters defined in the notebook — `impressions_90d > 0`, `content_age_days >= 90`, and deduplication on `content_id` — the modeling population remained **30,000 rows**; no rows were removed by these filters in this dataset release, and no duplicate `content_id` values were present. The population spans **32 distinct clients**.

**Target distribution** (`trend_direction`, n = 30,000):

| Trend | Rows |
|---|---:|
| down | 16,262 |
| stable | 5,962 |
| up | 4,388 |
| new | 2,236 |
| flat | 1,152 |

This gives an overall declining-label rate (`trend_direction == "down"`) of **54.207%**.

**Available signal categories** (fields actually present in the CSV and used as model inputs, described by group rather than exhaustively):

- **Search/keyword context:** `search_volume`, `competition`, `competition_level`, `cpc`, `main_intent`
- **Content characteristics:** `content_type`, `word_count`, `char_count`, and their tiered variants
- **Visibility / traffic (90-day window):** `impressions_90d`, `clicks_90d`, `pageviews_90d`, `sessions_90d`, `users_90d`
- **Engagement:** `engaged_sessions_90d`, `ai_sessions_90d`, `scroll_events_90d`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`
- **Consistency of visibility:** `days_with_impressions`, `days_with_sessions`
- **Position / CTR:** `avg_position`, `ctr`, `position_tier`
- **Freshness / age:** `content_age_days`, `age_tier`, `age_tier_order`, `days_since_last_update`, `freshness_tier`

Only fields actually present in the CSV and referenced in the notebook's feature lists are used; no additional fields were invented for this paper.

---

## 5. Data Safety and Public-Safe Handling

The dataset supplied for this capstone is already anonymized: it does not contain client names, domains, URLs, private search queries, credentials, or private exports. `client_id` and `content_id` are opaque anonymized identifiers used only to (a) construct the client-level train/test split and (b) deduplicate rows; neither is used as a model input, and neither is reproduced in this paper. All figures reported below are aggregate statistics or model outputs computed over the anonymized dataset. No sensitive identifiers appear anywhere in this document.

---

## 6. Feature Engineering

**Numeric features (23):** `search_volume`, `competition`, `cpc`, `word_count`, `char_count`, `impressions_90d`, `clicks_90d`, `pageviews_90d`, `sessions_90d`, `users_90d`, `engaged_sessions_90d`, `ai_sessions_90d`, `scroll_events_90d`, `days_with_impressions`, `days_with_sessions`, `content_age_days`, `age_tier_order`, `days_since_last_update`, `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`.

**Categorical features (9):** `competition_level`, `content_type`, `main_intent`, `age_tier`, `freshness_tier`, `word_count_tier`, `char_count_tier`, `impression_tier`, `position_tier`.

This gives **32 total model inputs**. Numeric features are median-imputed with missingness indicators, then standardized; categorical features are imputed with the most frequent training-set category and one-hot encoded, with unseen categories at inference time ignored safely by the encoder. All of this preprocessing is fit only on the training partition, inside a single `sklearn` `Pipeline`, so no information from the held-out test partition leaks into imputation statistics, scaling parameters, or encoder categories.

**Explicitly excluded fields (12), all defined in the notebook's `LEAKAGE_COLUMNS` list:**

| Field(s) | Reason for exclusion |
|---|---|
| `content_id`, `client_id` | Identifiers; `client_id` is used only to construct the grouped train/test split, never as a feature. |
| `trend_direction`, `trend_pct` | These directly define the label — including them would make the target trivially recoverable from the inputs. |
| `impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`, `impressions_prev_30d`, `clicks_prev_30d`, `sessions_prev_30d` | These 30-day windows are the fields used to compute the observed trend; including them would leak the outcome the model is trying to predict. |
| `provider_used`, `model_used` | Not decision-relevant to a content-review prioritization model. |

The notebook asserts programmatically that none of these 12 fields intersect with the 32-field feature set, so this exclusion is enforced in code, not only in documentation.

---

## 7. Leakage Prevention

Leakage prevention in this project operates at three levels:

1. **Target-derived fields.** `trend_direction` and `trend_pct` are the fields the label is computed from and are excluded from the feature matrix entirely.
2. **Outcome-window fields.** The `*_last_30d` and `*_prev_30d` fields are the specific windows used to derive the observed trend; they are excluded for the same reason as the target fields themselves, since a model given these fields would effectively be shown the answer.
3. **Identifiers.** `content_id` and `client_id` are excluded as features. `client_id` is retained in a separate array used exclusively to construct the grouped train/test split, never passed to any estimator.

**Preprocessing is fit inside the pipeline.** Imputers, the scaler, and the one-hot encoder are all fit only on the training fold as part of a single `sklearn.pipeline.Pipeline`, so no training-partition statistic is influenced by test-partition data.

**Validation separation.** All models are compared on an identical held-out test partition constructed by client, not by row (see Section 10).

These measures reduce, but do not eliminate, all possible leakage. They specifically target label-derived leakage and identifier leakage; they do not, for example, guarantee the complete absence of subtler within-client behavioral correlation between training and test rows for clients that remain unseen but share domain-level characteristics with training clients. This is discussed further in Section 16.

---

## 8. Baseline

A transparent, non-learned baseline ("Week-4 Baseline") is computed only on the held-out test population, using the following rule, reproduced exactly from the notebook:

```python
visibility_percentile      = rank_pct(log1p(impressions_90d))
freshness_risk_percentile  = rank_pct(days_since_last_update)
stale_visible               = (days_since_last_update >= 180) & (impressions_90d >= 500)

baseline_score = 2.0 * stale_visible + 0.5 * visibility_percentile + 0.5 * freshness_risk_percentile
```

The logic is: a page earns the largest score boost if it is simultaneously stale (no update in ≥180 days) and still visible (≥500 impressions in the last 90 days) — the combination most likely to represent a neglected but still-relevant page — with smaller additive contributions from how visible the page is overall and how long it has gone without an update.

A baseline is necessary because it establishes a decision-relevant, easy-to-explain floor. A learned model is only worth the added complexity and reduced transparency if it meaningfully outperforms this simple, auditable rule on the same evaluation population.

---

## 9. Modeling

Three learned classifiers were evaluated, all trained inside the same leakage-screened preprocessing pipeline:

- **Logistic Regression** (`class_weight="balanced"`, `max_iter=1000`) — a linear, directly interpretable classifier via standardized coefficients. Included as the simplest learned baseline against which the added complexity of tree-based methods can be justified.
- **Decision Tree** (`max_depth=5`, `min_samples_leaf=50`, `class_weight="balanced"`) — tests whether a small number of nonlinear splits capture meaningful structure the linear model cannot, while remaining shallow enough to inspect.
- **Random Forest** (`n_estimators=200`, `max_depth=10`, `min_samples_leaf=25`, `class_weight="balanced_subsample"`) — tests whether an ensemble of trees, which can capture feature interactions, improves ranking quality over a single tree or a linear model.

No other model families were evaluated in this notebook; none are claimed here.

---

## 10. Validation Design

All models and the baseline are compared on an identical **80/20 client-held-out split**, constructed deterministically with `random_state=42`. Clients (not individual rows) are randomly assigned to the test partition until approximately 20% of clients are held out; every row belonging to a held-out client goes to the test set, and no client appears in both partitions.

- **Training partition:** 27,675 rows across 26 clients (55.476% declining-label rate).
- **Test partition:** 2,325 rows across 6 clients (39.097% declining-label rate).

The notebook explicitly checks that both partitions contain both classes before proceeding.

Client-level separation matters because content items from the same client are likely to share editorial style, publishing cadence, template structure, and domain-level SEO characteristics. A random row-level split could place near-duplicate or highly similar pages from the same client in both the training and test sets, inflating apparent performance by rewarding the model for recognizing a specific client rather than for learning generalizable signal-decline relationships. Holding out entire clients is a stronger test of whether learned patterns transfer to organizations the model has not seen at all.

Two limitations of this design should be stated plainly. First, **client-held-out validation is not the same as future time-based validation** — it tests generalization across clients within the same time window, not generalization to future time periods, which is a different and separately important question this dataset cannot answer. Second, because clients were assigned to partitions without stratifying on the label, the training and test partitions ended up with materially different declining-label base rates (55.5% vs. 39.1%). This is a real consequence of client-level (rather than row-level) splitting and should be kept in mind when interpreting the absolute precision/recall values, though it does not invalidate the relative comparison between models and baseline, since all of them are scored on the exact same test rows. The model passing this validation split is evidence of client-level generalization within this dataset; it is not, by itself, evidence that the system is production-ready.

---

## 11. Evaluation Metrics

The notebook reports:

- **ROC-AUC** — overall ability to rank declining pages above non-declining pages across all thresholds.
- **Average Precision / PR-AUC** — precision-recall trade-off, more informative than ROC-AUC when the operational interest is concentrated at the top of a ranking rather than uniform across all thresholds.
- **Precision@10 / Precision@20 / Precision@50** — the fraction of true declining pages among the top K ranked pages, directly measuring the quality of a small review queue.
- **Precision, Recall, F1** — standard classification metrics at a 0.5 probability threshold, included for completeness.

Because the intended use is a small, capacity-constrained review queue rather than an exhaustive classification of every page, **Precision@50 is treated as the primary decision metric** in this paper, consistent with the notebook's stated model-selection rule. Precision@50 measures how concentrated the top 50 model recommendations are with true declining pages — the metric most directly tied to "if an analyst only has time to review 50 pages this week, how many of them will actually be worth reviewing?" Accuracy is not used as the primary metric, since with a roughly balanced-to-majority positive class and a ranking-oriented use case, accuracy would not reflect queue quality at realistic review sizes.

---

## 12. Results

All models below are evaluated on the identical 2,325-row, 6-client held-out test partition described in Section 10.

| Model | ROC-AUC | PR-AUC (Avg. Precision) | Precision@10 | Precision@20 | Precision@50 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.731 | 0.626 | 0.80 | 0.80 | **0.82** | 0.647 | 0.570 | 0.606 |
| Random Forest | 0.746 | 0.607 | 0.80 | 0.70 | 0.72 | 0.566 | 0.741 | 0.642 |
| Decision Tree | 0.742 | 0.575 | 0.60 | 0.65 | 0.66 | 0.569 | 0.716 | 0.634 |
| Week-4 Baseline | — | — | 0.70 | 0.35 | 0.26 | — | — | — |

Selected by the notebook's rule (highest Precision@50, then PR-AUC, then ROC-AUC): **Logistic Regression**.

All three learned models outperform the transparent baseline at every queue size, with the gap widening sharply as the queue grows: at Precision@50 the baseline captures only 0.26 (26% of the top-50-ranked pages are truly declining), while Logistic Regression reaches 0.82. This is the central operational result — for a review queue of the size a team is actually likely to work through in a review cycle, the learned model concentrates far more true declining pages near the top than the simple staleness/visibility rule does.

It is worth noting that Random Forest has the highest ROC-AUC (0.746) and highest Recall (0.741) and F1 (0.642) of the three learned models, but Logistic Regression has the highest Precision@50 (0.82 vs. 0.72), which is the metric most relevant to this project's stated decision (a small, precision-sensitive review queue rather than a threshold classifier optimized for balanced error rates). This is a legitimate case where the "best" model depends on which metric matches the actual decision being supported — a team more concerned with **not missing** declining pages (higher recall) might reasonably prefer Random Forest, while a team with strictly limited review capacity, where each queue slot should count, is better served by Logistic Regression's tighter top-K precision.

For the confusion matrix of the selected model (Logistic Regression, 0.5 threshold, n=2,325): 1,133 true negatives, 283 false positives, 391 false negatives, 518 true positives.

**Charts** (generated from these results):
- `capstone_precision_at_k.png` — Precision@10/20/50 across all four approaches.
- `capstone_action_mix.png` — distribution of recommended actions across the full ranked queue.
- `capstone_top_feature_importance.png` — top 10 model signals for the selected model.

---

## 13. Model Interpretation

Because Logistic Regression was the selected model, "importance" here refers to the absolute magnitude of the model's coefficients on standardized features — how strongly each feature moves the model's log-odds of the declining label, holding the linear combination of other features fixed — rather than a tree-based split-importance measure.

**Top 10 model signals (Logistic Regression, standardized |coefficient|):**

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | word_count | 1.028 |
| 2 | position_tier = top_3 | 0.912 |
| 3 | char_count | 0.869 |
| 4 | users_90d | 0.758 |
| 5 | sessions_90d | 0.684 |
| 6 | days_with_impressions | 0.590 |
| 7 | days_with_sessions | 0.462 |
| 8 | word_count_tier = &lt;1000 | 0.455 |
| 9 | word_count_tier = 3500+ | 0.408 |
| 10 | main_intent = navigational | 0.408 |

Content length (`word_count`, `char_count`, and their tier indicators) and consistency of visibility/traffic (`users_90d`, `sessions_90d`, `days_with_impressions`, `days_with_sessions`) are the strongest signals in the fitted model, alongside already holding a top-3 search position. Content age and freshness fields (`content_age_days`, `days_since_last_update`, `freshness_tier`) appear further down the fitted model's coefficient ranking, not among the top 10.

These are **predictive associations within the fitted model on this dataset**, not causal claims. Content length was among the stronger predictive signals in the fitted model — this does not mean that short or very long content causes decline, or that changing word count would change a page's trajectory. Feature importance describes what the model used to discriminate between the two classes in this dataset under this validation design; it is not evidence about which factors Google's ranking systems weight, and it is not evidence that manipulating any single feature would produce a specific outcome.

---

## 14. Ranked Recommendations

The final ranking is produced by refitting the selected model (Logistic Regression) on the full 30,000-row modeling population and scoring every page with `decline_probability`. This produces a full ranked review queue rather than a single automated action.

The notebook defines four possible action categories, assigned by `decline_probability` thresholds combined with the staleness/visibility flags used in the baseline:

| Action | Trigger condition | Rows in full queue (n=30,000) |
|---|---|---:|
| `refresh_and_review` | `decline_probability ≥ 0.75` **and** stale (≥180 days since update) **and** still visible (≥500 impressions/90d) | 0 |
| `priority_review` | `decline_probability ≥ 0.75` | 1,783 |
| `monitor_closely` | `decline_probability ≥ 0.50` | 14,518 |
| `monitor` | (default) | 13,699 |

An honest observation from this run: the `refresh_and_review` category — reserved for pages that are simultaneously high-probability, stale, and still visible — did not trigger for any page in this dataset. Reason-code counts show that the joint "stale + visible" condition is itself rare across the whole population (only 14–17 pages out of 30,000 meet it, from the reason-code breakdown below), so very few pages combine high predicted decline probability with clear staleness and continued visibility at the same time. This is a real property of this dataset population, not a fabricated or adjusted result, and it is worth surfacing to a content team: most flagged pages in this release are not simply "old and forgotten," so a staleness-driven refresh playbook alone would miss most of what the model is actually flagging.

Reason codes attached to each page (not mutually exclusive) give an analyst a starting hypothesis for **why** a page was ranked where it was:

| Reason code | Rows |
|---|---:|
| model-ranked review (no specific rule matched) | 19,293 |
| older content | 6,400 |
| visible CTR review | 2,777 |
| visible CTR review; older content | 1,356 |
| stale content | 149 |
| stale + visible | 14 |
| stale content; older content | 8 |
| stale + visible; visible CTR review | 3 |

**How an analyst should use this:** `priority_review` pages (the top 1,783 by probability) are the recommended starting point for human review, not automatic edits. `monitor_closely` pages are a larger secondary tier worth periodic checking. A model score and reason code are a prioritization signal — they tell a reviewer where to look and offer an initial hypothesis, not an instruction to publish a specific change.

---

## 15. Results-to-Decision Workflow

```
Data
  ↓
Feature preparation (numeric/categorical pipelines, leakage-column exclusion)
  ↓
Leakage checks (programmatic assertion that excluded fields never enter FEATURES)
  ↓
Baseline (rule-based, computed on held-out test population)
  ↓
Model (Logistic Regression, Decision Tree, Random Forest — same pipeline, same split)
  ↓
Validation (80/20 client-held-out split, random_state=42)
  ↓
Ranked scores (best model refit on full population, decline_probability per page)
  ↓
Reason codes (rule-based explanations attached to each ranked page)
  ↓
Human review (analyst inspects top-ranked pages and reason codes)
  ↓
Content action (decided and executed by a person, not the model)
```

The final action always remains subject to human review; nothing in this pipeline publishes or edits content automatically.

---

## 16. Limitations

1. **Proxy target.** `is_declining_label` is an observed trend-direction outcome, not a validated measure of content quality or refresh need. A page can decline for reasons unrelated to the content itself, and a stable page is not necessarily a page in no need of improvement.
2. **Association, not causation.** All reported relationships — including the feature-importance ranking in Section 13 — are associations learned from observational data. Nothing in this analysis establishes that changing a feature (e.g., word count, update recency) would cause a page's trend to change.
3. **Client-held-out vs. time-based validation.** The validation design tests generalization across clients within the same time window. It does not test generalization to future time periods, and the model has not been validated against genuinely future data.
4. **Train/test base-rate shift.** Because the split is client-level rather than stratified by label, the training partition (55.5% declining) and test partition (39.1% declining) have different declining-label prevalence. This is a real consequence of the chosen validation design and should be kept in mind when interpreting absolute (rather than relative, baseline-vs-model) metric values.
5. **Dataset scope.** The analysis uses a single anonymized starter release covering 32 clients and 30,000 rows. Findings should not be assumed to generalize automatically to other datasets, industries, or content ecosystems.
6. **Small number of held-out clients.** Only 6 clients make up the entire test partition. Held-out performance for this specific 6-client sample can be sensitive to the particular clients drawn into the test set under this random seed.
7. **Model stability.** Only a single train/test split (one random seed) was evaluated; no repeated splits, cross-validation across client folds, or confidence intervals around the reported metrics are available from this notebook.
8. **Threshold and queue-size dependence.** Precision/Recall/F1 are reported at a single 0.5 probability threshold; Precision@K results are reported only for K = 10, 20, 50. Performance at other queue sizes or thresholds is not directly reported here.
9. **Feature importance limitations.** Logistic Regression coefficient magnitudes reflect linear, additive relationships on standardized features; they do not capture interaction effects the way a tree-based importance measure might, and — as stated above — are not causal.
10. **Potential distribution shift.** Search behavior, engagement patterns, and content norms change over time; a model trained on this window's data may degrade in relevance if page behavior patterns change substantially after this dataset was collected.

---

## 17. Honest Interpretation

**What this study does demonstrate:**
- The tested visibility, freshness, content, and engagement signals contain predictive information about the observed declining label in this dataset.
- Logistic Regression achieved the strongest top-K ranking performance (Precision@50 = 0.82) among the three tested learned approaches, and all three learned approaches substantially outperformed the transparent rule-based baseline (Precision@50 = 0.26) under the specified client-held-out validation design.
- The resulting probability scores, combined with rule-based reason codes, can support prioritization of a human review queue.

**What this study does not demonstrate:**
- It does not reverse-engineer or approximate Google's ranking algorithm.
- It does not prove that refreshing, expanding, or otherwise editing any specific page will cause its search performance to improve.
- It does not guarantee improved rankings, traffic, or engagement after any content change made using this queue.
- It does not prove generalization to other websites, industries, or future time periods beyond the client-held-out sample evaluated here.

---

## 18. Conclusion

Content teams need a way to decide which of many pages to review first, and a ranked, explainable queue is more actionable for that decision than an isolated binary prediction. Using a 30,000-row anonymized dataset of content, search, and engagement signals, this study compared a transparent rule-based baseline against three learned classifiers on an identical client-held-out test partition. Logistic Regression produced the strongest top-K ranking performance (Precision@50 = 0.82 vs. 0.26 for the baseline; ROC-AUC = 0.731), with content length and traffic-consistency signals contributing most to the fitted model's discrimination. The practical value of this work is a repeatable, leakage-aware, human-reviewed workflow for prioritizing which content pages deserve editorial attention first — not a claim about how Google ranks pages, and not a guarantee that any specific content action will improve outcomes. The main limitation to keep in view is that every result here is an observed association under one validation design on one dataset release; it supports prioritization, not automation.

---

## 19. Reproducibility

This paper is based on:
- `work/notebooks/capstone.ipynb` — the capstone notebook, executed top to bottom without modification.
- `data/raw/content_refresh_anonymized.csv` — the 30,000-row, 44-column anonymized dataset described in Section 4.

The notebook's own markdown references earlier "weekly" notebooks as background context for this capstone; those notebooks were not included in the materials used to produce this paper, so no claims are made here about their specific content — only the capstone notebook itself was executed and reported on.

**Environment:** the notebook imports `pathlib`, `numpy`, `pandas`, `matplotlib`, and `scikit-learn` (`sklearn.compose`, `sklearn.impute`, `sklearn.linear_model`, `sklearn.ensemble`, `sklearn.tree`, `sklearn.pipeline`, `sklearn.preprocessing`, `sklearn.metrics`). No specific package versions are pinned in the notebook itself; a `pip freeze` of the working environment is recommended if exact-version reproducibility is required.

**To reproduce:**
1. Place `content_refresh_anonymized.csv` at `data/raw/content_refresh_anonymized.csv` relative to the repository root.
2. Run `work/notebooks/capstone.ipynb` top to bottom (`Runtime → Run all` or equivalent). The notebook auto-detects the repository root by checking for the data file at `data/raw/content_refresh_anonymized.csv` relative to the current working directory.
3. All stochastic components (the client split and the Random Forest) use `RANDOM_STATE = 42`, so results are deterministic given the same dataset.

**Outputs written to `work/outputs/`:**
- `capstone_model_comparison.csv`
- `capstone_top_features.csv`
- `capstone_ranked_recommendations_top50.csv`
- `capstone_action_mix.csv`
- `capstone_reason_mix.csv`
- `capstone_precision_at_k.png`
- `capstone_action_mix.png`
- `capstone_top_feature_importance.png`

---

## 20. Acknowledgments and Data Credit

Built on the FlyRank ML Internship dataset.

[https://flyrank.ai](https://flyrank.ai)

No private or client-identifying information is included in this paper.
