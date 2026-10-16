# Can Job Details Predict Salary in the Age of AI?

**DATA 4380 · Data Problems** — Instructor: Dr. Masoud Rostami
**Team:** Quynh Huong Vo · Maryam Irfan · Ruth Slaughter · Subhi Paulraj

Predicting the salary tier of data and AI jobs (2020–2025) from job, location and company details, and testing whether a generative-AI exposure score from the ILO adds predictive signal.

---

## Overview

Generative AI is changing the tasks people do at work. This project asks two questions:

1. Can job, location and company details predict which **salary tier** a record falls into?
2. Does a job's **exposure to generative AI** (the ILO index by Gmyrek et al., 2025) improve that prediction?

Salary (`salary_in_usd`) is binned into four quartile tiers, **Low, Lower-Mid, Upper-Mid and High**, which turns the problem into a balanced 4-class classification task. Random guessing scores 25%.

## Key Results

| Model | Accuracy | Macro F1 | ROC-AUC | Within 1 tier |
|---|---|---|---|---|
| Dummy (most frequent) | 0.250 | 0.100 | 0.500 | 0.500 |
| Logistic Regression (baseline) | 0.417 | 0.403 | 0.685 | 0.792 |
| Random Forest | 0.411 | 0.399 | 0.678 | 0.786 |
| Gradient Boosting | 0.419 | 0.398 | 0.686 | 0.786 |
| Logistic + ILO exposure | 0.417 | 0.403 | 0.685 | 0.792 |
| **Logistic + title features (final)** | **0.447** | **0.429** | **0.713** | **0.811** |
| XGBoost + title features | 0.444 | 0.425 | 0.714 | 0.812 |
| Ridge log-salary → tier | 0.405 | 0.404 | n/a | 0.888 |

All models were scored on the same 14,383 held-out test rows. Precision, recall and F1 are macro-averaged.

**Findings**

- **The baseline works.** A leakage-safe logistic regression improves accuracy from 25% (chance) to 41.7%.
- **The original features hit an information ceiling.** The 57,530 training rows collapse into only 2,201 unique feature profiles. A model that memorized every profile could reach at most 42.8% accuracy, which explains why the linear, bagged and boosted models all tie.
- **New information beats a bigger model.** Adding the raw job title (keyword flags plus TF-IDF) lifts accuracy to 44.7% and ROC-AUC to 0.713. XGBoost ties logistic regression, so the simpler, interpretable model was kept.
- **The gain is robust.** It holds in 5-fold cross-validation, the 95% bootstrap confidence interval for the F1 gain is [+0.019, +0.032], and a model trained on 2020–2024 still improves accuracy on 2025 data (0.409 → 0.440).
- **The AI exposure score adds nothing here.** Each role family maps to exactly one ILO score, so the score is just `role_family` relabelled. This dataset therefore cannot test the link between AI exposure and pay. That is a finding about the data, not evidence that AI exposure is unrelated to pay.
- **The most important features** are job title, then experience level, then employee and company location, according to both SHAP and permutation importance.

## Data

- **Source:** [foorilla — ai-jobs-net-salaries](https://github.com/foorilla/ai-jobs-net-salaries) (CC0 1.0), 71,913 self-reported salary records, 17 columns, 2020–2025.
- **AI exposure:** scores averaged from Gmyrek et al. (2025), *Generative AI and Jobs*, ILO Working Paper 140, Table A1, attached through the `isco_group_hint` column. Five ISCO groups are matched, covering 44% of rows.
- **File expected by the notebook:** `ai_jobs_salaries_clean.csv`, placed in the same folder as the notebook.

| Column | Description |
|---|---|
| `work_year` | Year of the salary record |
| `experience_level` | EN / MI / SE / EX (Entry, Mid, Senior, Executive) |
| `employment_type` | FT / PT / CT / FL |
| `job_title` | Free-text job title (422 distinct values) |
| `salary`, `salary_currency` | Salary in local currency *(dropped: leakage)* |
| `salary_in_usd` | **Target**, binned into 4 quartile tiers |
| `employee_residence` | Employee's country (ISO-2) |
| `remote_ratio` | 0 = on-site, 50 = hybrid, 100 = remote |
| `company_location` | Company's country (ISO-2) |
| `company_size` | S / M / L |
| `experience_level_label`, `employment_type_label`, `work_mode` | Readable duplicates *(dropped)* |
| `salary_outlier_flag` | Derived from salary *(dropped: leakage)* |
| `role_family` | 11-category grouping of `job_title` |
| `isco_group_hint` | Link to ILO occupation group (56% missing) |

## Methodology

1. **Exploratory analysis.** Checked missing values, variable balance, the salary distribution, salary by role, work arrangement and year, and AI exposure against mean salary.
2. **Leakage audit.** Dropped columns that reveal or duplicate the target: `salary`, `salary_currency`, `salary_outlier_flag`, the `*_label` columns and `work_mode`. `job_title` and `isco_group_hint` were also dropped from the baseline as redundant with `role_family`.
3. **Preprocessing.**
   - Stratified 80/20 train/test split (`random_state=42`).
   - Quartile tier cut-offs computed from the **training data only**: $96,000, $138,750 and about $190K.
   - One-hot encoding of the categorical columns and standardization of the numeric ones, fit inside a scikit-learn `Pipeline` so nothing is learned from the test set.
4. **Baseline.** A dummy classifier and logistic regression.
5. **Challenger models.** Each tests one hypothesis:
   - Random Forest: do feature interactions matter?
   - Gradient Boosting: does a stronger nonlinear learner help?
   - Logistic regression + ILO score (median imputation plus a missing indicator): does AI exposure add signal?
6. **Diagnosis.** An information-ceiling analysis shows the 8 base features cap accuracy at 0.428.
7. **Feature engineering.** The job title is added back as keyword flags (senior, management and junior terms) plus TF-IDF on single words and word pairs (`min_df=5`).
8. **Alternative framing.** Ridge regression on log(salary), with the prediction converted to a tier. Its mean absolute error is about $46K.
9. **Robustness checks.** 5-fold cross-validation, a paired bootstrap (1,000 resamples) and a temporal test (train on 2020–2024, test on 2025).
10. **Interpretation.** Permutation importance, SHAP on the XGBoost model, logistic-regression odds ratios and an error analysis by salary band.

## Repository Structure

```
├── README.md
├── Project_Code.ipynb             # Full analysis: EDA, baseline, models, evaluation
├── Project_Presentation.pptx      # Final presentation slides
└── ai_jobs_salaries_clean.csv     # Dataset (download from source if not included)
```

> Rename the files above to match the names in your repo.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter
jupyter notebook Project_Code.ipynb
```

Then choose **Kernel → Restart & Run All**. Random seeds are fixed at 42, so the results are reproducible. Part 4 reloads the CSV, so it does not depend on variables set earlier in the notebook.

## Limitations

- Salaries are self-reported. About 84% of records are US-based, and 92% are from 2024–25.
- Four tiers hide wide pay ranges. The middle tiers ($96K–$190K) are the hardest to predict.
- Job titles are free text and vary between companies.
- The AI exposure score has only one value per role family, so it cannot vary within a role.
- Feature importance shows association, not causation.

## Future Work

- Match each job title to a 4-digit ISCO occupation code so AI exposure varies within roles.
- Ordinal models that respect tier order (for example cumulative-link models or CORAL).
- Cost-of-living adjustment for non-US salaries.
- Recompute the tier cut-offs each year as salaries drift upward.

## References

- foorilla. (2025). *ai-jobs-net-salaries* [Data set]. GitHub. CC0 1.0.
- Gmyrek, P., Berg, J., Kamiński, K., Konopczyński, F., Ładna, A., Nafradi, B., Rosłaniec, K., & Troszyński, M. (2025). *Generative AI and jobs: A refined global index of occupational exposure* (ILO Working Paper 140). International Labour Office. https://doi.org/10.54394/HETP0387
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*, 785–794.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.
