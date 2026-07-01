# Naive Bayes Classification — NBA Player Longevity Prediction

## Environment Setup
```
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Dataset
**extracted_nba_players_data.csv** — 1,340 NBA player records with pre-engineered features.

| Feature | Description |
|---------|-------------|
| `fg` | Field goal percentage |
| `3p` | Three-point percentage |
| `ft` | Free throw percentage |
| `reb` | Rebounds per game |
| `ast` | Assists per game |
| `stl` | Steals per game |
| `blk` | Blocks per game |
| `tov` | Turnovers per game |
| `total_points` | Aggregate scoring metric |
| `efficiency` | Composite efficiency metric |
| `target_5yrs` | **Target**: 1 = lasted 5+ years, 0 = did not |

No missing values. Class split: 62% lasted (831) / 38% did not (509).

## Modeling Approach
1. **Preprocessing**: Checked feature distributions; applied `log1p` transform to highly right-skewed features (|skew| > 1) to better satisfy the Gaussian NB normality assumption.
2. **Split**: 80/20 stratified train/test split (1,072 train / 268 test).
3. **Model**: `GaussianNB` from scikit-learn — models each feature as a Gaussian distribution per class, then applies Bayes' theorem for classification.
4. **Evaluation**: Confusion matrix, precision, recall, F1, ROC-AUC.
5. **Independence assumption analysis**: Critically assessed whether the Naive Bayes independence assumption holds for basketball stats.

## Final Performance (268-player test set)

| Metric | Score |
|--------|-------|
| Accuracy | 67.9% |
| Precision | 77.4% |
| Recall | 68.1% |
| F1 Score | 72.4% |
| ROC-AUC | 0.728 |

**Confusion Matrix:**
| | Predicted: Did Not Last | Predicted: Lasted 5+ |
|---|---|---|
| **Actually: Did Not Last** | 69 (TN) | 33 (FP) |
| **Actually: Lasted 5+yr** | 53 (FN) | 113 (TP) |

## Independence Assumption Analysis
The Naive Bayes assumption that features are conditionally independent **is violated** in basketball data:
- `total_points` and `efficiency` are mathematically linked (both derived from scoring)
- `reb` and `blk` correlate by player role (big men)
- `ast` and `efficiency` share components

Despite this, Gaussian NB delivers useful ranking performance (ROC-AUC = 0.73), confirming that even when independence is violated, the model can correctly rank players — it just produces miscalibrated probabilities. For scouting use, it should be treated as a **ranking/screening tool**, not a literal probability estimator.

## Top Predictive Features for Longevity
Based on Cohen's d (standardized mean difference between classes):
1. **`efficiency`** — strongest overall discriminator
2. **`total_points`** — cumulative scoring
3. **`ast`** — playmaking (assists per game)
4. **`reb`** — rebounding

## Scouting Recommendations
- Use the model as a **first-pass filter** for large player pools; shortlist predicted probability > 0.70
- First-round picks scoring < 0.40 warrant a scouting audit
- Do not use predicted probabilities as literal chances — use rankings only
- Retrain every 2–3 seasons as the league evolves

## Files
- `naive_bayes_nba.ipynb` — full executed notebook with all visualizations
- `extracted_nba_players_data.csv` — source dataset
- `README.md` — this file
- 
