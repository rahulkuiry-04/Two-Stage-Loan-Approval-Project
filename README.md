# Loan Approval Project — Two-Stage ML Application

A two-stage model on the [Loan Approval Prediction dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset):

1. **Stage 1 — Classification:** predict whether an applicant is `Approved` / `Rejected`.
2. **Stage 2 — Regression:** for applicants predicted `Approved`, estimate the loan amount.



---

## Project Structure

```
.
|
├── app
|   ├── __init__.py
|   ├── loafder.py
|   ├── predict.py
|   └── utils.py
├── models/
│   ├── stage_1_rf_classifier_pipeline.pkl
│   └── stage_2_rf_regression_pipeline.pkl
├── streamlit_app.py              
├── main.py                       
├── config.yaml                   
└── requirements.txt
```

## Quickstart (local)

1. Create venv:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```
2. Place the dataset as `loan_approval_dataset.csv` in the project root.
3. Train both stages and print evaluation metrics:
   ```bash
   uv run python refactored_pipeline.py
   ```
   This fits Stage 1 and Stage 2, prints the confusion matrix, regression metrics, and the regression error-bucket table to stdout. Add a `joblib.dump(...)` call at the end (or extend `main()`) to persist `stage_1_...pkl` / `stage_2_...pkl` into `models/` for the app to consume.
4. Run locally:
   - UI: `uv run streamlit run streamlit_app.py`
   - CLI: `uv run python main.py`

## Config

See `config.yaml` for runtime parameters (model paths).

## To install/freeze additional libraries using UV

```bash
uv pip install -r requirements.txt
uv pip freeze > requirements.txt
```

## Git instructions

```bash
git init
git add .
git commit -m "message"
git remote add origin https://url_of_your_git_repo.git
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## Notes

- The model version used to train must match your local environment when running `main.py` / the Streamlit app.
- We use Python 3.12 for the virtual environment.
- `RANDOM_STATE = 10` is fixed throughout `refactored_pipeline.py` for reproducibility.

---


## Evaluation (on the leakage-fixed test partitions)

**Stage 1 — Classification** (n=854)

| | Pred Rejected | Pred Approved |
|---|---|---|
| Actual Rejected | TN = 317 | FP = 6 |
| Actual Approved | FN = 8 | TP = 523 |

Accuracy 0.9836 · Precision 0.9887 · Recall 0.9849 · F1 0.9868 · ROC-AUC 0.9971

**Stage 2 — Regression** (n=531 approved applicants)

RMSE ≈ 3,339,722 · MAE ≈ 2,518,988 · R² = 0.8700

| Error bucket | % of test |
|---|---|
| Underestimate >20% | 11.3% |
| Underestimate 10–20% | 23.2% |
| Underestimate 5–10% | 7.3% |
| Within ±5% | 14.5% |
| Overestimate 5–10% | 6.6% |
| Overestimate 10–20% | 10.6% |
| Overestimate >20% | 26.6% |

Re-run `python refactored_pipeline.py` to reproduce these numbers exactly (fixed `RANDOM_STATE = 10`).

## 🔗 Live Demo

[Try it here](https://rahulkuiry-04-two-stage-loan-approval-proj-streamlit-app-rpd7ls.streamlit.app/)

