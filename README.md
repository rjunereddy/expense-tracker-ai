# Expense Tracker with AI Analytics

An expense tracking dashboard where the "AI" is load-bearing rather than
decorative: transactions are auto-categorized from raw, noisy merchant text,
unusual spending is flagged at both the transaction and weekly-pattern level,
next-week spend is forecast per category with a confidence interval, and
plain-English questions about spending are answered by a grounded
query-parsing layer instead of a hallucination-prone LLM call.

## Why this project is structured the way it is

- **Categorization is the hardest, most "real" ML problem here.** Bank
  statement strings look like `AMZN MKTP US*2K3`, not `Amazon Shopping`.
  The categorizer uses character n-gram TF-IDF + a calibrated linear SVM,
  which handles truncation, embedded codes, and inconsistent spacing far
  better than word-level tokenization would.
- **Anomaly detection works at two levels on purpose.** A per-transaction
  Isolation Forest catches one-off unusual charges; a separate weekly
  z-score check catches pattern-level drift ("3x your usual dining spend
  this week") that a single-transaction model would never see.
- **The NLP query layer never lets a model invent a number.** A question
  like "how much did I spend on food last month" is parsed into a
  structured intent (category + date range + aggregation), and the actual
  number always comes from a direct pandas computation over the real data.
  This is the safer architecture for any product that surfaces financial
  figures — plug in Claude/an LLM for the *parsing* step later without ever
  letting it touch the arithmetic.
- **Forecasting is intentionally a simple weighted linear regression**, not
  Prophet/ARIMA — the interesting engineering here is the full pipeline
  (categorize → detect → forecast → explain), not forecasting sophistication.
  Swapping in a heavier time-series model is a clean "future work" item.

## Project structure

```
expense_tracker_ai/
├── generate_data.py   # synthetic transaction generator (realistic noisy merchant text)
├── categorize.py      # TF-IDF + Linear SVM auto-categorization model
├── anomaly.py         # Isolation Forest + weekly z-score anomaly detection
├── forecast.py        # per-category next-week spend forecasting
├── nlp_query.py        # rule-based natural-language query parser
├── app.py             # Streamlit dashboard tying everything together
├── requirements.txt
└── data/
    ├── transactions.csv     # generated on first run
    └── categorizer.joblib   # trained model, generated on first run
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first run it generates 180 days of synthetic transaction data and trains
the categorizer automatically. You can also upload your own CSV
(`date, description, amount` columns) from the sidebar — it will be
auto-categorized using the trained model.

## Using your own real data

The synthetic generator exists so the project runs with zero setup, but the
categorizer is designed to generalize: export a real bank/card statement as
CSV with `date`, `description`, `amount` columns, retrain
(`python categorize.py` after replacing `data/transactions.csv`, once you
have ground-truth category labels for a training set), and upload through
the sidebar.

## Possible extensions

- Swap the rule-based NLP query parser for an LLM-based intent classifier
  (keep the "LLM never computes the number" guarantee)
- Swap linear regression forecasting for Prophet/ARIMA and compare
  accuracy on real multi-year data
- Add a budget-setting feature with proactive alerts before a limit is hit
- Deploy on Streamlit Community Cloud or Render for a live demo link
