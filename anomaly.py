"""
Anomaly detection over categorized transactions.

Two complementary signals:
1. Per-category Isolation Forest on individual transaction amounts -> flags
   single transactions that are unusual for their category (e.g. a 6000
   rupee "Food & Dining" charge when most are 100-1200).
2. Weekly per-category spend vs that category's rolling historical average
   -> flags pattern-level anomalies ("3x your usual dining spend this week").
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def flag_transaction_outliers(df: pd.DataFrame, contamination=0.03, random_state=42):
    """Per-category isolation forest on amount. Returns df with an is_outlier column."""
    df = df.copy()
    df["is_outlier"] = False
    df["outlier_score"] = 0.0

    for cat, group in df.groupby("category"):
        if len(group) < 10:
            continue  # not enough data to model this category meaningfully
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        amounts = group[["amount"]].values
        preds = iso.fit_predict(amounts)
        scores = iso.score_samples(amounts)
        df.loc[group.index, "is_outlier"] = preds == -1
        df.loc[group.index, "outlier_score"] = scores

    return df


def weekly_spend_alerts(df: pd.DataFrame, z_thresh=1.8):
    """
    Compare each category's most recent week of spend against its historical
    weekly average + std. Returns a list of human-readable alert strings.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)

    weekly = df.groupby(["category", "week"])["amount"].sum().reset_index()

    alerts = []

    for cat, group in weekly.groupby("category"):
        # use this category's own most recent week (categories aren't hit
        # every single week, so a global latest-week would miss most of them)
        cat_latest_week = group["week"].max()
        history = group[group["week"] < cat_latest_week]["amount"]
        current = group[group["week"] == cat_latest_week]["amount"]
        if len(history) < 3 or current.empty:
            continue

        mean, std = history.mean(), history.std(ddof=0)
        current_val = current.iloc[0]
        if std == 0:
            continue
        z = (current_val - mean) / std

        if z > z_thresh:
            multiple = current_val / mean if mean > 0 else float("inf")
            alerts.append(
                f"You spent {multiple:.1f}x your usual on {cat} this week "
                f"(₹{current_val:,.0f} vs typical ₹{mean:,.0f})."
            )
    return alerts


if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv")
    flagged = flag_transaction_outliers(df)
    n_out = flagged["is_outlier"].sum()
    print(f"Flagged {n_out} outlier transactions out of {len(df)}")
    print(flagged[flagged["is_outlier"]][["date", "description", "amount", "category"]].head(10))

    print("\nWeekly spend alerts:")
    for a in weekly_spend_alerts(df):
        print(" -", a)
