"""
Spend forecasting per category using weighted linear regression on weekly
aggregates, with a simple confidence interval derived from residual std.

Kept intentionally simple (no Prophet/ARIMA dependency) so it runs anywhere
in under a second — the interesting part of this project is the pipeline
(categorize -> detect -> forecast -> explain), not the forecasting model
itself. Swapping in Prophet/ARIMA later is a natural "future work" bullet.
"""
import pandas as pd
import numpy as np


def forecast_next_period(df: pd.DataFrame, periods_ahead=1):
    """
    Returns a DataFrame: category, forecast_amount, lower_bound, upper_bound
    for the next `periods_ahead` week(s), per category.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)

    weekly = df.groupby(["category", "week"])["amount"].sum().reset_index()

    results = []
    for cat, group in weekly.groupby("category"):
        group = group.sort_values("week")
        y = group["amount"].values
        n = len(y)

        if n < 4:
            # not enough history for a trend line — fall back to mean
            forecast = y.mean() if n else 0
            resid_std = y.std(ddof=0) if n > 1 else forecast * 0.3
        else:
            x = np.arange(n)
            # linear fit with recency weighting (recent weeks matter more)
            weights = np.linspace(0.5, 1.5, n)
            coeffs = np.polyfit(x, y, deg=1, w=weights)
            forecast = np.polyval(coeffs, n)  # next time step
            fitted = np.polyval(coeffs, x)
            resid_std = np.std(y - fitted)

        forecast = max(forecast, 0)
        results.append({
            "category": cat,
            "forecast_amount": round(forecast, 2),
            "lower_bound": round(max(forecast - 1.28 * resid_std, 0), 2),  # ~80% CI
            "upper_bound": round(forecast + 1.28 * resid_std, 2),
        })

    return pd.DataFrame(results).sort_values("forecast_amount", ascending=False)


def get_forecast_timeline(df: pd.DataFrame, periods_ahead=4):
    """
    Computes both historical weekly spends and projected spends for the next
    `periods_ahead` weeks, alongside lower/upper bounds for confidence intervals.
    Returns a DataFrame: category, week, amount, type ('Historical' or 'Forecast'), lower_bound, upper_bound
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    
    weekly = df.groupby(["category", "week"])["amount"].sum().reset_index()
    
    timeline_rows = []
    
    for cat, group in weekly.groupby("category"):
        group = group.sort_values("week")
        weeks = group["week"].tolist()
        amounts = group["amount"].tolist()
        n = len(amounts)
        
        # Add all historical weeks
        for i in range(n):
            timeline_rows.append({
                "category": cat,
                "week": weeks[i],
                "amount": amounts[i],
                "type": "Historical",
                "lower_bound": amounts[i],
                "upper_bound": amounts[i]
            })
            
        if n < 4:
            mean_val = np.mean(amounts) if n else 0
            std_val = np.std(amounts) if n > 1 else mean_val * 0.3
            
            last_week = weeks[-1] if n else pd.to_datetime("today")
            for h in range(1, periods_ahead + 1):
                f_week = last_week + pd.Timedelta(weeks=h)
                timeline_rows.append({
                    "category": cat,
                    "week": f_week,
                    "amount": round(max(mean_val, 0), 2),
                    "type": "Forecast",
                    "lower_bound": round(max(mean_val - 1.28 * std_val, 0), 2),
                    "upper_bound": round(mean_val + 1.28 * std_val, 2)
                })
        else:
            x = np.arange(n)
            # recency weighting
            weights = np.linspace(0.5, 1.5, n)
            coeffs = np.polyfit(x, amounts, deg=1, w=weights)
            
            fitted = np.polyval(coeffs, x)
            resid_std = np.std(np.array(amounts) - fitted)
            
            last_week = weeks[-1]
            for h in range(1, periods_ahead + 1):
                f_week = last_week + pd.Timedelta(weeks=h)
                f_val = np.polyval(coeffs, n + h - 1)
                f_val = max(f_val, 0)
                # uncertainty grows over time
                h_std = resid_std * (1.0 + 0.15 * (h - 1))
                
                timeline_rows.append({
                    "category": cat,
                    "week": f_week,
                    "amount": round(f_val, 2),
                    "type": "Forecast",
                    "lower_bound": round(max(f_val - 1.28 * h_std, 0), 2),
                    "upper_bound": round(f_val + 1.28 * h_std, 2)
                })
                
    return pd.DataFrame(timeline_rows)


if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv")
    fc = forecast_next_period(df)
    print(fc.to_string(index=False))
    print(f"\nTotal predicted spend next week: ₹{fc['forecast_amount'].sum():,.0f}")
    
    # test timeline
    timeline = get_forecast_timeline(df, periods_ahead=2)
    print(f"Timeline records: {len(timeline)} (Historical: {len(timeline[timeline['type']=='Historical'])}, Forecast: {len(timeline[timeline['type']=='Forecast'])})")
