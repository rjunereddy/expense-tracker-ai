"""
Subscription and Recurring Expense Detection.

Analyzes transaction history to find recurring patterns (same merchant, 
similar amount, consistent time interval like weekly or monthly).
"""
import pandas as pd
import numpy as np
from categorize import clean_text

def detect_subscriptions(df: pd.DataFrame):
    """
    Detects recurring transactions and returns a summary DataFrame.
    Criteria for a subscription:
    - At least 3 occurrences
    - Very similar cleaned merchant name
    - Low variance in amount (or identical)
    - Consistent interval (~7 days or ~30 days)
    """
    df = df.copy()
    if df.empty or "amount" not in df.columns:
        return pd.DataFrame()
        
    df["date"] = pd.to_datetime(df["date"])
    
    # We only look at expenses (amount > 0), assuming income is separate or positive
    expenses = df[df["amount"] > 0].copy()
    
    # Clean merchant name to group similar transactions
    expenses["clean_merchant"] = expenses["description"].apply(lambda x: clean_text(x).split()[0] if clean_text(x) else "UNKNOWN")
    
    subs = []
    
    for merchant, group in expenses.groupby("clean_merchant"):
        if len(group) < 3:
            continue
            
        group = group.sort_values("date")
        
        # Check amount consistency (std dev / mean should be very small)
        mean_amt = group["amount"].mean()
        std_amt = group["amount"].std()
        
        if std_amt / mean_amt > 0.15: # Allow small variations
            continue
            
        # Check time interval consistency
        date_diffs = group["date"].diff().dt.days.dropna()
        mean_diff = date_diffs.mean()
        std_diff = date_diffs.std()
        
        # Is it monthly (~30 days) or weekly (~7 days)?
        is_monthly = 25 <= mean_diff <= 35 and std_diff < 5
        is_weekly = 5 <= mean_diff <= 9 and std_diff < 3
        
        if is_monthly or is_weekly:
            subs.append({
                "merchant": merchant,
                "frequency": "Monthly" if is_monthly else "Weekly",
                "avg_amount": round(mean_amt, 2),
                "last_paid": group["date"].max().strftime("%Y-%m-%d"),
                "estimated_yearly_cost": round(mean_amt * (12 if is_monthly else 52), 2)
            })
            
    return pd.DataFrame(subs).sort_values("estimated_yearly_cost", ascending=False)
