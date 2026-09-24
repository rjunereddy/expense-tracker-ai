"""
Generates synthetic bank-transaction-like data for the expense tracker.
Mimics messy real-world merchant strings (e.g. "AMZN MKTP US*2K3") so the
categorization model has to work on realistic noisy text, not clean labels.
"""
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# category -> list of (raw_description_templates, typical_amount_range)
CATEGORIES = {
    "Food & Dining": {
        "merchants": ["SWIGGY BANGALORE", "ZOMATO ORDER", "MCDONALDS IN", "DOMINOS PIZZA",
                      "STARBUCKS COFFEE", "CCD CAFE COFFEE DAY", "BURGER KING IND"],
        "amount": (100, 1200),
    },
    "Groceries": {
        "merchants": ["BIGBASKET ONLINE", "DMART RETAIL", "MORE SUPERMARKET", "RELIANCE FRESH",
                      "BLINKIT GROCERY", "ZEPTO INSTANT"],
        "amount": (300, 3500),
    },
    "Shopping": {
        "merchants": ["AMZN MKTP US*2K3", "FLIPKART INTERNET", "MYNTRA DESIGNS",
                      "AJIO TRENDS", "NYKAA FASHION"],
        "amount": (200, 5000),
    },
    "Transport": {
        "merchants": ["UBER TRIP HELP.UBER.COM", "OLA CABS BANGALORE", "IRCTC RAIL TICKET",
                      "RAPIDO BIKE TAXI", "BMTC BUS PASS"],
        "amount": (50, 900),
    },
    "Utilities": {
        "merchants": ["BESCOM ELECTRICITY", "AIRTEL POSTPAID", "JIO FIBER RECHARGE",
                      "BWSSB WATER BOARD", "ACT FIBERNET"],
        "amount": (300, 2500),
    },
    "Entertainment": {
        "merchants": ["NETFLIX.COM", "SPOTIFY PREMIUM", "PVR CINEMAS", "BOOKMYSHOW EVENT",
                      "HOTSTAR DISNEY"],
        "amount": (150, 1500),
    },
    "Health": {
        "merchants": ["APOLLO PHARMACY", "PRACTO CONSULT", "1MG HEALTHCARE", "CULT FIT GYM"],
        "amount": (200, 4000),
    },
    "Rent & Housing": {
        "merchants": ["NOBROKER RENT PAY", "HOUSING RENT NEFT", "NESTAWAY PROPERTY"],
        "amount": (8000, 25000),
    },
    "Education": {
        "merchants": ["UDEMY COURSE PAY", "COURSERA SUBSCRIP", "PESU FEE PAYMENT",
                      "BYJUS LEARNING"],
        "amount": (500, 15000),
    },
    "Travel": {
        "merchants": ["MAKEMYTRIP BOOKING", "GOIBIBO FLIGHT", "AIRBNB STAY", "REDBUS TICKET"],
        "amount": (1000, 20000),
    },
    "Income": {
        "merchants": ["SALARY NEFT", "FREELANCE PAYOUT", "DIVIDEND ACH", "CASH DEPOSIT"],
        "amount": (40000, 120000),
    }
}

NOISE_SUFFIXES = ["", " REF#{}".format(random.randint(1000, 9999)), "*{}".format(random.randint(10, 99)),
                   " TXN{}".format(random.randint(100000, 999999)), ""]


def _noisy(merchant):
    """Add realistic noise to a merchant string, like real bank statements do."""
    suffix = random.choice(NOISE_SUFFIXES)
    if random.random() < 0.3:
        merchant = merchant + suffix
    if random.random() < 0.15:
        merchant = merchant.replace(" ", random.choice(["  ", "_", " "]))
    return merchant


def generate(n_days=180, start_date=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if start_date is None:
        start_date = datetime.now() - timedelta(days=n_days)

    rows = []
    txn_id = 1
    for day_offset in range(n_days):
        date = start_date + timedelta(days=day_offset)
        # simulate variable number of transactions per day (more on weekends)
        is_weekend = date.weekday() >= 5
        n_txns = np.random.poisson(3.5 if is_weekend else 2.0)

        for _ in range(n_txns):
            cat = random.choices(
                list(CATEGORIES.keys()),
                weights=[0.20, 0.15, 0.15, 0.12, 0.08, 0.08, 0.06, 0.05, 0.05, 0.04, 0.02],
            )[0]
            info = CATEGORIES[cat]
            merchant = random.choice(info["merchants"])
            desc = _noisy(merchant)
            lo, hi = info["amount"]
            amount = round(np.random.uniform(lo, hi), 2)

            # occasionally inject a spending spike -> useful for anomaly detection later
            if random.random() < 0.02:
                amount *= random.uniform(3, 6)

            rows.append({
                "txn_id": txn_id,
                "date": date.strftime("%Y-%m-%d"),
                "description": desc,
                "amount": round(amount, 2),
                "category": cat,  # ground truth label, used only for training/eval
                "type": "Income" if cat == "Income" else "Expense"
            })
            txn_id += 1

    # rent is monthly, not daily — add explicitly once a month
    for month in range(n_days // 30 + 1):
        date = start_date + timedelta(days=month * 30 + random.randint(0, 4))
        merchant = random.choice(CATEGORIES["Rent & Housing"]["merchants"])
        lo, hi = CATEGORIES["Rent & Housing"]["amount"]
        rows.append({
            "txn_id": txn_id,
            "date": date.strftime("%Y-%m-%d"),
            "description": _noisy(merchant),
            "amount": round(np.random.uniform(lo, hi), 2),
            "category": "Rent & Housing",
            "type": "Expense"
        })
        txn_id += 1

    # Inject exact recurring subscriptions for detection feature
    for month in range(n_days // 30 + 1):
        date = start_date + timedelta(days=month * 30 + 2)
        rows.append({
            "txn_id": txn_id, "date": date.strftime("%Y-%m-%d"),
            "description": "NETFLIX.COM", "amount": 649.00,
            "category": "Entertainment", "type": "Expense"
        })
        txn_id += 1
        rows.append({
            "txn_id": txn_id, "date": date.strftime("%Y-%m-%d"),
            "description": "SPOTIFY PREMIUM", "amount": 119.00,
            "category": "Entertainment", "type": "Expense"
        })
        txn_id += 1

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # deliberately inject a visible spending spike in the final week so the
    # anomaly-alert feature has something concrete to demonstrate
    last_week_start = start_date + timedelta(days=n_days - 6)
    for _ in range(4):
        date = last_week_start + timedelta(days=random.randint(0, 5))
        merchant = random.choice(CATEGORIES["Food & Dining"]["merchants"])
        rows.append({
            "txn_id": txn_id,
            "date": date.strftime("%Y-%m-%d"),
            "description": _noisy(merchant),
            "amount": round(np.random.uniform(900, 1600), 2),
            "category": "Food & Dining",
            "type": "Expense"
        })
        txn_id += 1

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df



if __name__ == "__main__":
    df = generate(n_days=180)
    df.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(df)} transactions across {df['category'].nunique()} categories")
    print(df.groupby("category")["amount"].agg(["count", "sum"]).sort_values("sum", ascending=False))
