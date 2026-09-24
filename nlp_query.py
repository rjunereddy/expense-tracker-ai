"""
Natural-language query layer over the structured transaction data.

Design choice: rather than piping the question straight to an LLM and
trusting whatever number it hallucinates, this parses the question into a
structured intent (category, time range, aggregation) and then runs a real
pandas query — the LLM (or, here, a rule-based fallback) only chooses WHICH
query to run, it never invents the numeric answer. This is the safer pattern
for any product surfacing financial figures.

If ANTHROPIC_API_KEY is available in the environment, this module can be
extended to use Claude for intent parsing on messier phrasing; the rule-based
parser below is a fully offline fallback so the project runs with zero
external dependencies or API keys.
"""
import re
import pandas as pd
from datetime import datetime, timedelta

CATEGORY_ALIASES = {
    "food": "Food & Dining", "dining": "Food & Dining", "eating out": "Food & Dining",
    "groceries": "Groceries", "grocery": "Groceries",
    "shopping": "Shopping",
    "transport": "Transport", "travel": "Travel", "commute": "Transport",
    "utilities": "Utilities", "bills": "Utilities",
    "entertainment": "Entertainment", "movies": "Entertainment",
    "health": "Health", "medical": "Health",
    "education": "Education", "courses": "Education",
    "income": "Income", "salary": "Income", "earnings": "Income",
}

MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"]


def _find_category(q: str):
    for alias, cat in CATEGORY_ALIASES.items():
        if alias in q:
            return cat
    return None


def _find_time_range(q: str, today: datetime, df_dates=None):
    q = q.lower()
    
    # Check for specific date ranges first (YYYY-MM-DD or YYYY/MM/DD)
    dates = re.findall(r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b', q)
    if len(dates) >= 2:
        try:
            start = datetime.strptime(dates[0].replace('/', '-'), "%Y-%m-%d")
            end = datetime.strptime(dates[1].replace('/', '-'), "%Y-%m-%d")
            return start, end
        except ValueError:
            pass
    elif len(dates) == 1:
        try:
            date_val = datetime.strptime(dates[0].replace('/', '-'), "%Y-%m-%d")
            if "after" in q or "since" in q or "from" in q:
                return date_val, today
            elif "before" in q or "until" in q or "to" in q:
                min_date = pd.to_datetime(df_dates.min()) if df_dates is not None else date_val - timedelta(days=30)
                return min_date, date_val
            else:
                # just that day
                return date_val, date_val + timedelta(hours=23, minutes=59)
        except ValueError:
            pass

    if "last month" in q:
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start, last_month_end
    if "this month" in q:
        return today.replace(day=1), today
    if "last week" in q:
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        return start, end
    if "this week" in q:
        start = today - timedelta(days=today.weekday())
        return start, today
    for i, m in enumerate(MONTH_NAMES):
        if m in q:
            year = today.year
            start = datetime(year, i + 1, 1)
            end_month = i + 2
            end_year = year
            if end_month > 12:
                end_month, end_year = 1, year + 1
            end = datetime(end_year, end_month, 1) - timedelta(days=1)
            # If the calculated range is in the future, check if last year's month makes more sense
            if start > today:
                start = start.replace(year=year - 1)
                end = end.replace(year=year - 1)
            return start, end
    # default: last 30 days
    return today - timedelta(days=30), today


def answer_query(question: str, df: pd.DataFrame, today: datetime = None) -> str:
    """Parses a natural-language question and returns a grounded answer
    computed directly from the dataframe (never from a generative model)."""
    if today is None:
        today = pd.to_datetime(df["date"]).max()

    q = question.lower()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    category = _find_category(q)
    start, end = _find_time_range(q, today, df["date"])

    mask = (df["date"] >= start) & (df["date"] <= end)
    
    # 1. Category filter
    if category:
        mask &= (df["category"] == category)

    # 2. Merchant filter
    merchant_filter = None
    # Check for direct mentions of known merchants or regex extraction
    match = re.search(r'\b(?:at|for|from|to|with|merchant)\s+([a-z0-9\s&*#-]+)', q)
    if match:
        term = match.group(1).strip()
        # filter out common stopwords or categories
        if term not in CATEGORY_ALIASES and term not in ["last month", "this month", "last week", "this week", "today", "yesterday"]:
            merchant_filter = term
    
    # fallback to matching known merchants in the string
    known_merchants = [
        "swiggy", "zomato", "mcdonalds", "dominos", "starbucks", "cafe coffee day", "burger king",
        "bigbasket", "dmart", "more supermarket", "reliance fresh", "blinkit", "zepto",
        "amazon", "amzn", "flipkart", "myntra", "ajio", "nykaa",
        "uber", "ola", "irctc", "rapido", "bmtc",
        "bescom", "airtel", "jio", "bwssb", "act fibernet",
        "netflix", "spotify", "pvr", "bookmyshow", "hotstar",
        "apollo", "practo", "1mg", "cult fit",
        "nobroker", "housing rent", "nestaway",
        "udemy", "coursera", "pesu", "byjus",
        "makemytrip", "goibibo", "airbnb", "redbus"
    ]
    for m in known_merchants:
        if m in q:
            merchant_filter = m
            break

    if merchant_filter:
        mask &= df["description"].str.lower().str.contains(merchant_filter, regex=False)

    # 3. Amount/Numeric comparators filter
    amount_min = None
    amount_max = None
    
    # Check "between X and Y"
    between_match = re.search(r'\bbetween\s+(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?)\s+and\s+(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?)', q)
    if between_match:
        amount_min = float(between_match.group(1))
        amount_max = float(between_match.group(2))
    else:
        # Check above/greater/more/over
        above_match = re.search(r'\b(?:above|greater\s+than|more\s+than|over|>)\s+(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?)', q)
        if above_match:
            amount_min = float(above_match.group(1))
        
        # Check below/less/under/lower/under
        below_match = re.search(r'\b(?:below|less\s+than|under|lower\s+than|<)\s+(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?)', q)
        if below_match:
            amount_max = float(below_match.group(1))

    if amount_min is not None:
        mask &= (df["amount"] >= amount_min)
    if amount_max is not None:
        mask &= (df["amount"] <= amount_max)

    subset = df[mask]

    period_str = f"{start.strftime('%b %d')} to {end.strftime('%b %d, %Y')}"
    
    # Build filter description
    filter_details = []
    if category:
        filter_details.append(f"on {category}")
    if merchant_filter:
        filter_details.append(f"at '{merchant_filter.upper()}'")
    if amount_min is not None and amount_max is not None:
        filter_details.append(f"between ₹{amount_min:,.0f} and ₹{amount_max:,.0f}")
    elif amount_min is not None:
        filter_details.append(f"above ₹{amount_min:,.0f}")
    elif amount_max is not None:
        filter_details.append(f"below ₹{amount_max:,.0f}")
        
    scope = " ".join(filter_details) if filter_details else "overall"

    if subset.empty:
        return f"No transactions found {scope} between {period_str}."

    total = subset["amount"].sum()
    count = len(subset)

    # Aggregations
    if "how many" in q or "count" in q or "number of" in q:
        return f"You made {count} transaction{'s' if count > 1 else ''} {scope} between {period_str}, totaling ₹{total:,.2f}."
    if "average" in q or "avg" in q:
        return f"Your average transaction {scope} between {period_str} was ₹{total / count:,.2f} ({count} transaction{'s' if count > 1 else ''})."
    if "highest" in q or "max" in q or "most expensive" in q:
        max_row = subset.loc[subset["amount"].idxmax()]
        return f"Your highest transaction {scope} between {period_str} was ₹{max_row['amount']:,.2f} on {pd.to_datetime(max_row['date']).strftime('%b %d')} for '{max_row['description']}'."
    if "top merchants" in q or "where did i spend the most" in q:
        top_merchants = subset.groupby("description")["amount"].sum().sort_values(ascending=False).head(3)
        merchants_str = ", ".join([f"'{m}' (₹{a:,.0f})" for m, a in top_merchants.items()])
        return f"Your top 3 merchants {scope} between {period_str} were: {merchants_str}."
    if "lowest" in q or "min" in q or "cheapest" in q:
        min_row = subset.loc[subset["amount"].idxmin()]
        return f"Your lowest transaction {scope} between {period_str} was ₹{min_row['amount']:,.2f} on {pd.to_datetime(min_row['date']).strftime('%b %d')} for '{min_row['description']}'."

    # Default to total spend
    return f"You spent ₹{total:,.2f} {scope} between {period_str} ({count} transaction{'s' if count > 1 else ''})."


if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv")
    questions = [
        "How much did I spend on food last month?",
        "What's my total grocery spend this month?",
        "How many shopping transactions did I make last week?",
        "What was my highest dining transaction above 1000 last month?",
        "How much did I spend at Uber in June?",
        "How many transactions between 500 and 2000 did I make this month?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {answer_query(q, df)}\n")
