import os
import pandas as pd

# Load environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)

if USE_SUPABASE:
    from supabase import create_client, Client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_transactions(username: str) -> pd.DataFrame:
    if USE_SUPABASE:
        try:
            response = supabase.table("transactions").select("*").eq("username", username).execute()
            df = pd.DataFrame(response.data)
            if df.empty:
                df = pd.DataFrame(columns=["txn_id", "date", "description", "amount", "category", "type", "username"])
            return df
        except Exception as e:
            print(f"Supabase read error: {e}")
            return pd.DataFrame()
    else:
        user_path = f"data/transactions_{username}.csv"
        if os.path.exists(user_path):
            df = pd.read_csv(user_path)
            if "type" not in df.columns:
                df["type"] = "Expense"
            return df
        return pd.DataFrame(columns=["txn_id", "date", "description", "amount", "category", "type"])

def save_transactions(username: str, df: pd.DataFrame):
    """Saves the entire dataframe (used after editing in the UI)"""
    if USE_SUPABASE:
        try:
            df_copy = df.copy()
            df_copy["username"] = username
            records = df_copy.to_dict(orient="records")
            # Upsert requires a primary key in Supabase (txn_id, username)
            supabase.table("transactions").upsert(records).execute()
        except Exception as e:
            print(f"Supabase save error: {e}")
    else:
        os.makedirs("data", exist_ok=True)
        user_path = f"data/transactions_{username}.csv"
        df.to_csv(user_path, index=False)

def append_transaction(username: str, row: dict):
    """Appends a single transaction (used by bots and manual logger)"""
    if USE_SUPABASE:
        try:
            row_copy = row.copy()
            row_copy["username"] = username
            supabase.table("transactions").insert(row_copy).execute()
        except Exception as e:
            print(f"Supabase append error: {e}")
    else:
        df = get_transactions(username)
        # Auto-increment txn_id for CSV
        if "txn_id" not in row or row["txn_id"] is None:
            row["txn_id"] = int(df["txn_id"].max() + 1) if not df.empty else 1
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_transactions(username, df)
