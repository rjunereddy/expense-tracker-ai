import os
import hashlib
import uuid
import pandas as pd

USERS_PATH = "data/users.csv"


def hash_password(password: str, salt: str) -> str:
    """Hash password using SHA-256 and a unique salt."""
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def register_user(username: str, password: str, full_name: str) -> tuple[bool, str]:
    """Register a new user in the credentials store."""
    username = username.strip().lower()
    full_name = full_name.strip()
    
    if not username or not password or not full_name:
        return False, "All fields are required."
        
    os.makedirs("data", exist_ok=True)
    
    # Load existing users if the database exists
    if os.path.exists(USERS_PATH):
        df = pd.read_csv(USERS_PATH)
        if username in df["username"].astype(str).str.lower().values:
            return False, f"Username '{username}' is already taken."
    else:
        df = pd.DataFrame(columns=["username", "password_hash", "salt", "full_name"])
        
    # Generate unique salt and hash
    salt = uuid.uuid4().hex
    pwd_hash = hash_password(password, salt)
    
    # Append new user record
    new_user = {
        "username": username,
        "password_hash": pwd_hash,
        "salt": salt,
        "full_name": full_name
    }
    
    df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
    df.to_csv(USERS_PATH, index=False)
    return True, "Registration successful."


def authenticate_user(username: str, password: str) -> tuple[bool, dict]:
    """Authenticate user credentials. Returns status and user metadata."""
    username = username.strip().lower()
    if not username or not password:
        return False, {}
        
    if not os.path.exists(USERS_PATH):
        return False, {}
        
    df = pd.read_csv(USERS_PATH)
    user_rows = df[df["username"].astype(str).str.lower() == username]
    
    if user_rows.empty:
        return False, {}
        
    user = user_rows.iloc[0]
    stored_hash = user["password_hash"]
    salt = user["salt"]
    full_name = user["full_name"]
    
    computed_hash = hash_password(password, salt)
    if computed_hash == stored_hash:
        return True, {"username": username, "full_name": full_name}
        
    return False, {}
