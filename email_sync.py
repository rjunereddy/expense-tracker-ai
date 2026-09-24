import os
import re
import time
import email
import imaplib
import pandas as pd
from datetime import datetime
from email.header import decode_header
from categorize import load_model, predict
from db import append_transaction

# Configuration (to be set via environment variables)
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
EMAIL_ACCOUNT = os.getenv("EMAIL_ACCOUNT")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") # App Password if using Gmail
USERNAME = os.getenv("APP_USERNAME", "demo_user")

def parse_bank_email_body(body_text: str):
    """
    Tries to extract amount and merchant from a generic bank email body.
    """
    # Strip HTML tags
    body_text = re.sub(r'<[^>]+>', ' ', body_text)
    
    # Clean up whitespace and newlines for easier regex matching
    body_text = re.sub(r'\s+', ' ', body_text).replace(',', '')
    
    # Try to find amount (Rs 500, INR 500, ₹ 500, USD 50, $ 50)
    amt_match = re.search(r'(?:rs\.?|inr|₹|usd|\$)\s*(\d+(?:\.\d+)?)', body_text, re.IGNORECASE)
    if not amt_match:
        return None, None
    amount = float(amt_match.group(1))
    
    # Try to find merchant (at <merchant>, to <merchant>, for <merchant>)
    merchant = "Unknown Merchant"
    merchant_match = re.search(r'(?:at|to|info|for|from|in)\s+([A-Za-z0-9\s*#\-]+?)(?:\s+on|\s+ref|\s+via|\.|$)', body_text, re.IGNORECASE)
    if merchant_match:
        merchant = merchant_match.group(1).strip()
        
    return amount, merchant

def check_new_emails():
    if not EMAIL_ACCOUNT or not EMAIL_PASSWORD:
        print("ERROR: Please set EMAIL_ACCOUNT and EMAIL_PASSWORD environment variables.")
        return
        
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
        mail.select("inbox")
        
        # Search for all unread emails (we will filter in python to be safe)
        status, messages = mail.search(None, 'UNSEEN')
        
        if messages[0]:
            print(f"[*] Found {len(messages[0].split())} unread email(s) in inbox. Analyzing...")
        else:
            print("[-] No unread emails found in inbox.")
        
        if status == "OK" and messages[0]:
            email_ids = messages[0].split()
            for e_id in email_ids:
                res, msg_data = mail.fetch(e_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        subject, encoding = decode_header(msg["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding if encoding else "utf-8")
                            
                        # Extract body text
                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() in ["text/plain", "text/html"]:
                                    body += part.get_payload(decode=True).decode(errors="ignore") + " "
                        else:
                            body = msg.get_payload(decode=True).decode(errors="ignore")
                            
                        amount, merchant = parse_bank_email_body(body)
                        
                        if amount:
                            print(f"\n[+] Detected new transaction email!")
                            print(f"    Amount: ₹{amount:,.2f}")
                            print(f"    Merchant: {merchant}")
                            log_transaction(amount, merchant)
                        
                        # Note: The email is automatically marked as SEEN (read) by the fetch command.
                            
    except Exception as e:
        print(f"Error checking email: {e}")
    finally:
        try:
            mail.logout()
        except:
            pass

def log_transaction(amount, merchant):
    try:
        model = load_model("data/categorizer.joblib")
    except FileNotFoundError:
        print("[-] ML Model not found. Generating base model now...")
        from generate_data import generate
        from categorize import train
        train(generate(n_days=180))
        model = load_model("data/categorizer.joblib")
        
    pred_cat, conf = predict([merchant], model)[0]
    
    new_row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "description": merchant,
        "amount": round(amount, 2),
        "category": pred_cat,
        "type": "Expense"
    }
    
    append_transaction(USERNAME, new_row)
    print(f"[*] Logged & Categorized as '{pred_cat}' ({(conf*100):.0f}% confidence)")

if __name__ == "__main__":
    if os.getenv("RUN_ONCE"):
        print("="*60)
        print("📧 Running one-off Email Sync (GitHub Actions)")
        print("="*60)
        check_new_emails()
    else:
        print("="*60)
        print("📧 Email Sync Worker Started")
        print("Listening for unread bank alert emails...")
        print("="*60)
        while True:
            check_new_emails()
            time.sleep(60) # Polling every 60 seconds
