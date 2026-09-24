import os
import re
import pandas as pd
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from categorize import load_model, predict
from db import append_transaction

def parse_sms(text: str):
    """
    Attempts to extract Amount and Merchant from typical bank SMS or manual entry.
    """
    text = text.replace(',', '') # remove commas from numbers
    
    # Try to find amount (Rs 500, INR 500, 500)
    amt_match = re.search(r'(?:rs\.?|inr|₹|usd|\$)\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if not amt_match:
        # Check if they just typed "500 merchant"
        amt_match = re.search(r'^(\d+(?:\.\d+)?)\s+', text)
        
    if not amt_match:
        return None, None
        
    amount = float(amt_match.group(1))
    
    # Try to find merchant
    merchant = text
    # Match "at <merchant> on" or "to <merchant>"
    merchant_match = re.search(r'(?:at|to)\s+([A-Za-z0-9\s*#]+?)(?:\s+on|\s+ref|\s+via|\.|$)', text, re.IGNORECASE)
    if merchant_match:
        merchant = merchant_match.group(1).strip()
    else:
        # If manual entry like "500 swiggy"
        if re.search(r'^\d+(?:\.\d+)?\s+(.+)$', text):
            merchant = re.search(r'^\d+(?:\.\d+)?\s+(.+)$', text).group(1)
            
    return amount, merchant

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💸 Expense Tracker Bot is active!\n"
        "Forward your bank SMS here, or just type manual expenses like '500 Starbucks'.\n"
        "I will auto-categorize them and sync to your dashboard."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    
    amount, merchant = parse_sms(text)
    
    if not amount:
        await update.message.reply_text("Could not detect an amount. Try forwarding a bank SMS or type '500 Merchant Name'.")
        return
        
    # Categorize using our local ML model
    # We fallback to the general demo model if user-specific model isn't built yet
    try:
        model = load_model()
    except FileNotFoundError:
        await update.message.reply_text("ML Model not found. Please run the dashboard once to train the initial model.")
        return

    pred_cat, conf = predict([merchant], model)[0]
    
    # Link Telegram Username to App Username (using telegram ID for safety/uniqueness)
    username = update.message.from_user.username or "demo_user"
    
    new_row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "description": merchant,
        "amount": round(amount, 2),
        "category": pred_cat,
        "type": "Expense"
    }
    
    append_transaction(username, new_row)
    
    await update.message.reply_text(
        f"✅ Logged! ₹{amount:,.2f} at {merchant}\n"
        f"🤖 Auto-categorized as: {pred_cat} ({(conf*100):.0f}% confidence)"
    )

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("="*60)
        print("ERROR: TELEGRAM_BOT_TOKEN environment variable not set.")
        print("\nHow to start:")
        print("1. Open Telegram and search for '@BotFather'")
        print("2. Send '/newbot' and follow the steps to get an API Token.")
        print("3. Run this command in your terminal to set it (Windows PowerShell):")
        print("   $env:TELEGRAM_BOT_TOKEN='your_token_here'")
        print("4. Run this script again: python telegram_bot.py")
        print("="*60)
        return
        
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Bot is running... Open Telegram and forward your bank SMS to your bot!")
    app.run_polling()

if __name__ == "__main__":
    main()
