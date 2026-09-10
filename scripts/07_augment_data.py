"""
Augments the training set with:
1. Legitimate transactional messages (bank/OTP/delivery/bill/appointment) -> ham
2. Modern smishing that MIMICS these patterns (shortened links, fake urgency,
   requests for PIN/CVV) -> spam
This targets the specific gap found during demo testing: the original dataset's
ham class lacks real-world transactional SMS, causing false positives on
legitimate bank/OTP/delivery texts.

Run from project root: python scripts/07_augment_data.py
Output: data/processed/train_augmented.csv (does NOT touch test.csv)
"""
import random
import pandas as pd
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
OUTPUT_PATH = PROCESSED_DIR / "train_augmented.csv"

banks = ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "Kotak Bank", "PNB", "Bank of Baroda"]
couriers = ["Amazon", "Flipkart", "BlueDart", "DTDC", "Delhivery", "FedEx", "India Post"]
names = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Rohan"]
telcos = ["Airtel", "Jio", "Vodafone Idea", "BSNL"]
apps = ["Netflix", "Spotify", "Amazon Prime", "Gym membership", "Health insurance"]
rides = ["Uber", "Ola", "Swiggy", "Zomato"]
airlines = ["IndiGo", "Air India", "Vistara", "SpiceJet"]

# ---------- LEGITIMATE (ham) templates, spanning many domains ----------
ham_templates = [
    # Banking / finance
    lambda: f"Rs.{random.randint(200,9999)} debited from A/c XX{random.randint(1000,9999)} on {random.randint(1,28)}-{random.choice(['Jul','Aug'])}-26 at {random.choice(banks)}. Avl bal: Rs.{random.randint(1000,50000)}. Not you? Call 1800{random.randint(1000000,9999999)}.",
    lambda: f"Rs.{random.randint(500,20000)} credited to A/c XX{random.randint(1000,9999)} on {random.randint(1,28)}-Aug-26. Avl bal: Rs.{random.randint(1000,50000)}. -{random.choice(banks)}",
    lambda: f"Your credit card statement for Aug 2026 is ready. Total due: Rs.{random.randint(500,15000)}. View in the bank app.",
    lambda: f"EMI of Rs.{random.randint(1000,8000)} for your loan A/c XX{random.randint(1000,9999)} is due on {random.randint(1,28)} Aug. -{random.choice(banks)}",
    # OTP / verification
    lambda: f"{random.randint(100000,999999)} is your OTP for net banking login. Valid 10 min. Do not share with anyone. -{random.choice(banks)}",
    lambda: f"Your OTP for transaction of Rs.{random.randint(100,5000)} is {random.randint(100000,999999)}. Valid for 5 minutes.",
    lambda: f"{random.randint(100000,999999)} is your verification code for account login. Do not share this code.",
    # E-commerce / delivery
    lambda: f"Your {random.choice(couriers)} order #{random.randint(100000,999999)} has been delivered. Track anytime at {random.choice(couriers).lower()}.com/orders",
    lambda: f"Your {random.choice(couriers)} package is out for delivery, expected today by 8 PM. Track at {random.choice(couriers).lower()}.com/track",
    lambda: f"Your return for order #{random.randint(100000,999999)} has been picked up. Refund will be processed in 3-5 days.",
    # Telecom
    lambda: f"Your {random.choice(telcos)} data balance is {random.randint(1,10)}GB, valid till {random.randint(1,28)} Aug. Recharge at official app.",
    lambda: f"Your {random.choice(telcos)} bill of Rs.{random.randint(200,1500)} is generated. Due date: {random.randint(10,28)} Aug 26.",
    # Utilities
    lambda: f"Your electricity bill of Rs.{random.randint(300,3000)} is due on {random.randint(10,28)} Aug. Pay via official MSEB app or website.",
    lambda: f"Hi {random.choice(names)}, your gas cylinder booking is confirmed. Delivery expected in 2-3 days.",
    # Government / tax
    lambda: f"Your Aadhaar update request has been processed successfully. No further action needed.",
    lambda: f"Your ITR for AY 2025-26 has been processed. Refund of Rs.{random.randint(1000,20000)}, if any, will be credited to your bank account.",
    # Healthcare
    lambda: f"Reminder: appointment with Dr. {random.choice(names)} tomorrow at {random.randint(9,17)}:00. Reply CONFIRM or call clinic to reschedule.",
    lambda: f"Your lab test results are ready. Visit the diagnostic center or check the patient portal to view.",
    lambda: f"Your prescription refill for next month is ready for pickup at the pharmacy.",
    # Education
    lambda: f"Your semester exam results have been declared. Login to the college portal to check your grades.",
    lambda: f"Fee payment reminder: Rs.{random.randint(5000,50000)} due by {random.randint(10,28)} Aug for this semester.",
    # Travel
    lambda: f"Your {random.choice(airlines)} flight {random.choice(['6E','AI','UK','SG'])}{random.randint(100,999)} is confirmed for {random.randint(10,28)} Aug. Web check-in opens 48hrs prior.",
    lambda: f"Your train PNR {random.randint(1000000000,9999999999)} status: Confirmed, Seat {random.randint(1,72)}, Coach {random.choice(['A1','B2','S3'])}.",
    # Subscriptions
    lambda: f"Your {random.choice(apps)} subscription renews on {random.randint(1,28)} Aug for Rs.{random.randint(99,999)}. Manage anytime in account settings.",
    lambda: f"Thanks for your payment of Rs.{random.randint(100,2000)}. Your subscription is active until {random.randint(1,28)} Sep 26.",
    # Ride-hailing / food delivery
    lambda: f"Your {random.choice(rides)} ride is confirmed. Driver arriving in {random.randint(2,10)} mins.",
    lambda: f"Your {random.choice(rides)} order has been placed and will arrive in {random.randint(20,45)} mins.",
    # Work / professional
    lambda: f"Reminder: Team meeting scheduled tomorrow at {random.randint(9,17)}:00. Check calendar invite for details.",
    lambda: f"Your interview with the hiring team is confirmed for {random.randint(10,28)} Aug at {random.randint(9,17)}:00.",
    # Casual / personal (reinforces existing ham distribution)
    lambda: f"Hey, are we still on for lunch tomorrow at {random.choice([12, 1, 2])}?",
    lambda: f"Hi {random.choice(names)}, reaching home in 10 mins, see you soon.",
]

# ---------- MODERN SMISHING (spam) templates that mimic the above, across domains ----------
spam_templates = [
    lambda: f"URGENT: Your {random.choice(banks)} account will be suspended. Verify now at bit.ly/verify{random.randint(10,99)} or lose access.",
    lambda: f"Your OTP {random.randint(100000,999999)} was requested. If not you, click bit.ly/secure{random.randint(10,99)} to block immediately.",
    lambda: f"Your {random.choice(couriers)} package is held at customs. Pay Rs.{random.randint(20,99)} fee now: bit.ly/pay{random.randint(10,99)}",
    lambda: f"Dear customer, your account is temporarily locked. Confirm your PIN and CVV at bit.ly/unlock{random.randint(10,99)} to restore access.",
    lambda: f"Congratulations! You've won a refund of Rs.{random.randint(500,5000)}. Claim now: bit.ly/refund{random.randint(10,99)}",
    lambda: f"Your bank KYC is expiring today. Update immediately at bit.ly/kyc{random.randint(10,99)} or account will be blocked.",
    lambda: f"Delivery failed. Reschedule and pay Rs.{random.randint(10,50)} redelivery fee here: bit.ly/redeliver{random.randint(10,99)}",
    lambda: f"Your {random.choice(telcos)} SIM will be deactivated today. Verify your number now: bit.ly/simverify{random.randint(10,99)}",
    lambda: f"Your Aadhaar has been suspended due to suspicious activity. Verify immediately: bit.ly/aadhaar{random.randint(10,99)}",
    lambda: f"Income tax refund of Rs.{random.randint(5000,50000)} pending. Claim now at bit.ly/itr{random.randint(10,99)} before it expires.",
    lambda: f"Your {random.choice(apps)} subscription payment failed. Update card details now: bit.ly/pay{random.randint(10,99)} or lose access.",
    lambda: f"Your electricity connection will be disconnected tonight. Pay pending bill now: bit.ly/elec{random.randint(10,99)}",
]

def build_augmented(n_ham=350, n_spam=120):
    ham_rows = [{"text": random.choice(ham_templates)(), "label": 0, "source": "augmented"} for _ in range(n_ham)]
    spam_rows = [{"text": random.choice(spam_templates)(), "label": 1, "source": "augmented"} for _ in range(n_spam)]
    return pd.DataFrame(ham_rows + spam_rows)


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH)
    print(f"Original train.csv: {len(train)} rows")
    print(train["label"].value_counts())

    aug = build_augmented(n_ham=350, n_spam=120)
    print(f"\nGenerated {len(aug)} synthetic rows ({(aug['label']==0).sum()} ham, {(aug['label']==1).sum()} spam)")

    combined = pd.concat([train, aug], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"])
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(combined)} rows to {OUTPUT_PATH}")
    print(combined["label"].value_counts())