"""
CLI entry point for the combined smishing pipeline.

    python run_pipeline.py "VM-HDFCBK: Your account is suspended. Verify at http://hdfc-secure.tk/login"
    python run_pipeline.py --sender VM-HDFCBK "Your account is suspended. Verify at hdfc-secure.tk/login"
    python run_pipeline.py --demo
    python run_pipeline.py --json "..."
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pipeline.pipeline import analyze_sms, format_report

DEMO_MESSAGES = [
    "VM-HDFCBK: Your account has been suspended. Verify immediately at http://hdfc-secure-verify.tk/login",
    "AD-AMAZON: Your order #402-889 has been shipped. Track it at https://www.amazon.com/orders",
    "Hey, are we still on for dinner at 8?",
    "Congratulations! You have won a $500 gift card. Claim now: bit.ly/3xKq9Zz",
]


def main():
    parser = argparse.ArgumentParser(description="Run an SMS through the smishing pipeline.")
    parser.add_argument("message", nargs="?", help="the incoming SMS text")
    parser.add_argument("--sender", help="sender ID, when it arrives as metadata rather than in the body")
    parser.add_argument("--json", action="store_true", help="print the raw result dict as JSON")
    parser.add_argument("--demo", action="store_true", help="run the built-in sample messages")
    args = parser.parse_args()

    if args.demo:
        messages = [(m, None) for m in DEMO_MESSAGES]
    elif args.message:
        messages = [(args.message, args.sender)]
    else:
        parser.error("give a message, or use --demo")

    for msg, sender in messages:
        result = analyze_sms(msg, sender_id=sender)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_report(result))
            print()


if __name__ == "__main__":
    main()
