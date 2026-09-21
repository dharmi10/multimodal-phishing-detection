"""
CLI entry point for the combined smishing pipeline.

    python run_pipeline.py "VM-HDFCBK: Your account is suspended. Verify at http://hdfc-secure.tk/login"
    python run_pipeline.py --sender VM-HDFCBK "Your account is suspended. Verify at hdfc-secure.tk/login"
    python run_pipeline.py --image screenshot.png "Check this"
    python run_pipeline.py --image qr.png ""          # image-only message
    python run_pipeline.py --demo
    python run_pipeline.py --json "..."

--image may be repeated. An image is decoded by layer 4, whose QR links go to
layer 2 and whose OCR text goes to layer 1 - layer 4 has no vote of its own.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pipeline.message import Message
from src.pipeline.pipeline import analyze_message, format_report

DEMO_MESSAGES = [
    "VM-HDFCBK: Your account has been suspended. Verify immediately at http://hdfc-secure-verify.tk/login",
    "AD-AMAZON: Your order #402-889 has been shipped. Track it at https://www.amazon.com/orders",
    "Hey, are we still on for dinner at 8?",
    "Congratulations! You have won a $500 gift card. Claim now: bit.ly/3xKq9Zz",
]


def load_images(paths: Optional[List[str]]) -> List[bytes]:
    """
    Read each --image path into bytes.

    A path that cannot be read is a user error worth stopping for, not something
    to silently analyse without: a missing attachment would quietly turn an
    image-only message into an empty one.
    """
    images = []
    for raw_path in paths or []:
        path = Path(raw_path)
        try:
            images.append(path.read_bytes())
        except OSError as exc:
            raise SystemExit(f"could not read image {path}: {type(exc).__name__}: {exc}")
    return images


def main():
    parser = argparse.ArgumentParser(description="Run an SMS through the smishing pipeline.")
    parser.add_argument("message", nargs="?", help="the incoming SMS text")
    parser.add_argument("--sender", help="sender ID, when it arrives as metadata rather than in the body")
    parser.add_argument("--image", action="append", metavar="PATH",
                        help="attached image to decode with layer 4; repeatable")
    parser.add_argument("--json", action="store_true", help="print the raw result dict as JSON")
    parser.add_argument("--demo", action="store_true", help="run the built-in sample messages")
    args = parser.parse_args()

    images = load_images(args.image)

    if args.demo:
        messages = [Message(sender=None, text=m) for m in DEMO_MESSAGES]
    elif args.message is not None and (args.message or images):
        messages = [Message(sender=args.sender, text=args.message, images=images)]
    elif images:
        # --image with no message argument at all: an image-only message.
        messages = [Message(sender=args.sender, text="", images=images)]
    else:
        parser.error("give a message, an --image, or use --demo")

    for message in messages:
        result = analyze_message(message)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_report(result))
            print()


if __name__ == "__main__":
    main()
