"""
Precompute the demo samples so the frontend works with no network.

Layer 2 makes live HTTP requests, so the demo cannot depend on them. This
script runs the real pipeline on the sample messages and writes each result to
web/public/demo/samples.json. The frontend reads that file when the API is
unreachable.

    venv/Scripts/python.exe -m api.export_samples

Re-run it after any change to the detection code, or the demo will show stale
scores.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.message import Message  # noqa: E402
from src.pipeline.pipeline import analyze_message  # noqa: E402

OUT_PATH = PROJECT_ROOT / "web" / "public" / "demo" / "samples.json"

# The three samples the analyzer offers. The text and sender are used exactly
# as they will be sent from the UI, so the cached result matches a live call.
SAMPLES = [
    {
        "id": "legit",
        "kind": "legitimate",
        "sender": "VM-HDFCBK",
        "text": "Your HDFC Bank OTP is 724193. Valid for 10 minutes. "
                "Do not share it with anyone.",
    },
    {
        "id": "phish",
        "kind": "phishing",
        "sender": "VM-HDFCBK",
        "text": "Your account has been suspended. Verify immediately at "
                "http://hdfc-secure-verify.tk/login",
    },
    {
        "id": "borderline",
        "kind": "borderline",
        "sender": "BLUDRT-S",
        "text": "Your order with Blue Dart AWB# 36178371235 was delivered. "
                "Rate our service: https://acl.cc/BLUDRT/o78mMZR5",
    },
]


def main():
    exported = []
    for sample in SAMPLES:
        message = Message(sender=sample["sender"], text=sample["text"])
        result = analyze_message(message)
        exported.append({**sample, "result": result})
        print(f"{sample['id']:<11} {result['fusion']['score']:.4f} "
              f"{result['fusion']['label']}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": exported,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
