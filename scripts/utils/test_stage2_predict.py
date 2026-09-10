"""
Smoke test for the layer 2 URL model — scores a handful of known-good and
known-bad URLs. Makes a live HTTP request per URL, so it takes a few seconds.

Goes through src/pipeline/layer2_url.py rather than importing stage 2's
predict.py directly. That wrapper applies the scikit-learn version shim the
pickles need (they were fitted under 1.8.0, this venv runs 1.4.2); importing
predict.py raw raises "NotFittedError: idf vector is not fitted". Using the
wrapper also means this tests the path the pipeline actually runs.

Run from the project root:  python scripts/utils/test_stage2_predict.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.layer2_url import predict_url

test_sites = [
    ("http://192.168.1.1/login-verify-update.php", "expect HIGH (bad)"),
    ("https://www.google.com", "expect LOW (good)"),
    ("https://www.github.com", "expect LOW (good)"),
    ("https://en.wikipedia.org", "expect LOW (good)"),
    ("https://www.paypal.com", "expect LOW (good)"),
]

for url, note in test_sites:
    result = predict_url(url)
    if result["error"]:
        print(f"ERROR   -  {url}   ({result['error']})")
    else:
        print(f"{result['phish_score']:.4f}  -  {url}   ({note})")
