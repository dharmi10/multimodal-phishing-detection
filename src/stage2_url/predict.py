import sys
import joblib
from pathlib import Path
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix

# Resolve everything relative to THIS file so stage 2 can be imported from the
# project root (by the combined pipeline) as well as run directly from inside
# this folder. The trained artifacts live in saved_models/ with every other
# model in the project; rebuild them with scripts/12_train_stage2_url.py.
# Model/logic below is unchanged.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from features import (
    tokenize_url,
    extract_lexical_features,
    check_website_html,
    analyze_html,
    fetch_resource,
)

_MODELS = _HERE.parents[1] / "saved_models"

model = joblib.load(_MODELS / "stage2_url_model.pkl")
cv = joblib.load(_MODELS / "stage2_feature_transformer.pkl")

lexical_cols = ['has_ip_address', 'slash_count', 'has_port_number', 'has_at_symbol', 'hyphen_count', 'url_length', 'domain_length']

TRUSTED_DOMAINS = {
    'paypal.com', 'google.com', 'github.com', 'wikipedia.org',
    'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com',
    'netflix.com', 'linkedin.com'
}

def _get_text_lexical_score(url: str) -> float:
    tokens = tokenize_url(url)
    text_joint = ' '.join(tokens)
    X_vec = cv.transform([text_joint])

    lex_feats = extract_lexical_features(url)
    lex_values = [lex_feats[col] for col in lexical_cols]
    X_lex_sparse = csr_matrix([lex_values], dtype=float)

    X_combined = hstack([X_vec, X_lex_sparse])

    proba = model.predict_proba(X_combined)[0]
    malicious_index = list(model.classes_).index('bad')
    return float(proba[malicious_index])


def _red_flags_from_html(html_features) -> float:
    if html_features is None:
        # Page couldn't be fetched (down, blocked, timeout) — treat as neutral, not a red flag
        return 0.0

    red_flags = sum([
        html_features['external_link_ratio'] > 0.5,
        html_features['has_hidden_iframe'],
        html_features['has_suspicious_form'],
        html_features['blocks_right_click'],
    ])
    return red_flags / 4.0  # normalize 0 to 1


def _get_html_red_flag_score(url: str) -> float:
    """Kept for callers that only want the red-flag half, fetch included."""
    return _red_flags_from_html(check_website_html(url))


def predict_url_detail(url: str) -> dict:
    """
    Score one URL, and say what the response actually was.

    Branches on the Content-Type header that the fetch was already receiving
    and discarding:

      text/html    -> the original path, unchanged: 0.7 * text/lexical model
                      + 0.3 * HTML red flags.
      image/*      -> ABSTAIN. This model's red-flag half parses markup; run it
                      over a JPEG and the four checks all come back false, which
                      reads as a clean bill of health for a page that was never
                      examined. The bytes are handed back for layer 4 to decode
                      instead, and this URL contributes no score at all.
      anything else-> unchanged from before: the body is handed to the HTML
                      parser as it always was.

    Returns a dict rather than a float because an abstention is not a score and
    must not be flattened into one. phish_score is None exactly when this URL
    abstained.
    """
    check_url = url if url.startswith(('http://', 'https://')) else 'http://' + url
    domain = urlparse(check_url).netloc.replace('www.', '')

    detail = {
        'url': url,
        'phish_score': None,
        'abstained': False,
        'abstain_reason': None,
        'content_type': None,
        'image_bytes': None,
    }

    # Unchanged: a trusted domain short-circuits to 0.0 without being fetched.
    if domain in TRUSTED_DOMAINS:
        detail['phish_score'] = 0.0
        return detail

    text_lexical_score = _get_text_lexical_score(url)

    resource = fetch_resource(url)
    if resource is None:
        # Unreachable. Same as before: neutral red flags, not a penalty.
        detail['phish_score'] = float(min(1.0, 0.7 * text_lexical_score))
        return detail

    detail['content_type'] = resource['content_type']

    if resource['is_image']:
        detail['abstained'] = True
        detail['image_bytes'] = resource['content']
        detail['abstain_reason'] = (
            f"response is {resource['content_type']}, not a web page - "
            "handed to layer 4 for decoding"
        )
        return detail

    html_red_flag_score = _red_flags_from_html(analyze_html(resource['text'], url))

    # Weighted blend: text/lexical model carries most of the weight,
    # HTML red flags nudge the score up when present
    final_score = (0.7 * text_lexical_score) + (0.3 * html_red_flag_score)

    detail['phish_score'] = float(min(1.0, final_score))
    return detail


def predict_url_score(url: str):
    """
    The original single-number API.

    Now returns None when the URL abstained (the response was an image), because
    there is no honest float for "this was not a web page". Callers that need the
    reason or the image bytes should use predict_url_detail.
    """
    return predict_url_detail(url)['phish_score']