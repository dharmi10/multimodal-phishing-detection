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

from features import tokenize_url, extract_lexical_features, check_website_html

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


def _get_html_red_flag_score(url: str) -> float:
    html_features = check_website_html(url)
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


def predict_url_score(url: str) -> float:
    check_url = url if url.startswith(('http://', 'https://')) else 'http://' + url
    domain = urlparse(check_url).netloc.replace('www.', '')

    if domain in TRUSTED_DOMAINS:
        return 0.0

    text_lexical_score = _get_text_lexical_score(url)
    html_red_flag_score = _get_html_red_flag_score(url)

    # Weighted blend: text/lexical model carries most of the weight,
    # HTML red flags nudge the score up when present
    final_score = (0.7 * text_lexical_score) + (0.3 * html_red_flag_score)

    return float(min(1.0, final_score))