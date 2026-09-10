"""
End-to-end smishing pipeline.

Flow (current scope):
    incoming SMS
        -> stage 0: extract sender_id + url(s) + text
        -> layer 1: SMS text model      -> smish_probability
        -> layer 2: URL model           -> url_phish_score
        -> layer 3: sender reputation   -> sender_phish_score
        -> fusion:  combine 3 scores    -> final verdict + explanation
"""
from . import extract as extraction
from .layer1_sms import predict_sms
from .layer2_url import predict_urls
from .layer3_sender import lookup_sender
from .fusion import fuse


def analyze_sms(raw_sms: str, sender_id: str = None) -> dict:
    """Run one incoming SMS through the layers that are built so far."""
    parts = extraction.extract(raw_sms, sender_id=sender_id)

    layer1 = predict_sms(parts["text"])
    layer2 = predict_urls(parts["urls"])
    layer3 = lookup_sender(parts["sender_id"])
    fusion = fuse(layer1, layer2, layer3)

    return {
        "input": {
            "raw": parts["raw"],
            "sender_id": parts["sender_id"],
            "text": parts["text"],
            "urls": parts["urls"],
        },
        "layer1_sms": layer1,
        "layer2_url": layer2,
        "layer3_sender": layer3,
        "fusion": fusion,
    }


def format_report(result: dict) -> str:
    """Human-readable summary of one analysis, for CLI output."""
    inp = result["input"]
    l1 = result["layer1_sms"]
    l2 = result["layer2_url"]
    l3 = result["layer3_sender"]
    fz = result["fusion"]

    lines = []
    lines.append("=" * 62)
    lines.append("INCOMING SMS")
    lines.append("=" * 62)
    lines.append(f"  Sender ID : {inp['sender_id'] or '(not provided)'}")
    lines.append(f"  Text      : {inp['text']}")
    lines.append(f"  URLs      : {', '.join(inp['urls']) if inp['urls'] else '(none found)'}")

    lines.append("")
    lines.append("-" * 62)
    lines.append("LAYER 1 - SMS TEXT MODEL (TF-IDF + SVM v2)")
    lines.append("-" * 62)
    if l1["smish_probability"] is None:
        lines.append(f"  {l1['note']}")
    else:
        lines.append(f"  Prediction        : {l1['prediction'].upper()}")
        lines.append(f"  P(smish)          : {l1['smish_probability']:.4f}")
        lines.append(f"  Confidence        : {l1['confidence']:.2%}")
        lines.append(f"  Cleaned text      : {l1['cleaned_text']}")

    lines.append("")
    lines.append("-" * 62)
    lines.append("LAYER 2 - URL MODEL (stage 2)")
    lines.append("-" * 62)
    if not l2["results"]:
        lines.append("  No URL in message - layer 2 skipped.")
    else:
        for r in l2["results"]:
            if r["error"]:
                lines.append(f"  {r['url']}  ->  ERROR: {r['error']}")
            else:
                lines.append(f"  {r['url']}  ->  phish score {r['phish_score']:.4f}")
        if l2["url_phish_score"] is not None:
            lines.append(f"  Aggregate (max)   : {l2['url_phish_score']:.4f}  [{l2['worst_url']}]")

    lines.append("")
    lines.append("-" * 62)
    lines.append("LAYER 3 - SENDER REPUTATION (custom dataset lookup)")
    lines.append("-" * 62)
    lines.append(f"  Status            : {l3['status'].upper()}")
    if l3["sender_phish_score"] is None:
        lines.append(f"  {l3['note']}")
    else:
        lines.append(f"  Sender risk       : {l3['sender_phish_score']:.4f}")

    lines.append("")
    lines.append("=" * 62)
    lines.append("FINAL VERDICT")
    lines.append("=" * 62)
    banner = "PHISHING" if fz["verdict"] == "phishing" else "LEGITIMATE"
    lines.append(f"  {banner}   score {fz['score']:.4f}  (threshold {fz['threshold']})")
    lines.append(f"  Confidence : {fz['confidence'].upper()} - {fz['layers_used']} of 3 layers contributed")
    lines.append("")
    lines.append("  Why:")
    for reason in fz["reasons"]:
        lines.append(f"    - {reason}")
    lines.append("=" * 62)

    return "\n".join(lines)
