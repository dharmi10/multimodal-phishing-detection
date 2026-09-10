"""
Streamlit demo app for panel presentation.

Runs the full four-stage pipeline, not just the text model:
  stage 0  extract sender ID + URLs from the incoming SMS
  layer 1  TF-IDF + SVM (v3) on the message text
  layer 2  stage 2 URL model on each extracted link
  layer 3  sender ID lookup against the custom dataset
  fusion   logistic regression over the three scores -> verdict + reasons

Uses st.form to wrap the input + submit button. Without a form, a bare
st.button can trigger the widget to visually reset/show its placeholder
right after rerun (a known Streamlit timing quirk) even though the
prediction itself is computed correctly from the submitted text. A form
only reruns once on submit, which avoids this entirely.

Note layer 2 fetches each URL over the network (5s timeout), so analysis of
a message containing links takes a few seconds - hence the spinner.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from src.pipeline.pipeline import analyze_sms

st.set_page_config(page_title="Smishing Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ Smishing Detection System")
st.caption(
    "Four-stage pipeline: SMS text model → URL model → sender ID lookup → fusion. "
    "Reproduction of: Enhancing Mobile Security through Robust Smishing Detection "
    "(Piyumal & Perera, 2026)"
)

EXAMPLES = {
    "— pick an example —": ("", ""),
    "Phishing: fake bank alert": (
        "VK-YES8NK",
        "Dear User, INR 499 debited from your account. If not done by you, "
        "verify at http://secure-yesbank-verify.click/account",
    ),
    "Phishing: prize bait, numeric sender": (
        "741247",
        "Congratulations! You have won a free flight voucher. Claim now: http://rebrand.ly/mjQ1Uqy",
    ),
    "Legitimate: booking confirmation": (
        "VK-OYOROO",
        "Booking confirmed! Your OYO Rooms hotel stay on 25 Sep. Details https://www.oyorooms.com/booking",
    ),
    "Legitimate: bank debit alert": (
        "JM-ICICIB",
        "Rs.2,340 debited from A/c XX8891 on 04-09-26. Not you? Call 18001080.",
    ),
}


@st.cache_resource
def warm_up():
    """Load every model once so the first real submit isn't slow."""
    analyze_sms("warm up message", sender_id="AD-AMAZON")
    return True


def score_bar(label: str, score, caption: str):
    """One layer's contribution, or why it had nothing to say."""
    if score is None:
        st.markdown(f"**{label}**")
        st.caption(f"— {caption}")
        return
    st.markdown(f"**{label}** &nbsp; `{score:.3f}`", unsafe_allow_html=True)
    st.progress(min(max(float(score), 0.0), 1.0))
    st.caption(caption)


try:
    warm_up()
except FileNotFoundError as exc:
    st.error(f"{exc}\n\nRun `python scripts/11_train_fusion.py` to build the fusion model.")
    st.stop()

choice = st.selectbox("Try an example, or write your own below:", list(EXAMPLES))
example_sender, example_text = EXAMPLES[choice]

with st.form("sms_form"):
    sender_id = st.text_input(
        "Sender ID (optional)",
        value=example_sender,
        placeholder="e.g. VM-HDFCBK — on a real handset this arrives as carrier metadata",
    )
    message = st.text_area(
        "Incoming SMS:",
        value=example_text,
        height=110,
        placeholder="e.g. Your account has been suspended. Verify at http://hdfc-secure.tk/login",
    )
    submitted = st.form_submit_button("Analyze Message", type="primary")

if submitted:
    if not message.strip():
        st.warning("Please enter a message.")
        st.stop()

    with st.spinner("Running all layers (layer 2 fetches each link, this can take a few seconds)…"):
        result = analyze_sms(message, sender_id=sender_id.strip() or None)

    inp, l1, l2, l3, fz = (
        result["input"], result["layer1_sms"], result["layer2_url"],
        result["layer3_sender"], result["fusion"],
    )

    st.divider()

    # --- Final verdict ---
    banner = f"**{fz['verdict'].upper()}** — score {fz['score']:.3f} (threshold {fz['threshold']})"
    if fz["verdict"] == "phishing":
        st.error(f"🚨 {banner}")
    else:
        st.success(f"✅ {banner}")

    confidence = fz["confidence"].upper()
    st.caption(f"Confidence: **{confidence}** — {fz['layers_used']} of 3 layers contributed")
    if fz["layers_used"] < 3:
        st.info(
            "Fewer than three layers had anything to say, so this verdict rests on less "
            "evidence than usual. A missing layer is treated as *no information*, not as *safe*."
        )

    # --- Why ---
    st.subheader("Why")
    for reason in fz["reasons"]:
        st.markdown(f"- {reason}")

    # --- What was extracted ---
    st.subheader("Extracted from the message")
    left, right = st.columns(2)
    left.metric("Sender ID", inp["sender_id"] or "—")
    right.metric("Links found", len(inp["urls"]))
    if inp["urls"]:
        for url in inp["urls"]:
            st.code(url, language=None)

    # --- Per-layer detail ---
    st.subheader("Layer scores")

    score_bar(
        "Layer 1 · SMS text model",
        l1["smish_probability"],
        f"predicted {l1['prediction']}" if l1["smish_probability"] is not None
        else (l1["note"] or "no text signal"),
    )
    score_bar(
        "Layer 2 · URL model",
        l2["url_phish_score"],
        f"worst of {len(l2['results'])} link(s): {l2['worst_url']}" if l2["url_phish_score"] is not None
        else "no link in the message",
    )
    score_bar(
        "Layer 3 · Sender ID lookup",
        l3["sender_phish_score"],
        f"listed as {l3['status']} in the custom dataset" if l3["sender_phish_score"] is not None
        else (l3["note"] or "no sender information"),
    )

    if l2["results"]:
        with st.expander("Per-link detail"):
            for r in l2["results"]:
                if r["error"]:
                    st.write(f"`{r['url']}` — error: {r['error']}")
                else:
                    st.write(f"`{r['url']}` — {r['phish_score']:.3f}")

    with st.expander("See cleaned text (what layer 1 actually sees)"):
        st.code(l1["cleaned_text"] or "(empty after cleaning)", language=None)

st.divider()
st.caption(
    "Layer 1: TF-IDF + SVM v3 (adds transactional SMS to the legitimate class). "
    "Layer 2: stage 2 URL model. Layer 3: sender lookup. "
    "Fusion: logistic regression, 5-fold CV accuracy 0.9815 / recall 1.000 on the custom dataset."
)
