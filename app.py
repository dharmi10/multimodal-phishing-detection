"""
Streamlit demo app for panel presentation.

Runs the full pipeline, not just the text model:
  stage 0  extract sender ID + URLs from the incoming SMS
  layer 1  TF-IDF + SVM (v3) on the message text
  layer 2  stage 2 URL model on each extracted link
  layer 3  sender ID against the TRAI header register + impersonation rules
  layer 4  decode attached images -> QR links to layer 2, OCR text to layer 1
  fusion   logistic regression over the three scores -> verdict + reasons

Layer 4 has no vote in the fusion layer. Its image-only score is displayed
because it is useful to a human reading the result, but the fusion model was
trained on a dataset containing no images and has no coefficient for it.

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
from src.pipeline.message import Message
from src.pipeline.pipeline import analyze_message, analyze_sms, image_urls

st.set_page_config(page_title="Smishing Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ Smishing Detection System")
st.caption(
    "SMS text model → URL model → sender ID register → image decoder → fusion. "
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
    uploaded = st.file_uploader(
        "Attached image(s) — optional",
        type=["png", "jpg", "jpeg", "bmp", "gif", "webp"],
        accept_multiple_files=True,
        help="A screenshot or a QR code. Layer 4 decodes it; the links inside go "
             "to layer 2 and the text inside goes to layer 1.",
    )
    submitted = st.form_submit_button("Analyze Message", type="primary")

if submitted:
    images = [f.getvalue() for f in (uploaded or [])]

    if not message.strip() and not images:
        st.warning("Please enter a message or attach an image.")
        st.stop()

    with st.spinner("Running all layers (layer 2 fetches each link, this can take a few seconds)…"):
        result = analyze_message(Message(
            sender=sender_id.strip() or None,
            text=message,
            images=images,
            channel="mms" if images else "sms",  # recorded only; no layer reads it
        ))

    inp, l1, l2, l3, l4, fz = (
        result["input"], result["layer1_sms"], result["layer2_url"],
        result["layer3_sender"], result["layer4_image"], result["fusion"],
    )

    st.divider()

    # --- Final verdict ---
    # The label rather than the raw verdict: INSUFFICIENT EVIDENCE and UNCERTAIN
    # get their own neutral banner instead of a green tick, because a green tick
    # is read as a clearance. The score is shown in every case.
    banner = f"**{fz['label']}** — score {fz['score']:.3f} (threshold {fz['threshold']})"
    if fz["label"].startswith("PHISHING"):
        # startswith, not ==, so the layer 3 rule labels ("PHISHING - SENDER
        # IMPERSONATION") get the red banner rather than falling through to the
        # neutral one.
        st.error(f"🚨 {banner}")
    elif fz["label"] == "LEGITIMATE":
        st.success(f"✅ {banner}")
    else:
        st.warning(f"❓ {banner}")

    if fz.get("label_reason"):
        st.caption(fz["label_reason"])
    if fz.get("sender_override"):
        st.caption(f"The fusion model on its own said: **{fz['model_label']}**. "
                   "The label above comes from a sender rule, not the model.")

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

    # Every link that actually reached layer 2, whatever its source. Counting
    # only the body links reported "Links found: 0" for a message whose QR code
    # had produced a link and had it scored at 0.603 — the headline number
    # contradicting the layer below it.
    scored_links = [r["url"] for r in l2["results"]]
    body_links = inp["urls"]
    image_links = [u for u in scored_links if u not in body_links]

    left, right = st.columns(2)
    left.metric("Sender ID", inp["sender_id"] or "—")
    right.metric("Links found", len(scored_links))

    if image_links:
        st.caption(
            f"{len(body_links)} in the message body, "
            f"{len(image_links)} decoded from the image — all sent to layer 2."
        )
    for url in scored_links:
        origin = "" if url in body_links else "   (from the image)"
        st.code(f"{url}{origin}", language=None)

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
        "Layer 3 · Sender ID register",
        l3["sender_phish_score"],
        f"registered to {l3['entity_name']}" if l3.get("entity_name")
        else (l3["note"] or "no sender information"),
    )

    if l3.get("sender_id"):
        wrapping = []
        if l3.get("prefix"):
            wrapping.append(f"prefix `{l3['prefix']}`")
        if l3.get("suffix"):
            wrapping.append(f"suffix `{l3['suffix']}`")
        stripped = f", stripped {' and '.join(wrapping)}" if wrapping else ""
        st.caption(
            f"Received as `{l3['sender_id']}` → normalised to `{l3['header']}`{stripped} "
            f"— state **{l3['status'].upper()}**"
        )

    if l3.get("impersonation") and l3.get("impersonation_tier") == "high":
        st.error(
            f"🚨 **Sender impersonation (high)** — `{l3['header']}` is not on the register "
            f"and differs from `{l3['resembles']}`, registered to "
            f"**{l3.get('resembles_entity') or 'an unnamed entity'}**, by one "
            f"{l3.get('impersonation_edit')}. This is a rule, not a model score, and it "
            "overrides the verdict."
        )
    elif l3.get("impersonation"):
        # MEDIUM gets a neutral note, not an alarm. The same single edit is how a
        # brand extends its own header range, so this is context for the reader
        # rather than a finding — and it does not touch the verdict.
        st.info(
            f"ℹ️ Resembles `{l3['resembles']}` "
            f"(**{l3.get('resembles_entity') or 'an unnamed entity'}**) — one "
            f"{l3.get('impersonation_edit')} apart. Registered brands often extend "
            "their own header range this way, so this is noted but does **not** "
            "change the verdict."
        )
    elif l3.get("invalid_header"):
        st.error(
            f"🚨 **Invalid sender header** — `{l3['header']}` is a {len(l3['header'])}-digit "
            "number. No registered sender header has that shape. This is a rule, not a "
            "model score."
        )

    # --- Layer 4: shown like the others, but explicitly not a fusion input ---
    if l4["images_scored"]:
        score_bar(
            "Layer 4 · Image decoder",
            l4["image_score"],
            f"image-only signals across {l4['images_scored']} image(s) — "
            "not a fusion input, see below",
        )

        if l4["ocr_degraded"]:
            st.warning(
                "⚠️ **OCR output was degraded** — too little readable text came out of "
                "the image, so it was discarded rather than scored. "
                + (
                    "Layer 1 fell back to the message body."
                    if l1["smish_probability"] is not None
                    else "There was no body text either, so **layer 1 abstained** rather "
                         "than return a low score for text it could not read. A damaged "
                         "image is not evidence of safety."
                )
            )
        elif l4.get("ocr_empty"):
            # Deliberately a caption, not a warning. Most images contain no text
            # and a QR code on a plain background is the commonest of them —
            # there is nothing wrong here to alarm anyone about, and crying wolf
            # on the ordinary case is what makes the real warning above ignorable.
            st.caption(
                "No text found in this image — nothing for layer 1 to read. "
                + (
                    "The message body was scored instead."
                    if l1["smish_probability"] is not None
                    else "The message has no body text either, so layer 1 has nothing "
                         "to score."
                )
            )

        found_links = image_urls(l4)
        if found_links:
            st.caption("Links found in the image, and sent to layer 2:")
            for url in found_links:
                origin = "QR code" if url in l4["qr_urls"] else "OCR text"
                st.code(f"{url}   ({origin})", language=None)

        if l4["notes"]:
            with st.expander("Image findings"):
                for note in l4["notes"]:
                    st.markdown(f"- {note}")
                st.caption(
                    "These drive the image-only score above. The decoded text and links "
                    "are deliberately excluded from it — they are scored by layers 1 and "
                    "2 instead, so no signal is counted twice."
                )

        if l1.get("ocr_used"):
            st.caption(
                f"Layer 1 scored body {l1.get('body_smish_probability')} and OCR text "
                f"{l1.get('ocr_smish_probability')}, taking the max."
            )

    if l2["results"]:
        with st.expander("Per-link detail"):
            for r in l2["results"]:
                if r["error"]:
                    st.write(f"`{r['url']}` — error: {r['error']}")
                elif r.get("abstained"):
                    st.write(f"`{r['url']}` — abstained: {r.get('note')}")
                else:
                    st.write(f"`{r['url']}` — {r['phish_score']:.3f}")

    with st.expander("See cleaned text (what layer 1 actually sees)"):
        st.code(l1["cleaned_text"] or "(empty after cleaning)", language=None)

st.divider()
st.caption(
    "Layer 1: TF-IDF + SVM v3 (adds transactional SMS to the legitimate class). "
    "Layer 2: stage 2 URL model. Layer 3: TRAI sender-header register (23,237 headers) "
    "plus impersonation and invalid-shape rules. "
    "Layer 4: QR + OCR decoder — feeds layers 1 and 2, and is not a fusion input, "
    "because the fusion training data contains no images. "
    "Fusion: logistic regression, 5-fold CV accuracy 0.9815 / recall 1.000 on the custom dataset."
)
