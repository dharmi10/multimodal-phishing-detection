"""
End-to-end smishing pipeline.

Flow:
    incoming Message (text + attached images + sender)
        -> stage 0: extract sender_id + url(s) + text
        -> layer 4: decode images        -> QR urls, OCR text, image_score
        -> layer 2: URL model            -> url_phish_score
        -> layer 1: SMS text model       -> smish_probability
        -> layer 3: sender reputation    -> sender_phish_score
        -> fusion:  combine 3 scores     -> final verdict + explanation

Layer 4 is a decoder, not a fourth opinion. What it finds is routed into the
layers that already know how to score it:

    QR payloads that are URLs  -> layer 2, appended to the message's own links
    URLs written in the image  -> layer 2, found by running stage 0's own URL
                                  extractor over the OCR text
    OCR text                   -> layer 1, combined with the body text by max()
    The image's top OCR lines  -> layer 3, when no sender arrived any other way

That second route matters more than the QR one. Text-as-image is the common
image-phishing technique - a screenshot of a fake bank SMS - and its link is
typed in the picture, not encoded in a QR code. Without it, an image reading
"Verify now: meribank-kyc.co.in/verify" scores 1.00 at layer 1 while layer 2
reports no URL at all.

max() rather than a separate fusion feature, for the same reason layer 2 takes
the max across links: the two numbers are outputs of the SAME SVM, so giving
them separate features would split one coefficient across two copies of one
signal. Whichever text is more alarming is the one that matters.

The layer 3 route was the last one missing, and its absence was visible: a
screenshot of a real SMS had its text scored and its link scored while layer 3
reported "no sender ID was supplied" - with the header sitting in plain sight at
the top of the picture. It is read off the image only when nothing better is
available; an explicit sender always wins, because carrier metadata is not a
guess and an OCR read of a screenshot is.

image_score never reaches the fusion layer at all - see layer4_image.py.
"""
import textwrap
from typing import Dict, List, Optional

from . import extract as extraction
from .layer1_sms import predict_sms
from .layer2_url import predict_urls, merge_results
from .layer3_sender import STATUS_MISSING, STATUS_UNKNOWN, lookup_sender
from .layer4_image import predict_images
from .message import Message
from .fusion import fuse

# Where the sender ID the pipeline actually used came from. Reported because a
# header READ OFF A PICTURE is weaker evidence than one that arrived as carrier
# metadata, and whoever reads the verdict is entitled to know which it was.
SENDER_SOURCE_PROVIDED = "carrier metadata"
SENDER_SOURCE_BODY = "parsed from the message body"
SENDER_SOURCE_IMAGE = "read from the image"


def analyze_message(message: Message) -> dict:
    """
    Run one incoming message through every layer.

    `message.channel` is not read here and must not be read anywhere downstream
    - it is provenance metadata only. See message.py for why.
    """
    parts = extraction.extract(message.text, sender_id=message.sender)

    # --- Layer 4, first pass: the images that arrived with the message -------
    layer4 = predict_images(message.images)

    # --- Layer 2: the message's own links, plus every link the image yielded --
    urls = list(parts["urls"])
    for url in image_urls(layer4):
        if url not in urls:
            urls.append(url)

    layer2 = predict_urls(urls)
    fetched_images = layer2.pop("fetched_images", [])

    # --- Layer 4, second pass: images that layer 2 fetched -------------------
    # A link can serve a picture rather than a page, and that picture can carry
    # its own QR code. Those newly discovered links are scored in a second layer
    # 2 pass and merged in.
    #
    # Exactly ONE extra round. image -> QR -> image -> QR could otherwise chain
    # indefinitely, and each hop is a live HTTP request; a fixed bound is worth
    # more here than completeness against an adversary who can always add one
    # more hop.
    if fetched_images:
        layer4 = predict_images(list(message.images) + list(fetched_images))

        already_scored = {r["url"] for r in layer2["results"]}
        new_urls = [u for u in image_urls(layer4) if u not in already_scored]
        if new_urls:
            second_pass = predict_urls(new_urls)
            second_pass.pop("fetched_images", None)
            layer2 = merge_results(layer2, second_pass)
            layer2.pop("fetched_images", None)

    # --- Layer 1: body text, combined with OCR text when it is usable --------
    layer1 = _score_text(
        body_text=parts["text"],
        usable_ocr=_usable_ocr(layer4),
        ocr_degraded=layer4["ocr_degraded"],
        ocr_empty=layer4.get("ocr_empty", False),
        images_present=bool(message.images or fetched_images),
    )

    # --- Layer 3: the sender, from metadata, the body, or failing both the image
    sender_id, sender_source, sender_from_image = _resolve_sender(
        provided=message.sender,
        parsed=parts["sender_id"],
        layer4=layer4,
    )

    layer3 = lookup_sender(sender_id)
    fusion = fuse(layer1, layer2, layer3)

    return {
        "input": {
            "raw": parts["raw"],
            "sender_id": sender_id,
            "sender_source": sender_source,
            "sender_from_image": sender_from_image,
            "text": parts["text"],
            "urls": parts["urls"],
            "channel": message.channel,  # metadata for the reader; no layer uses it
            "images_attached": len(message.images),
        },
        "layer1_sms": layer1,
        "layer2_url": layer2,
        "layer3_sender": layer3,
        "layer4_image": layer4,
        "fusion": fusion,
    }


def analyze_sms(raw_sms: str, sender_id: str = None) -> dict:
    """
    Analyse a plain text SMS.

    Kept at its original signature so the CLI, the Streamlit app and anything
    else holding a raw string keep working untouched. A message with no images
    produces the same layer 1/2/3 scores and the same verdict it always did; the
    only addition is a layer4_image section reporting that there was no image.
    """
    return analyze_message(Message.from_sms(raw_sms, sender_id=sender_id))


def _resolve_sender(provided: Optional[str], parsed: Optional[str],
                    layer4: Dict) -> tuple:
    """
    Decide which sender ID layer 3 is given, and where it came from.

    Returns (sender_id, source, from_image), where `from_image` is the winning
    candidate dict when the image supplied it and None otherwise.

    Precedence - strongest evidence first, and the image is last for a reason:

      1. an explicitly supplied sender    - carrier metadata, not a guess
      2. a header parsed off the body     - someone typed it as part of the text
      3. a header read off the image      - OCR of a screenshot, which can
                                            misread a character, and a misread
                                            character is exactly what the
                                            impersonation rule looks for

    The image is consulted ONLY when the first two produced nothing, so no
    existing input path changes behaviour: a text-only message, or one with a
    sender in the form field, resolves exactly as it did before.
    """
    if provided and str(provided).strip():
        return str(provided).strip(), SENDER_SOURCE_PROVIDED, None

    if parsed:
        return parsed, SENDER_SOURCE_BODY, None

    candidate = _sender_from_images(layer4)
    if candidate is not None:
        return candidate["sender_id"], SENDER_SOURCE_IMAGE, candidate

    return None, None, None


def _sender_from_images(layer4: Dict) -> Optional[Dict]:
    """
    The best sender header visible at the top of any attached image, or None.

    Works per image, off `layer4["results"]`, rather than off the aggregated
    ocr_lines: these rules are POSITIONAL - "the header is near the top" - and
    the top of the second image sits in the middle of the flattened list, where
    that test means nothing.

    A degraded or empty image contributes no candidate, for the same reason it
    contributes no text and no URLs. A header is six characters with no
    redundancy in it; if OCR could not hold a whole message together, it cannot
    be trusted to have held those six.

    Choosing between candidates:

      1. shape tier       - how hard the shape is to produce by accident
      2. known to layer 3 - within one tier, prefer a candidate the register
                            actually recognises (registered, a near-miss of a
                            brand, or an impossible shape) over one it has
                            nothing to say about. Only used as a TIE-BREAK, not
                            as the primary key: promoting a recognised
                            candidate over a better-shaped one would let a
                            stoplist miss that happens to be a real header beat
                            the actual sender sitting on the line below it.
      3. position         - earlier image, then higher line
    """
    candidates: List[Dict] = []
    for position, result in enumerate(layer4.get("results") or [], 1):
        if result.get("ocr_degraded") or result.get("ocr_empty"):
            continue
        for candidate in extraction.sender_candidates_from_lines(
                result.get("ocr_lines") or []):
            candidates.append(dict(candidate, image=position))

    if not candidates:
        return None

    def rank(candidate: Dict) -> tuple:
        status = lookup_sender(candidate["sender_id"])["status"]
        unrecognised = status in (STATUS_UNKNOWN, STATUS_MISSING)
        return (
            extraction.SENDER_TIER_ORDER.index(candidate["tier"]),
            1 if unrecognised else 0,
            candidate["image"],
            candidate["line_index"],
        )

    best = min(candidates, key=rank)
    best["alternatives"] = [
        c["sender_id"] for c in sorted(candidates, key=rank)
        if c["sender_id"] != best["sender_id"]
    ]
    return best


def _usable_ocr(layer4: Dict) -> str:
    """
    The OCR text that passed the degradation check, or "" when it did not.

    One definition, used for BOTH what layer 1 scores and what URLs are pulled
    out for layer 2. If these two ever diverged, a link could be extracted from
    text the pipeline had already decided was too damaged to read - scoring a
    URL that OCR may well have misspelled.
    """
    text = layer4.get("ocr_text") or ""
    unusable = layer4.get("ocr_degraded") or layer4.get("ocr_empty")
    return text.strip() if (text and not unusable) else ""


def image_urls(layer4: Dict) -> list:
    """
    Every link an image yielded: QR payloads, plus URLs written in the OCR text.

    The OCR text goes through stage 0's extractor, the same one used on the
    message body, so a link typed in a screenshot is recognised by exactly the
    rules that recognise one typed in an SMS - including the schemeless
    "meribank-kyc.co.in/verify" form, which is the usual way they appear.

    Degraded OCR contributes no URLs, for the same reason it contributes no
    text: characters that survived noise badly enough to fail the token floor
    are not characters to build a domain name out of.
    """
    urls = list(layer4.get("qr_urls") or [])
    for url in extraction.extract_urls(_usable_ocr(layer4)):
        if url not in urls:
            urls.append(url)
    return urls


def _score_text(body_text: str, usable_ocr: str, ocr_degraded: bool,
                ocr_empty: bool, images_present: bool) -> Dict:
    """
    Layer 1 over the body text and, when it is trustworthy, the OCR text.

    `ocr_empty` and `ocr_degraded` are handled IDENTICALLY here - in both cases
    there is no usable OCR text and the rules below apply unchanged. They are
    separate only so the note explains which happened, because "this image has
    no text in it" and "this image's text was destroyed" read very differently
    to a person even though the pipeline does the same thing with both.

    Degradation policy (the OCR text is below the token floor):

      body text present -> the OCR text is ignored completely. The body is the
                           reliable signal and there is nothing to gain from
                           mixing partial noise into it.
      no body text      -> layer 1 ABSTAINS. It must not return a low score.
                           A damaged image whose phishing keywords were lost to
                           noise would otherwise be scored as calm, ordinary
                           text, which converts image damage into a false
                           negative. Abstaining hands the decision to the
                           fusion layer's presence-flag and mean-imputation
                           logic, which was built for exactly this case.

    With no images in play at all this returns predict_sms(body_text) unchanged,
    so a text-only message is scored exactly as it was before layer 4 existed.
    """
    body = predict_sms(body_text)

    if not images_present:
        return body

    if not usable_ocr:
        result = dict(body)
        result["source"] = "body"
        result["body_smish_probability"] = body["smish_probability"]
        result["ocr_smish_probability"] = None
        result["ocr_used"] = False

        scored_on_body = body["smish_probability"] is not None

        if ocr_degraded:
            if scored_on_body:
                reason = ("OCR text was too degraded to trust and was discarded; "
                          "scored on the message body instead")
            else:
                reason = ("OCR text was too degraded to trust and the message has no "
                          "body text, so layer 1 abstains rather than return a "
                          "low score for text it could not read")
        elif ocr_empty:
            # Neutral wording: nothing went wrong, the picture simply has no text.
            if scored_on_body:
                reason = "no text found in the image; scored on the message body"
            else:
                reason = ("no text found in the image and the message has no body "
                          "text, so layer 1 has nothing to score")
        else:
            reason = None

        if reason:
            result["note"] = f"{result['note']}; {reason}" if result["note"] else reason

        return result

    ocr = predict_sms(usable_ocr)

    body_prob = body["smish_probability"]
    ocr_prob = ocr["smish_probability"]

    if body_prob is None and ocr_prob is None:
        base, source = body, "body"
    elif body_prob is None:
        base, source = ocr, "ocr"
    elif ocr_prob is None:
        base, source = body, "body"
    else:
        # max(): whichever reading of the message is more alarming wins.
        base, source = (ocr, "ocr") if ocr_prob > body_prob else (body, "body")
        source = f"{source} (max of body and OCR)"

    result = dict(base)
    result["source"] = source
    result["body_smish_probability"] = body_prob
    result["ocr_smish_probability"] = ocr_prob
    result["ocr_used"] = True
    return result


def format_report(result: dict) -> str:
    """Human-readable summary of one analysis, for CLI output."""
    inp = result["input"]
    l1 = result["layer1_sms"]
    l2 = result["layer2_url"]
    l3 = result["layer3_sender"]
    l4 = result.get("layer4_image")
    fz = result["fusion"]

    lines = []
    lines.append("=" * 62)
    lines.append("INCOMING SMS")
    lines.append("=" * 62)
    lines.append(f"  Sender ID : {inp['sender_id'] or '(not provided)'}")
    if inp.get("sender_source"):
        origin = inp["sender_source"]
        from_image = inp.get("sender_from_image")
        if from_image:
            origin += (f" - line {from_image['line_index']} of image "
                       f"{from_image['image']}, matched as: {from_image['tier']}")
        lines.append(f"  Sender via: {origin}")
    lines.append(f"  Text      : {inp['text']}")
    lines.append(f"  URLs      : {', '.join(inp['urls']) if inp['urls'] else '(none found)'}")
    if inp.get("images_attached"):
        lines.append(f"  Images    : {inp['images_attached']} attached")

    lines.append("")
    lines.append("-" * 62)
    lines.append("LAYER 1 - SMS TEXT MODEL (TF-IDF + SVM v3)")
    lines.append("-" * 62)
    if l1["smish_probability"] is None:
        lines.append(f"  {l1['note']}")
    else:
        lines.append(f"  Prediction        : {l1['prediction'].upper()}")
        lines.append(f"  P(smish)          : {l1['smish_probability']:.4f}")
        lines.append(f"  Confidence        : {l1['confidence']:.2%}")
        lines.append(f"  Cleaned text      : {l1['cleaned_text']}")
        if l1.get("ocr_used"):
            body_prob = l1.get("body_smish_probability")
            ocr_prob = l1.get("ocr_smish_probability")
            lines.append(
                f"  Scored from       : {l1['source']}"
                f"   (body {_fmt(body_prob)}, OCR {_fmt(ocr_prob)})"
            )
        if l1.get("note") and l1["smish_probability"] is not None:
            lines.append(f"  Note              : {l1['note']}")

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
            elif r.get("abstained"):
                lines.append(f"  {r['url']}  ->  ABSTAINED: {r.get('note')}")
            else:
                lines.append(f"  {r['url']}  ->  phish score {r['phish_score']:.4f}")
        if l2["url_phish_score"] is not None:
            lines.append(f"  Aggregate (max)   : {l2['url_phish_score']:.4f}  [{l2['worst_url']}]")

    lines.append("")
    lines.append("-" * 62)
    lines.append("LAYER 3 - SENDER ID (TRAI header register + impersonation rules)")
    lines.append("-" * 62)
    lines.append(f"  Sender as received: {l3.get('sender_id') or '(not provided)'}")
    if l3.get("header"):
        wrapping = []
        if l3.get("prefix"):
            wrapping.append(f"prefix {l3['prefix']}")
        if l3.get("suffix"):
            wrapping.append(f"suffix {l3['suffix']}")
        stripped = f"   (stripped {', '.join(wrapping)})" if wrapping else ""
        lines.append(f"  Normalised header : {l3['header']}{stripped}")
    lines.append(f"  Status            : {l3['status'].upper()}")
    if l3.get("entity_name"):
        lines.append(f"  Registered to     : {l3['entity_name']}")
    if l3.get("impersonation"):
        lines.append(f"  Resembles         : {l3['resembles']} "
                     f"({l3.get('resembles_entity') or 'unnamed entity'}) "
                     f"- one {l3.get('impersonation_edit')}")
        tier = (l3.get("impersonation_tier") or "").upper()
        overrides = "overrides the verdict" if tier == "HIGH" else "reported only, does not override"
        lines.append(f"  Impersonation risk: {tier} ({overrides})")
    if l3["sender_phish_score"] is None:
        lines.append(f"  {l3['note']}")
    else:
        lines.append(f"  Sender risk       : {l3['sender_phish_score']:.4f}")

    if l4 is not None:
        lines.append("")
        lines.append("-" * 62)
        lines.append("LAYER 4 - IMAGE DECODER (not a fusion input)")
        lines.append("-" * 62)
        if not l4["images_scored"]:
            lines.append("  No image in message - layer 4 skipped.")
        else:
            score = l4["image_score"]
            lines.append(f"  Images decoded    : {l4['images_scored']}")
            lines.append(
                f"  Image-only score  : {score:.4f}" if score is not None
                else "  Image-only score  : (none - no image could be decoded)"
            )
            if l4["qr_urls"]:
                lines.append(f"  QR links          : {', '.join(l4['qr_urls'])}  -> sent to layer 2")
            ocr_links = [u for u in image_urls(l4) if u not in l4["qr_urls"]]
            if ocr_links:
                lines.append(f"  Links in OCR text : {', '.join(ocr_links)}  -> sent to layer 2")
            if l4["ocr_degraded"]:
                lines.append("  OCR               : DEGRADED - text discarded, not scored")
            elif l4.get("ocr_empty"):
                lines.append("  OCR               : no text found in this image")
            elif l4["ocr_text"]:
                lines.append(f"  OCR text          : {l4['ocr_text'][:200]}  -> sent to layer 1")
            for note in l4["notes"]:
                lines.append(f"    - {note}")

    lines.append("")
    lines.append("=" * 62)
    lines.append("FINAL VERDICT")
    lines.append("=" * 62)
    # The label, not the raw verdict: with three layers behind it they are the
    # same string, so a normal message prints exactly what it always did.
    lines.append(f"  {fz['label']}   score {fz['score']:.4f}  (threshold {fz['threshold']})")
    if fz.get("label_reason"):
        lines.append(f"  {fz['label_reason']}")
    if fz.get("sender_override"):
        # The model's own call stays visible. The rule replaced the label, not
        # the evidence, and hiding the disagreement would misrepresent both.
        lines.append(f"  (the fusion model on its own said: {fz['model_label']})")
    lines.append(f"  Confidence : {fz['confidence'].upper()} - {fz['layers_used']} of 3 layers contributed")

    lines.extend(_explanation_lines(fz))
    lines.append("=" * 62)

    return "\n".join(lines)


def _explanation_lines(fusion: dict) -> list:
    """
    The verdict's structured account, as fixed-width text.

    Three parts, in the order a reader needs them: the paragraph, the arithmetic
    that produced the score, and the rules that were checked beside the model.
    Falls back to the flat `reasons` list if `explanation` is somehow absent, so
    an older cached result still prints something.
    """
    explanation = fusion.get("explanation")
    if not explanation:
        out = ["", "  Why:"]
        out.extend(f"    - {reason}" for reason in fusion.get("reasons", []))
        return out

    lines = ["", "  WHY THIS VERDICT", "  " + "-" * 60]
    lines.extend(_wrap(explanation["summary"], width=58, indent="  "))

    lines.append("")
    lines.append("  HOW THE SCORE WAS BUILT   (log-odds; they sum exactly)")
    lines.append("  " + "-" * 60)
    baseline = explanation["baseline"]
    lines.append(f"    {'baseline, every layer silent':<34}"
                 f"{baseline['logit']:+8.3f}   score {baseline['score']:.4f}")

    for entry in explanation["layers"]:
        if not entry["contributed"]:
            lines.append(f"    {entry['name']:<34}{'   --   ':>8}   "
                         f"did not contribute")
            continue

        measured = ("" if entry["score"] is None
                    else f"measured {entry['score']:.3f}")
        lines.append(f"    {entry['name']:<34}{entry['contribution']:+8.3f}   "
                     f"{measured}")
        lines.extend(_wrap(
            f"{entry['direction']}, {entry['share']:.0%} of all movement; "
            f"without this layer the score is {entry['score_without']:.4f}"
            + ("  <- DECISIVE: removing it flips the verdict"
               if entry["decisive"] else ""),
            width=54, indent="      ", hanging="      "))

    arithmetic = explanation["arithmetic"]
    lines.append("    " + "-" * 56)
    lines.append(f"    {'final':<34}{arithmetic['logit']:+8.3f}   "
                 f"score {arithmetic['score']:.4f}")
    lines.extend(_wrap(explanation["decision"], width=54, indent="    ",
                       hanging="    "))

    lines.append("")
    lines.append("  WHAT EACH LAYER FOUND")
    lines.append("  " + "-" * 60)
    for entry in explanation["layers"]:
        headline = entry["finding"] or entry["absence_reason"]
        lines.extend(_wrap(f"{entry['name']}: {headline}", width=56,
                           indent="    ", hanging="      "))
        for detail in entry["details"]:
            lines.extend(_wrap(detail, width=52, indent="        - ",
                               hanging="          "))
        if entry.get("arithmetic_text"):
            lines.extend(_wrap(entry["arithmetic_text"], width=52,
                               indent="        = ", hanging="          "))

    lines.append("")
    lines.append("  RULES CHECKED BESIDE THE MODEL")
    lines.append("  " + "-" * 60)
    for rule in explanation["rules"]:
        marker = "!" if rule["fired"] else " "
        lines.extend(_wrap(f"{marker} {rule['name']}: {rule['effect']}", width=56,
                           indent="    ", hanging="      "))

    lines.append("")
    confidence = explanation["confidence"]
    lines.extend(_wrap(
        f"Confidence {confidence['level'].upper()} - "
        f"{confidence['layers_used']} of {confidence['layers_total']} layers. "
        f"{confidence['basis'][0].upper()}{confidence['basis'][1:]}.", width=58, indent="  "))
    return lines


def _wrap(text: str, width: int, indent: str, hanging: str = None) -> list:
    """
    Soft-wrap one paragraph for the fixed-width report.

    textwrap rather than a hand-rolled loop would pull in no dependency at all,
    so it is used - this helper exists only to apply the report's own indent
    conventions, including a hanging indent for continuation lines.
    """
    continuation = hanging if hanging is not None else indent
    wrapped = textwrap.wrap(text, width=width) or [""]
    return [indent + wrapped[0]] + [continuation + line for line in wrapped[1:]]


def _fmt(value: Optional[float]) -> str:
    """Format an optional probability for the report."""
    return "n/a" if value is None else f"{value:.4f}"
