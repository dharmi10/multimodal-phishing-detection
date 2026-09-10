"""
Stage 0 — extraction.

An incoming SMS is split into the three things the downstream layers need:
  1. sender_id  -> layer 3 (sender reputation lookup, not wired up yet)
  2. urls       -> layer 2 (src/stage2_url phishing model)
  3. text       -> layer 1 (TF-IDF + SVM smishing model)

Sender ID note: on a real handset the sender is transport metadata, not part
of the message body. For demo/eval input we also accept the common textual
forms ("VM-HDFCBK: your otp is ..." / "From: 121\n...") so a single raw string
can be fed in end to end.
"""
import re

# Matches URLs with a scheme (http/https), with a bare www., and schemeless
# domains like "sbi-verify.co.in/login" or "bit.ly/3xYz" — smishing messages
# very often drop the scheme, and the paper's cleaning step strips URLs before
# the text model ever sees them, so this is the only place they are captured.
URL_PATTERN = re.compile(
    r"""(?ix)
    \b(
        (?:https?://|www\.)[^\s<>"'\]\)]+       # explicit scheme or www.
        |
        (?:[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?\.)+ # domain labels
        (?:com|net|org|edu|gov|info|biz|io|co|in|uk|us|ru|cn|tk|ml|ga|cf|gq|xyz|top|link|click|live|online|site|shop|app|me|ly|be|cc|to|vip|icu|store|pro|win|bid|loan|work|fit|help|host|space|website|tech|club|fun|life|world|today|cyou|rest|sbs|buzz)
        (?:/[^\s<>"'\]\)]*)?                    # optional path
    )""",
)

# Trailing punctuation that belongs to the sentence, not the URL.
_TRAILING_JUNK = ".,;:!?)]}'\"" 

# Alphanumeric sender headers used by carriers, e.g. VM-HDFCBK, AD-SBIINB,
# TX-ICICIB, JD-AMAZON. Also plain short codes / phone numbers.
_SENDER = r"""(
        [A-Z]{2}-[A-Z0-9]{3,10}               # VM-HDFCBK style
        | \+?\d[\d\-\s]{1,15}\d               # phone number / short code (121, +919876543210)
        | [A-Z][A-Z0-9]{2,15}                 # bare alphanumeric header
    )"""

# Explicit "From: X" label — the body may then start after a colon, a dash,
# OR a line break, since that is how a labelled header is usually laid out.
LABELLED_SENDER_PATTERN = re.compile(
    r"^\s*from\s*[:\-]\s*" + _SENDER + r"(?:\s*[:\-]\s*|\s+)",
    re.VERBOSE | re.IGNORECASE,
)

# No label, e.g. "VM-HDFCBK: your otp is ...". Here a colon or dash is
# required: accepting a bare line break too would misread an ordinary
# message that merely opens with a shouted word ("URGENT" + newline)
# as having a sender header.
BARE_SENDER_PATTERN = re.compile(
    r"^\s*" + _SENDER + r"\s*[:\-]\s+",
    re.VERBOSE,
)


def extract_urls(text: str) -> list:
    """Return every URL found in the message, de-duplicated, order preserved."""
    if not isinstance(text, str):
        return []

    found = []
    for match in URL_PATTERN.finditer(text):
        url = match.group(1).rstrip(_TRAILING_JUNK)
        if url and url not in found:
            found.append(url)
    return found


def extract_sender_id(text: str):
    """
    Pull a sender header off the front of the message body, if one is present.
    Returns (sender_id, remaining_body). sender_id is None when the message
    carries no header — in that case the caller should supply it separately.
    """
    if not isinstance(text, str):
        return None, ""

    match = LABELLED_SENDER_PATTERN.match(text) or BARE_SENDER_PATTERN.match(text)
    if not match:
        return None, text.strip()

    sender = re.sub(r"\s+", "", match.group(1)).upper()
    body = text[match.end():].strip()
    return sender, body


def extract(raw_sms: str, sender_id: str = None) -> dict:
    """
    Split one incoming SMS into its parts.

    raw_sms   : the message as received (optionally with a "SENDER: " header)
    sender_id : pass explicitly when it arrives as metadata rather than in
                the body; an explicit value always wins over a parsed header.
    """
    parsed_sender, body = extract_sender_id(raw_sms)

    return {
        "raw": raw_sms,
        "sender_id": sender_id or parsed_sender,
        "text": body,
        "urls": extract_urls(body),
    }
