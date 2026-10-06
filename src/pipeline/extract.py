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


# ---------------------------------------------------------------------------
# Sender headers read off a screenshot
# ---------------------------------------------------------------------------
# A screenshot of an SMS carries its sender header, but not in a form the two
# patterns above can see. On a handset the header is not followed by a colon or
# a dash — it sits ALONE ON ITS OWN LINE above the message bubble, and that
# position is the only thing that marks it out:
#
#     10:45                                   78%      <- status bar
#     VM-MERIBNK                                       <- the sender
#     724193 is your One Time Password for the         <- the body
#
# Flattened into one string, which is how OCR text reaches layer 1,
# "VM-MERIBNK" is just another word and the sender is lost — which is exactly
# what happened: a screenshot had its text and its link read correctly while
# layer 3 reported "no sender ID was supplied". These rules work on the OCR
# LINES instead, so the position is still there to be read.
#
# Every rule here is deliberately conservative. A wrong sender is worse than no
# sender: it would be looked up against the register, and a near-miss of a bank
# header invented out of a misread word would raise a sender-impersonation
# alarm about a message that never had that sender.

# How far down from the top to look. The header is on line 1 or, with a status
# bar above it, line 2; a back arrow or a contact avatar on its own line can
# push it to 3. Past that we are reading the message body, where an all-caps
# word would be matched for its shape alone.
OCR_SENDER_SCAN_LINES = 4

# Confidence tiers, best first. The tier decides which candidate wins when a
# screenshot offers several, and it is ordered by how hard each shape is to
# produce by accident.
SENDER_TIER_PREFIXED = "operator-prefixed header"
SENDER_TIER_HEADER = "bare alphanumeric header"
SENDER_TIER_NUMERIC = "numeric short code"

SENDER_TIER_ORDER = (SENDER_TIER_PREFIXED, SENDER_TIER_HEADER, SENDER_TIER_NUMERIC)

# VM-MERIBNK, JK-JIOFBR-S. Two letters, a dash, the header, optionally the
# carrier's type suffix. No English word and no piece of app furniture has this
# shape, which is why it is the top tier and needs no stoplist.
OCR_PREFIXED_SENDER_PATTERN = re.compile(r"^[A-Z]{2}-[A-Z0-9]{3,10}(?:-[A-Z])?$")

# HDFCBK, MERIBNK. 99.2% of the 23,237 registered headers are exactly six
# uppercase letters, so the shape is right — but so is the shape of plenty of
# shouted words and button labels, hence the stoplist below.
OCR_BARE_SENDER_PATTERN = re.compile(r"^(?=[A-Z0-9]*[A-Z])[A-Z][A-Z0-9]{2,9}$")

# 121, 741247, and the 10-digit mobile number that layer 3 reports as an invalid
# header. Lowest tier: a line of nothing but digits in the header region is
# usually a short code, but it could be an OTP or an amount that OCR happened to
# isolate on its own line, so a better-shaped candidate always wins over one.
OCR_NUMERIC_SENDER_PATTERN = re.compile(r"^\d{3,12}$")

# Glyphs that messaging-app furniture puts next to the header and OCR reads as
# characters: back arrows, separators, bullets, a contact avatar read as "@".
# Stripped from the ENDS of a line only, so a clock ("10:45") or a date keeps
# its punctuation and simply fails every pattern below.
_LINE_CHROME = "<>«»|/\\•·–—―~^*@®©[](){}:;,.!?\"'“”‘’ \t"

# All-caps words that pass the bare-header shape test but are not senders.
# Two groups, and both are needed: the app's own furniture, which sits exactly
# where the header sits, and the words a smishing message shouts in its opening
# line, which is where the scan window lands when a screenshot is cropped to the
# message bubble alone.
OCR_SENDER_STOPWORDS = frozenset({
    # messaging app furniture
    "SMS", "MMS", "MESSAGE", "MESSAGES", "INBOX", "OUTBOX", "DRAFT", "DRAFTS",
    "ARCHIVE", "ARCHIVED", "STARRED", "SEARCH", "SETTINGS", "DELETE", "BLOCK",
    "REPORT", "SPAM", "JUNK", "CANCEL", "SELECT", "DETAILS", "REPLY", "SEND",
    "SENT", "CALL", "COPY", "SHARE", "FORWARD", "MORE", "BACK", "NEW", "ALL",
    "CHAT", "CHATS", "CONTACT", "CONTACTS", "INFO", "TRASH", "NOW", "TODAY",
    "YESTERDAY", "TEXT", "UNKNOWN", "NOTIFICATION", "NOTIFICATIONS", "MOBILE",
    "NETWORK", "WIFI", "DATA", "LTE", "VOLTE", "VOWIFI", "SIM",
    # words a message shouts in its opening line
    "URGENT", "ALERT", "ALERTS", "WARNING", "NOTICE", "IMPORTANT", "ATTENTION",
    "DEAR", "HELLO", "THANKS", "THANK", "FREE", "WINNER", "CONGRATS",
    "CONGRATULATIONS", "VERIFY", "UPDATE", "UPDATED", "KYC", "OTP", "PIN",
    "BANK", "ACCOUNT", "SECURITY", "CLICK", "HELP", "STOP", "START", "YES",
    "CONFIRM", "REMINDER", "FINAL", "EXPIRED", "SUSPENDED", "BLOCKED",
    "FROM", "SUBJECT", "REF", "NOTE",
})


def _header_tier(token: str):
    """
    The confidence tier `token` qualifies for as a sender header, or None.

    `token` is one whole OCR line with the app's furniture stripped off its
    ends. A header that is only PART of a line is deliberately not matched:
    surrounded by other words it is no longer in the position that identifies
    it, and matching it anyway is how a body word becomes a sender.
    """
    if not token:
        return None

    if OCR_PREFIXED_SENDER_PATTERN.match(token):
        return SENDER_TIER_PREFIXED

    if OCR_BARE_SENDER_PATTERN.match(token):
        return None if token in OCR_SENDER_STOPWORDS else SENDER_TIER_HEADER

    if OCR_NUMERIC_SENDER_PATTERN.match(token):
        return SENDER_TIER_NUMERIC

    return None


def sender_candidates_from_lines(lines, scan_lines: int = OCR_SENDER_SCAN_LINES) -> list:
    """
    Every plausible sender header in the top `scan_lines` lines of one image.

    lines : that image's OCR text as visual lines, top to bottom
            (src/stage4_image/decode.py -> "ocr_lines")

    Returns a list of dicts, best tier first:

        {"sender_id":  "VM-MERIBNK",
         "tier":       SENDER_TIER_PREFIXED,
         "line_index": 2,                  # 1-based, as a person counts lines
         "line":       "VM-MERIBNK"}       # the line exactly as OCR read it

    A list rather than one answer, because which candidate is right can depend
    on what the sender register makes of them, and that is layer 3's question,
    not stage 0's. Within a tier the order is top-to-bottom: nearer the top of
    the screenshot is nearer where a header belongs.
    """
    if not isinstance(lines, (list, tuple)):
        return []

    candidates = []
    for index, raw_line in enumerate(lines[:scan_lines], 1):
        if not isinstance(raw_line, str):
            continue
        token = raw_line.strip().strip(_LINE_CHROME)
        tier = _header_tier(token)
        if tier is None:
            continue
        candidates.append({
            "sender_id": token,
            "tier": tier,
            "line_index": index,
            "line": raw_line.strip(),
        })

    candidates.sort(key=lambda c: (SENDER_TIER_ORDER.index(c["tier"]), c["line_index"]))
    return candidates
