"""
Layer 3 — sender ID reputation.

Replaces the old exact-match lookup over the ~450-row custom dataset, which
contributed almost nothing: out of fold it produced 194 known-legit matches and
ZERO known-phishing ones, because a phishing sender ID is single-use and never
appears twice. It also failed on real traffic, because the same sender arrives
under several different headers — JIOFBR was seen with four operator prefixes
in the capture (JD, JK, JM, JX) and PNBSMS with eight — and an exact string
match treats every one of them as a different sender.

What this layer does now
------------------------
It answers one question — "is this header on the register?" — against the 23,237
legitimate headers in data/sender_registry_all.csv, after normalising away the
operator prefix and type suffix that the carrier adds in transit. Then, for the
headers that are NOT registered, it adds two narrow checks that the register
makes possible:

    IMPERSONATION  - a near-miss of a high-value brand header (YES8NK vs YESBNK)
    INVALID_HEADER - a shape no registered header has (a 10-digit mobile number)

Both are rules, not model output, and both are reported as such.

What it does NOT do
-------------------
It does not return a phishing score. `sender_phish_score` is 0.0 when the header
is registered and None in every other case — exactly the contract the trained
fusion model was fitted against. Coverage goes from ~450 headers to 23,237; the
meaning of the number does not change, so the model needs no retraining and gets
none. Impersonation reaches the verdict as a label override in fusion.py, not as
a feature, because the fusion's l3_known_phish column was all-zero in training
and carries a coefficient of ~0: a phishing score returned from here would be
multiplied by nothing.

The old data/custom-dataset_4600.csv.xls is deliberately left on disk and
untouched. It is still what scripts/10 builds fusion_features.csv from, and
removing it would break the fusion layer's provenance.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "data" / "sender_registry_all.csv"

# Only "legit" rows build the table. The file carries exactly one row labelled
# "phishing" (YES8NK, a typosquat of YESBNK) and loading it would register the
# precise impersonation this layer exists to catch — the header would come back
# REGISTERED with a 0.0 risk score. The filter is the whole point, not hygiene.
LEGIT_LABEL = "legit"

# --- Header normalisation --------------------------------------------------
# Carriers wrap the registered header in transit: a two-letter operator+region
# prefix in front, a single-letter service-type suffix behind. Neither belongs
# to the sender, and both vary run to run for the same sender.
TYPE_SUFFIXES = frozenset({"S", "T", "P", "G"})
SUFFIX_PATTERN = re.compile(r"-([A-Z])$")
PREFIX_PATTERN = re.compile(r"^([A-Z]{2})-")

# --- Structural plausibility ----------------------------------------------
# Measured over the 23,190 TRAI headers in the registry:
#   - 20,413 of 20,569 alphanumeric headers are exactly 6 uppercase A-Z (99.2%)
#   - numeric headers run 1 to 7 digits; 2,536 of 2,623 are 6 digits
#   - ZERO headers are 10 digits
#   - captured real traffic adds 8-digit international short codes
#     (Uber 57575711, NVIDIA 57273201), so 8 digits is legitimate too
#
# The only reliable negative is therefore the 10-digit mobile shape. 5, 6, 7 and
# 8 digit numeric headers are all real bank and service short codes, and
# flagging them would reject legitimate traffic wholesale. The threshold is 9
# rather than 10 only to catch the shape without assuming an exact length.
INVALID_NUMERIC_MIN_DIGITS = 9

# --- Impersonation matching ------------------------------------------------
# Matched against a SUBSET of the registry, never all of it. 39.8% of registered
# headers sit within edit distance 1 of another registered header, so matching
# against everything would fire on almost every unregistered header and mean
# nothing.
# Damerau-Levenshtein at cutoff 1, NOT Levenshtein at cutoff 2. Measured against
# 47 held-out legitimate headers (real senders the registry does not list) and 8
# typosquats:
#
#   Levenshtein, cutoff 2 : 19/47 legitimate headers falsely flagged (40.4%) -
#                           Blinkit vs BANKIT, Shein vs a cooperative bank,
#                           ISKCON vs Indraprastha Sehkari Bank. Unusable.
#   Damerau, cutoff 1     : 4/47 falsely flagged (8.5%), all of them MEDIUM
#   Damerau, cutoff 1,
#     HIGH tier only      : 0/47 falsely flagged, and 8/8 typosquats caught
#
# The transposition is what forces Damerau. HFDCBK -> HDFCBK is the canonical
# typosquat and plain Levenshtein scores it 2, so catching it with Levenshtein
# would mean a cutoff of 2 and the 40.4% above.
IMPERSONATION_MAX_DISTANCE = 1

# --- Edit types and tiers --------------------------------------------------
# What KIND of single edit separates a candidate from the brand header it
# resembles. The distinction is the whole reason the false-positive rate drops
# from 8.5% to 0: a legitimate sender extending its own header range differs by
# a letter (ICICIO / ICICIT alongside ICICIH, KOTAKA alongside KOTAKL), while an
# attacker swapping a digit in for a letter or transposing two characters is
# doing something no legitimate registrant has a reason to do.
EDIT_DIGIT_SUBSTITUTION = "digit-for-letter substitution"
EDIT_TRANSPOSITION = "adjacent transposition"
EDIT_LETTER_SUBSTITUTION = "letter substitution"
EDIT_INSERTION = "insertion"
EDIT_DELETION = "deletion"

TIER_HIGH = "high"
TIER_MEDIUM = "medium"

# Only these two edit types produce IMPERSONATION_HIGH, and only HIGH overrides
# the verdict label. Everything else is reported and left to the model.
# Measured: HIGH = 0/47 false positives; MEDIUM = 4/47 (BLNKIT vs BANKIT,
# ICICIO and ICICIT vs ICICIH, KOTAKA vs KOTAKL), three of which are the same
# brand extending its own header range - exactly what MEDIUM should mean.
HIGH_RISK_EDIT_TYPES = frozenset({EDIT_DIGIT_SUBSTITUTION, EDIT_TRANSPOSITION})

# Entity-name keywords that make a registered header worth protecting. Banking
# and finance, where an impersonation actually pays.
BRAND_ENTITY_KEYWORDS = (
    "BANK", "FINANCE", "FINSERV", "INSURANCE",
    "SECURITIES", "PAYMENTS", "MUTUAL FUND",
)

# Hand-maintained additions: high-value non-bank headers worth protecting that
# the keyword rule above will not catch — telecoms, wallets, major e-commerce,
# government services. Extend this list rather than editing code.
#
# Entries are intersected with the registry at load time, so a header listed
# here that is not on the register is dropped with a warning rather than
# becoming a match target. That matters: a brand header that was NOT registered
# would sit at edit distance 0 from itself and report every legitimate use of it
# as an impersonation of itself.
HIGH_VALUE_BRAND_HEADERS = frozenset({
    # telecom
    "AIRTEL", "JIOFBR",
    # wallets / UPI
    "PAYTMB", "PAYTMM", "PHONPE", "PHONEP",
    # e-commerce / delivery
    "FLPKRT", "FLIPKT", "MYNTRA", "SWIGGY", "ZOMATO", "BLNKIT",
    # government / public services
    #
    # INCTAX is deliberately NOT here. The registry assigns that header to "ABC
    # ACCOUNTANCY SERVICES PVT LTD" and holds no Income Tax Department entity at
    # all, so protecting it would report an impersonation against the wrong
    # organisation - naming a private accountancy firm as the victim of a
    # government-impersonation scam.
    "EPFOHO", "IRCTCI", "MYGOVT",
})

# Status values. Lowercase so the existing report, which upper-cases them for
# display, keeps working unchanged.
STATUS_REGISTERED = "registered"
STATUS_IMPERSONATION_HIGH = "impersonation_high"
STATUS_IMPERSONATION_MEDIUM = "impersonation_medium"
STATUS_INVALID = "invalid_header"
STATUS_UNKNOWN = "unknown"
STATUS_MISSING = "missing"


def normalize_header(sender_id: str) -> Dict[str, Optional[str]]:
    """
    Strip the carrier's wrapping off a sender ID.

        JK-JIOFBR-S  ->  header JIOFBR, prefix JK, suffix S
        HDFCBK-T     ->  header HDFCBK, prefix None, suffix T
        VM-HDFCBK    ->  header HDFCBK, prefix VM, suffix None
        JIOFBR       ->  header JIOFBR, prefix None, suffix None

    The prefix and suffix come back separately rather than being discarded, so
    the report can still show what actually arrived — "VM-HDFCBK (HDFCBK)" is
    more use to someone reading it than either half alone.

    Anything left that is not A-Z or 0-9 is removed last. That is what turns a
    phone number written as "+91 98765 43210" into the bare digits the
    structural check needs to see.
    """
    if not isinstance(sender_id, str):
        return {"raw": None, "header": "", "prefix": None, "suffix": None}

    raw = sender_id.strip()
    text = raw.upper()

    suffix = None
    match = SUFFIX_PATTERN.search(text)
    if match and match.group(1) in TYPE_SUFFIXES:
        suffix = match.group(1)
        text = text[: match.start()]

    prefix = None
    match = PREFIX_PATTERN.match(text)
    if match:
        prefix = match.group(1)
        text = text[match.end():]

    header = re.sub(r"[^A-Z0-9]", "", text)
    return {"raw": raw, "header": header, "prefix": prefix, "suffix": suffix}


# Kept under its old name so anything still importing it does not break. The old
# implementation only upper-cased and trimmed; this one also strips the carrier
# wrapping, which is the entire fix.
def normalize_sender(sender_id: str) -> str:
    """The bare registered header, with prefix and suffix removed."""
    return normalize_header(sender_id)["header"]


@lru_cache(maxsize=1)
def load_registry() -> Dict[str, str]:
    """
    Build {header: entity_name} from the legitimate rows of the registry.

    Duplicate headers keep the first entity name seen. The registry does carry
    repeats — the same header registered to the same entity under different
    TRAI records — and they do not conflict in a way worth resolving here.
    """
    df = pd.read_csv(
        REGISTRY_PATH,
        dtype=str,
        keep_default_na=False,
        usecols=["header", "label", "entity_name"],
    )

    total = len(df)
    df = df[df["label"].str.strip().str.lower() == LEGIT_LABEL]
    dropped = total - len(df)

    df = df[df["header"].str.strip() != ""]

    table: Dict[str, str] = {}
    for header, entity in zip(df["header"], df["entity_name"]):
        key = header.strip().upper()
        if key not in table:
            table[key] = entity.strip()

    logger.info(
        "layer 3 registry: %d headers loaded (%d non-legit rows excluded)",
        len(table), dropped,
    )
    return table


@lru_cache(maxsize=1)
def load_brand_headers() -> Dict[str, str]:
    """
    The subset of registered headers that impersonation is matched against.

    Keyword-matched banking and finance entities, plus the hand-maintained list,
    intersected with the registry.
    """
    registry = load_registry()

    brands: Dict[str, str] = {}
    for header, entity in registry.items():
        upper = entity.upper()
        if any(keyword in upper for keyword in BRAND_ENTITY_KEYWORDS):
            brands[header] = entity

    keyword_count = len(brands)

    missing = []
    for header in HIGH_VALUE_BRAND_HEADERS:
        if header in registry:
            brands[header] = registry[header]
        else:
            missing.append(header)

    if missing:
        logger.warning(
            "layer 3: %d hand-listed brand header(s) are not on the register and "
            "were dropped: %s", len(missing), ", ".join(sorted(missing)),
        )

    logger.info(
        "layer 3 impersonation subset: %d headers (%d by entity keyword, %d added "
        "by hand) out of %d registered",
        len(brands), keyword_count, len(brands) - keyword_count, len(registry),
    )
    return brands


def damerau_levenshtein(a: str, b: str,
                        max_distance: int = IMPERSONATION_MAX_DISTANCE) -> int:
    """
    Edit distance counting an ADJACENT TRANSPOSITION as one edit, alongside
    substitutions, insertions and deletions.

    This is the restricted (optimal string alignment) variant, which differs
    from unrestricted Damerau-Levenshtein only for strings needing edits
    *between* two transposed characters. At distance 1 the two are identical, and
    1 is the only cutoff this module uses.

    Implemented here rather than pulled in as a dependency: headers are at most
    about eight characters, so the textbook dynamic program is both fast enough
    and short enough to read.

    Returns max_distance + 1 as soon as the answer is known to exceed the cutoff,
    rather than the true distance. Callers only compare against the cutoff.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1

    len_a, len_b = len(a), len(b)
    # Three rows: two back for the transposition check, one back for the rest.
    two_back = None
    previous = list(range(len_b + 1))

    for i in range(1, len_a + 1):
        current = [i] + [0] * len_b
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            value = min(
                previous[j] + 1,            # deletion
                current[j - 1] + 1,         # insertion
                previous[j - 1] + cost,     # substitution
            )
            if (i > 1 and j > 1
                    and a[i - 1] == b[j - 2]
                    and a[i - 2] == b[j - 1]):
                value = min(value, two_back[j - 2] + 1)   # transposition
            current[j] = value

        if min(current) > max_distance:
            return max_distance + 1

        two_back, previous = previous, current

    return previous[-1]


def classify_edit(candidate: str, brand: str) -> Optional[str]:
    """
    Name the single edit separating `candidate` from `brand`, or None when they
    are not exactly one edit apart.

    Called only after damerau_levenshtein has confirmed a distance of 1, but it
    verifies the structure itself rather than trusting that: the insertion and
    deletion branches in particular need to confirm that removing one character
    really does close the gap, since a length difference of 1 says nothing on
    its own.

    A digit standing where the brand has a letter is the attacker's move
    (YES8NK for YESBNK). The reverse - a letter where the brand has a digit -
    is treated as an ordinary letter substitution, because no measured typosquat
    took that form and nothing justifies promoting it.
    """
    if candidate == brand:
        return None

    len_c, len_b = len(candidate), len(brand)

    if len_c == len_b:
        differing = [i for i in range(len_c) if candidate[i] != brand[i]]

        if len(differing) == 1:
            index = differing[0]
            if candidate[index].isdigit() and brand[index].isalpha():
                return EDIT_DIGIT_SUBSTITUTION
            return EDIT_LETTER_SUBSTITUTION

        if len(differing) == 2 and differing[1] == differing[0] + 1:
            first, second = differing
            if candidate[first] == brand[second] and candidate[second] == brand[first]:
                return EDIT_TRANSPOSITION

        return None

    if len_c == len_b + 1:
        # One character too many: dropping it must yield the brand header.
        for i in range(len_c):
            if candidate[:i] + candidate[i + 1:] == brand:
                return EDIT_INSERTION
        return None

    if len_c + 1 == len_b:
        for i in range(len_b):
            if brand[:i] + brand[i + 1:] == candidate:
                return EDIT_DELETION
        return None

    return None


def tier_for_edit(edit_type: Optional[str]) -> Optional[str]:
    """HIGH for the attacker-shaped edits, MEDIUM for the rest."""
    if edit_type is None:
        return None
    return TIER_HIGH if edit_type in HIGH_RISK_EDIT_TYPES else TIER_MEDIUM


def nearest_brand(header: str) -> Optional[Tuple[str, str, int, str, str]]:
    """
    The closest high-value brand header within the cutoff, or None.

    Returns (brand_header, entity_name, distance, edit_type, tier).

    A HIGH-tier match is preferred over a MEDIUM one at the same distance, since
    at cutoff 1 every match is distance 1 and the edit type is the only thing
    separating them — a candidate one transposition from one brand and one
    letter from another should be reported as the transposition. Remaining ties
    break on the alphabetically first header, so the answer is deterministic
    rather than dependent on dictionary ordering.
    """
    if not header:
        return None

    brands = load_brand_headers()
    best: Optional[Tuple[str, str, int, str, str]] = None

    for brand, entity in brands.items():
        distance = damerau_levenshtein(header, brand)
        if distance > IMPERSONATION_MAX_DISTANCE or distance == 0:
            continue

        edit_type = classify_edit(header, brand)
        if edit_type is None:
            continue
        tier = tier_for_edit(edit_type)

        if best is None:
            best = (brand, entity, distance, edit_type, tier)
            continue

        better = (
            distance < best[2]
            or (distance == best[2] and tier == TIER_HIGH and best[4] != TIER_HIGH)
            or (distance == best[2] and tier == best[4] and brand < best[0])
        )
        if better:
            best = (brand, entity, distance, edit_type, tier)

    return best


def is_structurally_invalid(header: str) -> bool:
    """
    True for header shapes that no registered header has.

    Only one shape qualifies: all digits, 9 or more of them — a mobile number
    sent as a sender ID. See INVALID_NUMERIC_MIN_DIGITS for the counts behind
    that being the only safe negative.
    """
    if not header or not header.isdigit():
        return False
    return len(header) >= INVALID_NUMERIC_MIN_DIGITS


def _result(raw, parts, status, score, note, **extra) -> Dict:
    """One shape for every return, so callers never have to test for keys."""
    result = {
        "sender_id": raw,
        "header": parts["header"],
        "prefix": parts["prefix"],
        "suffix": parts["suffix"],
        "status": status,
        "sender_phish_score": score,
        "entity_name": None,
        "impersonation": False,
        "impersonation_tier": None,
        "impersonation_edit": None,
        "invalid_header": False,
        "resembles": None,
        "resembles_entity": None,
        "resembles_distance": None,
        "note": note,
    }
    result.update(extra)
    return result


def lookup_sender(sender_id: str) -> Dict:
    """
    Resolve one sender ID against the register.

    Check order is REGISTERED, then INVALID_HEADER, then IMPERSONATION, then
    UNKNOWN. Exact match must win outright: 39.8% of registered headers are
    within edit distance 1 of another registered header, so a registered sender
    would otherwise be reported as impersonating whichever neighbour it happens
    to sit next to.

    sender_phish_score is 0.0 only for REGISTERED. Every other state returns
    None — "no information" — which is what the fusion model was trained to
    expect. Impersonation does not become a score here; it travels as a flag and
    is applied as a label override in fusion.py — and only the HIGH tier
    overrides. A MEDIUM match is reported and left for the reader to weigh,
    because at that tier the same edit is what a brand extending its own header
    range looks like.
    """
    if not sender_id or not str(sender_id).strip():
        parts = {"raw": None, "header": "", "prefix": None, "suffix": None}
        return _result(None, parts, STATUS_MISSING, None,
                       "no sender ID was supplied with the message")

    parts = normalize_header(sender_id)
    header = parts["header"]
    raw = parts["raw"]

    if not header:
        return _result(raw, parts, STATUS_UNKNOWN, None,
                       "sender ID contained no usable header characters")

    registry = load_registry()

    # 1. Registered wins outright.
    entity = registry.get(header)
    if entity is not None:
        shown = f" (registered to {entity})" if entity else ""
        return _result(
            raw, parts, STATUS_REGISTERED, 0.0,
            f"header {header} is on the sender register{shown}",
            entity_name=entity or None,
        )

    # 2. Structurally impossible.
    if is_structurally_invalid(header):
        return _result(
            raw, parts, STATUS_INVALID, None,
            f"header {header} is a {len(header)}-digit number; no registered "
            "sender header has that shape",
            invalid_header=True,
        )

    # 3. Near-miss of a protected brand.
    match = nearest_brand(header)
    if match is not None:
        brand, brand_entity, distance, edit_type, tier = match
        owner = brand_entity or "an unnamed entity"

        if tier == TIER_HIGH:
            status = STATUS_IMPERSONATION_HIGH
            note = (f"header {header} is not registered and differs from {brand} "
                    f"({owner}) by one {edit_type} - the shape of a deliberate "
                    "typosquat")
        else:
            status = STATUS_IMPERSONATION_MEDIUM
            note = (f"header {header} is not registered and differs from {brand} "
                    f"({owner}) by one {edit_type}; that is also how a brand "
                    "extends its own header range, so it is reported but not "
                    "treated as proof")

        return _result(
            raw, parts, status, None, note,
            impersonation=True,
            impersonation_tier=tier,
            impersonation_edit=edit_type,
            resembles=brand,
            resembles_entity=brand_entity or None,
            resembles_distance=distance,
        )

    # 4. Plausible but unseen.
    return _result(
        raw, parts, STATUS_UNKNOWN, None,
        f"header {header} is not on the sender register",
    )


def registry_stats() -> Dict:
    """Coverage summary, for sanity checks."""
    registry = load_registry()
    brands = load_brand_headers()
    return {
        "registered_headers": len(registry),
        "brand_headers": len(brands),
        "impersonation_cutoff": IMPERSONATION_MAX_DISTANCE,
        "impersonation_metric": "damerau-levenshtein",
        "high_risk_edit_types": sorted(HIGH_RISK_EDIT_TYPES),
        "invalid_numeric_min_digits": INVALID_NUMERIC_MIN_DIGITS,
    }
