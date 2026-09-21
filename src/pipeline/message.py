"""
The pipeline's input contract.

One object describing an incoming message, whatever carried it. This exists so
that adding image handling did not mean adding a second parameter to every
function in the chain, and so that the next carrier - RCS, a chat app export -
is a new field rather than a new signature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Carrier names seen so far. Not an enum and not validated: an unrecognised
# channel is recorded as given rather than rejected, because the field is
# descriptive only.
CHANNEL_SMS = "sms"
CHANNEL_MMS = "mms"
CHANNEL_RCS = "rcs"


@dataclass
class Message:
    """
    One incoming message.

    sender  : the sender ID as carrier metadata, or None when it is unknown or
              embedded in the body (stage 0 will try to parse it out of `text`).
    text    : the message body. "" for an image-only message.
    images  : attached images as raw bytes, in the order they arrived.
    channel : which carrier delivered this.

    CHANNEL IS METADATA ONLY.
    -------------------------
    No layer reads `channel`, and it must never become a feature. It records
    provenance for future MMS/RCS work and for reading logs back - nothing else.

    The reason is not stylistic. In any dataset this project could realistically
    collect, channel would correlate almost perfectly with label: images arrive
    over MMS/RCS, and the image-bearing samples anyone bothers to collect are
    the phishing ones. A model given that column would learn "MMS means
    phishing", which is a fact about the collection process and not about
    smishing - the same trap that keeps `l2_present` out of the fusion layer
    (see scripts/11_train_fusion.py). Keeping it unread by construction is
    cheaper than discovering later that something read it.
    """

    sender: Optional[str] = None
    text: str = ""
    images: List[bytes] = field(default_factory=list)
    channel: str = CHANNEL_SMS

    def has_images(self) -> bool:
        """True when at least one image is attached."""
        return bool(self.images)

    @classmethod
    def from_sms(cls, raw_sms: str, sender_id: Optional[str] = None) -> "Message":
        """
        Build a plain text SMS, matching the original analyze_sms() signature.

        The body is left exactly as received - stage 0 still parses a "VM-HDFCBK:"
        style header out of it, so a raw string keeps working end to end.
        """
        return cls(sender=sender_id, text=raw_sms, images=[], channel=CHANNEL_SMS)
