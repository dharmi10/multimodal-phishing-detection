"""
Smoke test for the layer 4 image decoder - runs decode_image on each image path
given on the command line and prints the result dict.

Manual, not pytest: it needs real image files to be pointed at, and the whole
point is to eyeball what came out of a picture, which no assertion captures.

A populated "error" is a legitimate outcome, not a crash. pytesseract is only a
wrapper around a separate Tesseract binary, so until that binary is installed
(and TESSERACT_CMD set, if it is not on PATH) every image will decode its QR
codes and report an OCR error. That is exactly what decode_image promises to do
rather than raise.

Run from the project root:
    python scripts/utils/test_decode.py path/to/image.png [more images...]
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage4_image.decode import decode_image

# Long OCR output is truncated in the printout only - decode_image itself
# returns the full string.
PREVIEW_CHARS = 400


def describe(path: Path) -> None:
    """Decode one image and print its result dict readably."""
    print("=" * 62)
    print(path)
    print("=" * 62)

    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        print(f"  could not read file: {type(exc).__name__}: {exc}\n")
        return

    result = decode_image(image_bytes)

    print(f"  size           : {result['width']} x {result['height']}")
    print(f"  qr_payloads    : {len(result['qr_payloads'])} found")
    for payload in result["qr_payloads"]:
        print(f"      - {payload}")

    print(f"  ocr_confidence : {result['ocr_confidence']:.2f}")
    print(f"  ocr_tokens     : {result['ocr_token_count']}  (alphanumeric, len >= 3)")
    text = result["ocr_text"]
    if text:
        preview = text[:PREVIEW_CHARS] + ("..." if len(text) > PREVIEW_CHARS else "")
        print(f"  ocr_text       : ({len(text)} chars) {preview}")
    else:
        print("  ocr_text       : (none)")

    print(f"  error          : {result['error']}")
    print()


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__.strip().splitlines()[-1].strip())
        print("Give it one or more image file paths.")
        return

    for raw_path in paths:
        describe(Path(raw_path))


if __name__ == "__main__":
    main()
