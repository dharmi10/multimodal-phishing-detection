"""
HTTP wrapper around the existing smishing pipeline.

This file adds no detection logic. It turns a request into a Message, calls
analyze_message(), and returns the result dict unchanged. Everything the
detector decides is still decided in src/pipeline/.

Run from the project root, with the project venv:

    venv/Scripts/python.exe -m uvicorn api.main:app --port 8000
"""
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from src.pipeline.message import Message  # noqa: E402
from src.pipeline.pipeline import analyze_message  # noqa: E402

# Same bound the Streamlit app implies. Long bodies cost nothing extra to score,
# but an unbounded field on a public endpoint is still a bad default.
MAX_TEXT = 1000
MAX_SENDER = 40
MAX_IMAGES = 5
MAX_IMAGE_BYTES = 5 * 1024 * 1024

app = FastAPI(title="Smishing Check API", version="1.0")

# The Vite dev server proxies /api, so CORS is only needed when the built
# frontend is served from a different origin. Keep it narrow.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(
    text: str = Form(""),
    sender: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[]),
):
    """
    Run one SMS, with any attached images, through every layer.

    Multipart form: `text` and `sender` are optional strings, `images` is zero
    or more image files. An image-only message (no text) is allowed, because
    layer 4 can read a message entirely from a screenshot.

    The response is the pipeline's own dict: input, layer1_sms, layer2_url,
    layer3_sender, layer4_image and fusion. Nothing is renamed or recomputed.
    """
    text = (text or "").strip()
    sender = (sender or "").strip() or None

    if len(text) > MAX_TEXT:
        raise HTTPException(status_code=422, detail=f"Message text is longer than {MAX_TEXT} characters.")
    if sender and len(sender) > MAX_SENDER:
        raise HTTPException(status_code=422, detail=f"Sender ID is longer than {MAX_SENDER} characters.")
    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=422, detail=f"Attach at most {MAX_IMAGES} images.")

    image_bytes = []
    for upload in images:
        if not (upload.content_type or "").startswith("image/"):
            raise HTTPException(status_code=422, detail=f"{upload.filename} is not an image.")
        data = upload.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=422, detail=f"{upload.filename} is larger than 5 MB.")
        image_bytes.append(data)

    if not text and not image_bytes:
        raise HTTPException(status_code=422, detail="Add the message text or an image.")

    # sync def, so FastAPI runs this in a worker thread. Layer 2 makes live
    # HTTP requests, and blocking the event loop on them would stall the server.
    message = Message(sender=sender, text=text, images=image_bytes)
    return analyze_message(message)
