"""SATYAVAANI - REST API (FastAPI; installed together with Gradio).
    python api_server.py      ->  http://127.0.0.1:8000/docs   (interactive docs)
Endpoints
    GET  /v1/health                      model + device info
    GET  /v1/voiceprints                 who is enrolled
    POST /v1/score   file, claimed_id?, context_prior?   -> risk, verdict, per-window details
    POST /v1/enroll  file, name                          -> new voiceprint id
"""
import os, tempfile
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import uvicorn
from sv_audio import SR, load16, rms_norm
from sv_engine import Engine

ENGINE = Engine.load()
app = FastAPI(title="SATYAVAANI API", version="1.0",
              description="Real-time AI voice-clone detection: one wav2vec2 pass, a deepfake head and a speaker head.")

def _read(upload: UploadFile):
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(upload.file.read()); path = f.name
    try:
        return load16(path)
    except Exception as e:
        raise HTTPException(400, f"could not read audio: {e}")
    finally:
        os.unlink(path)

@app.get("/v1/health")
def health():
    return {"ok": True, "device": ENGINE.info.get("gpu"), "ms_per_window": ENGINE.info.get("startup_ms"),
            "speaker_threshold": ENGINE.thr, "voiceprints": len(ENGINE.voiceprints)}

@app.get("/v1/voiceprints")
def voiceprints():
    return [{"id": vid, "name": name} for vid, (name, _) in ENGINE.voiceprints.items()]

@app.post("/v1/score")
def score(file: UploadFile = File(...), claimed_id: str = Form(None), context_prior: float = Form(0.0)):
    """Score a recording (3 s or more). claimed_id = the voiceprint id the caller claims to be (optional)."""
    if claimed_id and claimed_id not in ENGINE.voiceprints:
        raise HTTPException(404, f"unknown claimed_id '{claimed_id}' - see /v1/voiceprints")
    w = rms_norm(_read(file))
    return ENGINE.score_audio(w, claimed=claimed_id or None, prior=context_prior)

@app.post("/v1/enroll")
def enroll(file: UploadFile = File(...), name: str = Form(...)):
    w = _read(file)
    if len(w) < 5 * SR: raise HTTPException(400, "need at least 5 s of speech")
    vid, n = ENGINE.enroll(name, w)
    return {"id": vid, "name": name, "windows": n, "stored_bytes": 192 * 4}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
