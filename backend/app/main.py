# app/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
import torch
import uuid
import traceback
import logging
from typing import List, Optional
import os
from .tts_service import TTSService
from .utils import allowed_file_or_not, convert_to_wav_if_needed, farasa_diacritize

from TTS.api import TTS  # coqui-ai TTS or compatible

# attempt to allow some config classes for safe serialization
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
except Exception:
    pass

LOG = logging.getLogger("xtts_fastapi")
logging.basicConfig(level=logging.DEBUG)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "generated"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a","webm"}

# Environment-driven defaults
USE_GPU = bool(int(os.environ.get("USE_GPU", "1"))) if "os" in globals() else True
DEFAULT_SPEAKER_PATH = "saniya-nrl.mp3"

tts_service = TTSService(
    model_name=os.environ.get("MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2"),
    use_gpu=USE_GPU,
    default_speaker_path=DEFAULT_SPEAKER_PATH,
    fp16=os.environ.get("FP16", "1") in ("1", "true", "True"),
    synthesis_options={
        "temperature": float(os.environ.get("TTS_TEMPERATURE", 0.45)),
        "top_p": float(os.environ.get("TTS_TOP_P", 0.9)),
        "repetition_penalty": float(os.environ.get("TTS_REP_PEN", 1.08)),
    }
)

app = FastAPI(title="XTTS FastAPI (Ultimate)")

# Allow wide CORS for dev / emulator; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index():
    default_name = Path(DEFAULT_SPEAKER_PATH).name if DEFAULT_SPEAKER_PATH else "model-default"
    html = f"""
    <html><body>
      <h3>XTTS FastAPI — Ultimate</h3>
      <p>Default voice: {default_name}</p>
      <p>Endpoints:</p>
      <ul>
        <li>POST /upload_speakers (multipart/form-data) → uploads speaker files, returns list of server paths</li>
        <li>POST /synthesize (form-data) → synthesizes, fields: text, mode(default|clone|merge), speaker_paths (comma-separated or absent)</li>
        <li>GET /generated/{{filename}} → download generated wav</li>
        <li>POST /generate — legacy endpoint (multipart) still supported</li>
      </ul>
    </body></html>
    """
    return html


@app.post("/upload_speakers")
async def upload_speakers(files: List[UploadFile] = File(...)):
    """
    Accepts multiple speaker files and returns server paths (converted to wav 16k mono).
    Flutter client uses this endpoint in the project you sent.
    """
    debug = []
    saved_paths = []
    try:
        for up in files:
            if not up.filename:
                continue
            if not allowed_file_or_not(up.filename, ALLOWED_EXTENSIONS):
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {up.filename}")
            unique = f"{uuid.uuid4().hex}_{up.filename}"
            dest = UPLOAD_DIR / unique
            with dest.open("wb") as f:
                content = await up.read()
                f.write(content)
            debug.append(f"Saved upload: {dest}")
            wav_path = convert_to_wav_if_needed(str(dest))
            debug.append(f"Converted to wav: {wav_path}")
            saved_paths.append(wav_path)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid files uploaded.")

        # return relative paths (frontend expects list of paths)
        rel_paths = [str(Path(p).as_posix()) for p in saved_paths]
        return JSONResponse({"ok": True, "paths": rel_paths, "debug": debug})
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e), "debug": debug}, status_code=500)


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    mode: str = Form("default"),  # default | clone | merge
    speaker_paths: Optional[str] = Form(None),  # comma-separated paths returned from upload_speakers
    use_diacritization: str = Form("true"),
):
    """
    Compatible with Flutter client: synthesize using text and previously uploaded speaker files.
    If mode == 'clone' or 'merge', speaker_paths must be provided (comma-separated).
    """
    debug_logs = []
    try:
        original_text = (text or "").strip()
        debug_logs.append(f"Original text: {original_text}")
        if not original_text:
            raise HTTPException(status_code=400, detail="الرجاء إدخال نص بالعربية.")

        # diacritize if requested
        if use_diacritization.lower() in ("true", "1", "yes"):
            debug_logs.append("Diacritize: calling farasa_diacritize...")
            diacritized_text = farasa_diacritize(original_text)
        else:
            diacritized_text = original_text
        debug_logs.append(f"Diacritized text: {diacritized_text}")

        paths_list = None
        if mode in ("clone", "merge"):
            if not speaker_paths:
                raise HTTPException(status_code=400, detail="speaker_paths required for clone/merge mode.")
            # accept csv or comma separated
            parts = [p.strip() for p in speaker_paths.split(",") if p.strip()]
            if not parts:
                raise HTTPException(status_code=400, detail="No valid speaker paths provided.")
            # convert relative paths to absolute if necessary
            abs_parts = []
            for p in parts:
                p_path = Path(p)
                if not p_path.is_absolute():
                    p_path = UPLOAD_DIR / p_path.name
                abs_parts.append(str(p_path))
            paths_list = abs_parts
            debug_logs.append(f"Using speaker files: {paths_list}")

        out_name = f"{uuid.uuid4().hex}_output.wav"
        out_path = str(OUTPUT_DIR / out_name)
        debug_logs.append("Starting synthesis (may take time)...")

        # run synthesis in threadpool
        await run_in_threadpool(
            tts_service.synthesize,
            diacritized_text,
            paths_list,
            out_path,
            "ar",
            debug_logs
        )

        debug_logs.append(f"Saved output: {out_path}")
        return JSONResponse({
            "ok": True,
            "audio_url": f"/generated/{out_name}",
            "original_text": original_text,
            "diacritized_text": diacritized_text,
            "debug_logs": debug_logs
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        debug_logs.append(f"Exception: {e}")
        return JSONResponse({"ok": False, "error": str(e), "debug_logs": debug_logs}, status_code=500)


# Legacy endpoint kept for compatibility
@app.post("/generate")
async def generate_legacy(
    text: str = Form(...),
    mode: str = Form("default"),
    speaker: Optional[List[UploadFile]] = File(None),
):
    # This maps legacy form behavior to the new synthesize flow.
    debug_logs = []
    try:
        # prepare speaker_paths array like /synthesize would expect
        paths_list = None
        if mode == "clone":
            if not speaker or len(speaker) == 0:
                raise HTTPException(status_code=400, detail="الرجاء رفع ملفات صوتية للكلون.")
            saved_paths = []
            for up in speaker:
                if not up.filename:
                    continue
                if not allowed_file_or_not(up.filename, ALLOWED_EXTENSIONS):
                    raise HTTPException(status_code=400, detail=f"نوع ملف غير مدعوم: {up.filename}")
                unique = f"{uuid.uuid4().hex}_{up.filename}"
                dest = UPLOAD_DIR / unique
                with dest.open("wb") as f:
                    content = await up.read()
                    f.write(content)
                debug_logs.append(f"Saved speaker upload: {dest}")
                wav_path = convert_to_wav_if_needed(str(dest))
                debug_logs.append(f"Converted to wav: {wav_path}")
                saved_paths.append(wav_path)
            if not saved_paths:
                raise HTTPException(status_code=400, detail="لم يتم رفع ملفات صالحة.")
            paths_list = saved_paths

        out_name = f"{uuid.uuid4().hex}_output.wav"
        out_path = str(OUTPUT_DIR / out_name)
        debug_logs.append("Starting synthesis (legacy endpoint)...")

        await run_in_threadpool(
            tts_service.synthesize,
            text,
            paths_list,
            out_path,
            "ar",
            debug_logs
        )

        debug_logs.append(f"Saved output: {out_path}")
        return JSONResponse({
            "ok": True,
            "audio_url": f"/generated/{out_name}",
            "original_text": text,
            "debug_logs": debug_logs
        })

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        debug_logs.append(f"Exception: {e}")
        return JSONResponse({"ok": False, "error": str(e), "debug_logs": debug_logs}, status_code=500)


@app.get("/generated/{filename}")
def download_file(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)
