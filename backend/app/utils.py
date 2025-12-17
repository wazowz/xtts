# utils.py
import logging
from pathlib import Path
import soundfile as sf
import librosa
import os

# Farasa import guarded
try:
    from farasa.diacratizer import FarasaDiacritizer
    _farasa = FarasaDiacritizer()
    FARASA_AVAILABLE = True
except Exception:
    _farasa = None
    FARASA_AVAILABLE = False

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a","webm"}
logger = logging.getLogger("utils")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


def allowed_file_or_not(filename: str, allowed=ALLOWED_EXTENSIONS) -> bool:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in allowed


def convert_to_wav_if_needed(path: str, target_sr: int = 16000) -> str:
    """ensure WAV 16k mono and return path"""
    p = Path(path)
    out_path = str(p.with_suffix('.wav'))
    try:
        data, sr = librosa.load(str(p), sr=None, mono=True)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sf.write(out_path, data, target_sr)
        logger.debug(f"Converted {p} -> {out_path} ({target_sr}Hz)")
    except Exception as e:
        logger.warning(f"convert_to_wav_if_needed failed on {p}: {e}")
        return str(p)
    return out_path


def farasa_diacritize(text: str) -> str:
    """diacritize using Farasa if available; else return original text"""
    logger.debug("[Farasa] Diacritize called")
    if not FARASA_AVAILABLE or _farasa is None:
        logger.warning("[Farasa] Not available - returning original text")
        return text
    try:
        diac = _farasa.diacritize(text)
        logger.debug("[Farasa] Done diacritizing")
        return diac
    except Exception as e:
        logger.exception("[Farasa] ERROR during diacritize")
        return text
