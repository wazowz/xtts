# app/tts_service.py
import os
import time
import logging
from typing import Optional, List, Union
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
import webrtcvad
import torch
from pathlib import Path
from uuid import uuid4


from TTS.api import TTS  # coqui-ai TTS or compatible

# attempt to allow some config classes for safe serialization
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
except Exception:
    pass

logger = logging.getLogger("TTSService")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

class TTSService:
    def __init__(self,
                 model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 use_gpu: bool = False,
                 default_speaker_path: Optional[str] = None,
                 fp16: bool = False,
                 synthesis_options: dict = None):
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.default_speaker_path = default_speaker_path
        self.tts = None
        self.device = "cuda" if self.use_gpu else "cpu"
        self.fp16 = fp16
        self.synthesis_options = synthesis_options or {}
        logger.debug(f"[TTSService] init device={self.device} fp16={self.fp16} model={self.model_name}")

    def _ensure_model_loaded(self, log_collector: Optional[List[str]] = None):
        if self.tts is None:
            msg = f"[TTSService] Loading model: {self.model_name} ..."
            logger.debug(msg)
            if log_collector is not None: log_collector.append(msg)
            self.tts = TTS(self.model_name)
            try:
                if self.device == "cuda":
                    try:
                        self.tts.to(self.device)
                        logger.debug("[TTSService] moved model to CUDA")
                        if log_collector is not None: log_collector.append("Model moved to CUDA")
                    except Exception as ex:
                        logger.warning(f"[TTSService] failed to move to cuda: {ex}")
                if self.fp16:
                    try:
                        self.tts.model.half()
                        logger.debug("[TTSService] set model to fp16 (half)")
                        if log_collector is not None: log_collector.append("Model set to fp16")
                    except Exception as ex:
                        logger.warning(f"[TTSService] failed to set fp16: {ex}")
            except Exception:
                pass
            logger.debug("[TTSService] Model loaded.")

    # --------------------------- helpers ---------------------------
    def _rms(self, y: np.ndarray):
        return np.sqrt(np.mean(y ** 2) + 1e-12)

    def _match_rms(self, y: np.ndarray, target_rms: float):
        cur = self._rms(y)
        if cur == 0:
            return y
        return y * (target_rms / cur)

    def _preprocess_audio(self, file_path: str, target_sr: int = 16000, log_collector: Optional[List[str]] = None):
        """
        Preprocess single file:
         - load mono
         - resample to target_sr
         - light VAD trimming
         - light noise reduction
         - trim silence
         - return float32 array + sr
        """
        if log_collector is not None:
            log_collector.append(f"Preprocessing file: {file_path}")
        logger.debug(f"[TTSService] Preprocessing file: {file_path}")

        # load as mono
        y, sr = librosa.load(file_path, sr=None, mono=True)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            if log_collector is not None:
                log_collector.append(f"Resampled to {target_sr} Hz")

        # apply light VAD (webrtcvad) - less aggressive default
        try:
            y = self._apply_vad(y, sr, aggressiveness=1, log_collector=log_collector)
            if log_collector is not None:
                log_collector.append("Applied VAD trimming (light)")
        except Exception as e:
            logger.warning(f"[TTSService] VAD failed: {e}")
            if log_collector is not None:
                log_collector.append(f"VAD failed: {e}")

        # light noise reduction (prop_decrease lower to avoid artifacts)
        try:
            y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.4)
            if log_collector is not None:
                log_collector.append("Applied noise reduction (light)")
        except Exception as e:
            logger.warning(f"[TTSService] noise reduction failed: {e}")
            if log_collector is not None:
                log_collector.append(f"Noise reduction failed: {e}")

        # trim remaining silence (conservative)
        try:
            y, _ = librosa.effects.trim(y, top_db=35)
            if log_collector is not None:
                log_collector.append("Trimmed leading/trailing silence (conservative)")
        except Exception:
            pass

        # cast and return
        y = y.astype('float32')
        return y, sr

    def _apply_vad(self, y: np.ndarray, sr: int, aggressiveness: int = 1, log_collector: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply webrtcvad in a conservative way:
        - Convert to 16k if needed
        - Use 20ms frames
        - Keep frames where vad reports speech
        """
        target_sr = 16000
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            if log_collector is not None:
                log_collector.append(f"Resampled for VAD to {target_sr} Hz")

        try:
            int16 = (y * 32767).astype('int16')
            pcm_bytes = int16.tobytes()
            vad = webrtcvad.Vad(max(0, min(3, aggressiveness)))
            frame_duration = 20  # ms
            frame_bytes = int(sr * (frame_duration / 1000.0) * 2)
            voiced_frames = bytearray()
            for i in range(0, len(pcm_bytes), frame_bytes):
                frame = pcm_bytes[i:i + frame_bytes]
                if len(frame) < frame_bytes:
                    break
                try:
                    is_speech = vad.is_speech(frame, sample_rate=sr)
                except Exception:
                    is_speech = True
                if is_speech:
                    voiced_frames.extend(frame)

            if len(voiced_frames) == 0:
                if log_collector is not None:
                    log_collector.append("VAD found no voiced frames - returning original audio")
                return y

            voiced_int16 = np.frombuffer(bytes(voiced_frames), dtype='int16')
            voiced_float = voiced_int16.astype('float32') / 32767.0
            if log_collector is not None:
                log_collector.append(f"VAD kept {len(voiced_float)/sr:.2f}s of audio")
            return voiced_float
        except Exception as e:
            if log_collector is not None:
                log_collector.append(f"VAD exception: {e}")
            return y

    def _merge_audios(self, files: List[str], output_path: str, log_collector: Optional[List[str]] = None) -> str:
        """
        Preprocess each file individually (so VAD/NR apply per-file),
        then match loudness, concatenate, and write merged wav.
        """
        if log_collector is not None:
            log_collector.append(f"Preprocessing & merging {len(files)} files...")
        logger.debug(f"[TTSService] Preprocessing & merging {len(files)} files...")
        processed = []
        sr = 16000
        # target RMS we want for each file before merge
        target_rms = 0.03  # modest level to avoid clipping after merge

        for wav in files:
            try:
                y, sr = self._preprocess_audio(wav, target_sr=sr, log_collector=log_collector)
                # match RMS to target
                y = self._match_rms(y, target_rms)
                processed.append(y)
            except Exception as e:
                logger.warning(f"[TTSService] Failed to process {wav}: {e}")
                if log_collector is not None:
                    log_collector.append(f"Failed to process {wav}: {e}")

        if not processed:
            raise RuntimeError("No valid audio after preprocessing")

        merged = np.concatenate(processed)

        # final gentle normalization (avoid aggressive gain)
        merged = self._match_rms(merged, target_rms)

        merged_path = os.path.join(os.path.dirname(output_path), f"merged_{uuid_safe_basename(files[0])}.wav")
        sf.write(merged_path, merged, sr)
        if log_collector is not None:
            log_collector.append(f"Merged saved: {merged_path}")
        logger.debug(f"[TTSService] Merged saved: {merged_path}")
        return merged_path

    def synthesize(self,
                   text: str,
                   speaker_wav_path: Union[str, List[str], None],
                   output_path: str,
                   language: Optional[str] = "ar",
                   log_collector: Optional[List[str]] = None):
        """
        text: diacritized text
        speaker_wav_path: None | str path | list of str paths
        """
        self._ensure_model_loaded(log_collector=log_collector)
        start_all = time.time()

        # prepare speaker wav
        speaker_wav = None
        if isinstance(speaker_wav_path, list) and len(speaker_wav_path) > 1:
            if log_collector is not None:
                log_collector.append("Multiple speaker files provided -> will preprocess and merge")
            speaker_wav = self._merge_audios(speaker_wav_path, output_path, log_collector=log_collector)
        elif isinstance(speaker_wav_path, list) and len(speaker_wav_path) == 1:
            speaker_wav = speaker_wav_path[0]
        elif isinstance(speaker_wav_path, str):
            speaker_wav = speaker_wav_path
        else:
            speaker_wav = self.default_speaker_path

        if speaker_wav and not os.path.exists(speaker_wav):
            msg = f"[TTSService] Provided speaker file not found: {speaker_wav}"
            logger.warning(msg)
            if log_collector is not None:
                log_collector.append(msg)
            speaker_wav = None

        if log_collector is not None:
            log_collector.append(f"Using speaker: {speaker_wav or 'default model voice'}")
        logger.debug(f"[TTSService] Using speaker: {speaker_wav or 'default model voice'}")

        try:
            t0 = time.time()
            synth_kwargs = {
                "text": text,
                "speaker_wav": speaker_wav if speaker_wav else None,
                "language": language,
                "file_path": output_path
            }
            synth_kwargs.update(self.synthesis_options)

            if log_collector is not None:
                log_collector.append(f"Synthesizing text ({len(text)} chars)...")
            logger.debug("[TTSService] calling tts.tts_to_file ...")

            # call underlying TTS lib
            self.tts.tts_to_file(
                text=synth_kwargs["text"],
                speaker_wav=synth_kwargs.get("speaker_wav", None),
                language=synth_kwargs.get("language", None),
                file_path=synth_kwargs.get("file_path")
            )

            t_total = time.time() - t0
            msg = f"[TTSService] Audio generated: {output_path} (synthesis {t_total:.2f}s, total {(time.time()-start_all):.2f}s)"
            logger.debug(msg)
            if log_collector is not None:
                log_collector.append(msg)
        except Exception as e:
            logger.exception("[TTSService] Error during synthesis")
            if log_collector is not None:
                log_collector.append(f"Synthesis error: {e}")
            raise

# small helper
def uuid_safe_basename(path: str) -> str:
    try:
        return Path(path).name
    except Exception:
        return str(uuid4()).replace("-", "")[:8]
