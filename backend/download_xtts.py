import torch
from TTS.api import TTS  # coqui-ai TTS or compatible


# السماح للكلاسات المطلوبة (لـ torch 2.6+)
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig

    torch.serialization.add_safe_globals([
        XttsConfig,
        XttsAudioConfig,
        XttsArgs,
        BaseDatasetConfig
    ])
except Exception as e:
    print("Safe globals warning:", e)

from TTS.api import TTS

print("Downloading XTTS model...")
TTS(
    "tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True,
    gpu=False
)
print("XTTS model downloaded successfully")
