from pathlib import Path

import librosa
import numpy as np



def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    return librosa.to_mono(audio)


def resample(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    if original_sr == target_sr:
        return audio

    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)



def load_audio(
    audio_path: str | Path,
    sr: int | None = None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    ret, incoming_sr = librosa.load(
        audio_path,
        sr=sr,
        mono=mono,
    )

    return ret, int(incoming_sr)
