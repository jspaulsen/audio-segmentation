"""
Pull a small, deterministic subset of jspaulsen/libritts-r-aligned and write
audio + MFA word alignments into tests/fixtures/libritts/ for use by the
aligner accuracy tests.

Usage:
    uv run --with datasets --with soundfile --with librosa \
        python scripts/build_libritts_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset


FIXTURE_DIR = Path("tests/fixtures/libritts")
TARGET_SR = 16000
NUM_SAMPLES = 5
MIN_DURATION_S = 12.0
MAX_DURATION_S = 20.0


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "jspaulsen/libritts-r-aligned", split="train", streaming=True
    )

    selected: list[dict] = []
    for ex in ds:
        if len(selected) >= NUM_SAMPLES:
            break

        meta = ex["json"]
        duration = float(meta["duration"])
        if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
            continue
        if not meta.get("words"):
            continue

        audio = ex["flac"]
        # AudioDecoder objects expose .get_all_samples() / dict access depending
        # on datasets version; normalize to (array, sr).
        if hasattr(audio, "get_all_samples"):
            samples = audio.get_all_samples()
            arr = np.asarray(samples.data).squeeze()
            sr = int(samples.sample_rate)
        else:
            arr = np.asarray(audio["array"])
            sr = int(audio["sampling_rate"])

        if arr.ndim > 1:
            arr = librosa.to_mono(arr.T if arr.shape[0] > arr.shape[1] else arr)
        arr = arr.astype(np.float32)

        if sr != TARGET_SR:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)

        sample_id = meta["id"]
        flac_path = FIXTURE_DIR / f"{sample_id}.flac"
        json_path = FIXTURE_DIR / f"{sample_id}.json"

        sf.write(flac_path, arr, TARGET_SR, format="FLAC")
        json_path.write_text(
            json.dumps(
                {
                    "id": sample_id,
                    "text": meta["text"],
                    "duration": duration,
                    "words": meta["words"],
                    "sample_rate": TARGET_SR,
                },
                indent=2,
            )
        )
        selected.append(
            {"id": sample_id, "duration": duration, "n_words": len(meta["words"])}
        )
        print(
            f"  saved {sample_id}  dur={duration:.2f}s  words={len(meta['words'])}"
        )

    manifest_path = FIXTURE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(selected, indent=2))
    print(f"\nWrote {len(selected)} samples to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
