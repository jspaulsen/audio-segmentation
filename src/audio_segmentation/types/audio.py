from __future__ import annotations
import numpy as np

from audio_segmentation.utility import resample, convert_to_mono


class Audio:
    def __init__(self, data: np.ndarray, sr: int) -> None:
        self.data = data
        self.sr = sr
    
    def resample(self, target_sr: int) -> Audio:
        resampled_audio = resample(
            self.data,
            original_sr=self.sr,
            target_sr=target_sr,
        )

        return Audio(data=resampled_audio, sr=target_sr)

    def to_mono(self) -> Audio:
        return Audio(
            data=convert_to_mono(self.data),
            sr=self.sr,
        )
    
    def to_numpy(self) -> np.ndarray:
        return self.data

    def _ms_to_index(self, milliseconds: int) -> int:
        return int(milliseconds * self.sr / 1000)

    def __getitem__(self, key: slice) -> Audio:
        if isinstance(key, int):
            return self.data[key]

        if not isinstance(key, slice):
            raise TypeError("Invalid argument type.")

        start = self._ms_to_index(key.start) if key.start is not None else None
        stop = self._ms_to_index(key.stop) if key.stop is not None else None
        step = key.step

        return Audio(data=self.data[start:stop:step], sr=self.sr)

    def __len__(self) -> int:
        return int(self.data.shape[-1] * 1000 / self.sr)  # length in milliseconds

    def __repr__(self) -> str:
        return f"Audio(sr={self.sr}, duration={len(self)} ms)"
