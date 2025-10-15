from dataclasses import dataclass
import numpy as np


@dataclass
class Audio:
    data: np.ndarray
    sr: int

    def _ms_to_index(self, milliseconds: int) -> int:
        return int(milliseconds * self.sr / 1000)

    def __getitem__(self, key: slice) -> np.ndarray:
        if isinstance(key, int):
            return self.data[key]

        if not isinstance(key, slice):
            raise TypeError("Invalid argument type.")

        start = self._ms_to_index(key.start) if key.start is not None else None
        stop = self._ms_to_index(key.stop) if key.stop is not None else None
        step = key.step

        return self.data[start:stop:step]

    def __len__(self) -> int:
        return int(len(self.data) * 1000 / self.sr)  # length in milliseconds

    def __repr__(self) -> str:
        return f"Audio(sr={self.sr}, duration={len(self)} ms)"
