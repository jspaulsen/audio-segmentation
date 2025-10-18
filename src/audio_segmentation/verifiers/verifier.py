from typing import Protocol
import numpy as np
from torch import Tensor


class SpeakerVerifier(Protocol):
    @property
    def requires_mono_audio(self) -> bool:
        ...

    @property
    def required_sample_rate(self) -> int:
        ...

    def create_embedding(
        self,
        audio: np.ndarray | Tensor,
    ) -> Tensor:
        ...
