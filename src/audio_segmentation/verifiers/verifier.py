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

    def compute_similarity(
        self,
        first: Tensor,
        second: Tensor,
        threshold: float = 0.25,
    ) -> tuple[float, bool]:
        ...
