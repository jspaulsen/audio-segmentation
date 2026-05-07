from typing import Protocol, runtime_checkable

import numpy as np

from audio_segmentation.types.segment import RawSegment


@runtime_checkable
class Aligner(Protocol):
    """
    Protocol for forced alignment of transcripts to audio.

    Aligners take audio and a transcript, returning word-level segments
    with precise timing boundaries.
    """

    @property
    def required_sample_rate(self) -> int | None:
        """
        Required sample rate for the aligner.
        If None, any sample rate is accepted.
        """
        ...

    def align(
        self,
        audio: np.ndarray,
        sr: int,
        transcript: str,
    ) -> list[RawSegment]:
        """
        Align transcript to audio, returning word-level segments.

        Args:
            audio: Audio waveform as numpy array.
            sr: Sample rate of the audio.
            transcript: Text transcript to align to the audio.

        Returns:
            List of RawSegment with word-level timing boundaries.
        """
        ...
