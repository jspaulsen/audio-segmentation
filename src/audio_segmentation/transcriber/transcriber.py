# import abc
from dataclasses import dataclass
from typing import Protocol

import numpy as np
# import pydub

from audio_segmentation.types.segment import RawSegment


@dataclass
class RawTranscriptionResult:
    transcript: str
    """
    Entire transcription of the audio.
    """

    segments: list[RawSegment]
    """
    List of segments with start and end times. These can be
    word level or sentence level segments.
    """


class Transcriber(Protocol):
    def transcribe(
        self,
        audio: np.ndarray,
        sr: int,
        **kwargs,
    ) -> RawTranscriptionResult:
        ...

    @property
    def ideal_segment_length(self) -> int | None:
        """
        Ideal segment length in milliseconds for the transcriber.
        This is used to determine how to split the audio segment
        for transcription.
        """
        return None

    @property
    def required_sample_rate(self) -> int | None:
        """
        Required sample rate for the transcriber.
        If None, any sample rate is accepted.
        """
        return None

    @property
    def requires_mono_audio(self) -> bool:
        """
        Whether the transcriber requires mono audio.
        """
        ...

    @property
    def supports_word_level_segmentation(self) -> bool:
        """
        Returns whether the transcriber supports word level segmentation.
        """
        ...
