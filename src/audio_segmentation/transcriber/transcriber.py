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

    # TODO: Deprecate this. Just don't support transcribers that don't include punctuation.
    @property
    def includes_punctuation(self) -> bool:
        """
        Returns whether the transcriber includes punctuation in the transcription.
        """
        ...



# TODO: Convert this to a protocol
# class _Transcriber(abc.ABC):
#     """
#     Abstract base class for audio transcribers.
#     """

#     def transcribe(
#         self,
#         audio_segment: pydub.AudioSegment,
#         # word_level_segmentation: bool = True,
#         **kwargs,
#     ) -> RawTranscriptionResult:
#         raise NotImplementedError("Transcriber must implement the transcribe method.")

#     @property
#     def ideal_segment_length(self) -> int | None:
#         """
#         Ideal segment length in milliseconds for the transcriber.
#         This is used to determine how to split the audio segment
#         for transcription.
#         """
#         return None

#     @property
#     @abc.abstractmethod
#     def supports_word_level_segmentation(self) -> bool:
#         """
#         Returns whether the transcriber supports word level segmentation.
#         """
#         raise NotImplementedError("Transcriber must implement the supports_word_level_segmentation property.")

#     # TODO: Deprecate this. Just don't support transcribers that don't include punctuation.
#     @property
#     @abc.abstractmethod
#     def includes_punctuation(self) -> bool:
#         """
#         Returns whether the transcriber includes punctuation in the transcription.
#         """
#         raise NotImplementedError("Transcriber must implement the includes_punctuation property.")
