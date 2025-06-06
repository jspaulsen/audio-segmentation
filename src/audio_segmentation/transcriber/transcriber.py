import abc
from dataclasses import dataclass

import pydub

from audio_segmentation.segment import RawSegment


@dataclass
class TranscriptionResult:
    transcript: str
    """
    Entire transcription of the audio.
    """

    segments: list[RawSegment]
    """
    List of segments with start and end times. These can be 
    word level or sentence level segments.
    """


class Transcriber(abc.ABC):
    """
    Abstract base class for audio transcribers.
    """

    def transcribe(
        self, 
        audio_segment: pydub.AudioSegment,
        word_level_segmentation: bool = True,
    ) -> TranscriptionResult:
        raise NotImplementedError("Transcriber must implement the transcribe method.")
    
    @property
    @abc.abstractmethod
    def ideal_segment_length(self) -> int | None:
        """
        Ideal segment length in milliseconds for the transcriber.
        This is used to determine how to split the audio segment
        for transcription.
        """
        return None
