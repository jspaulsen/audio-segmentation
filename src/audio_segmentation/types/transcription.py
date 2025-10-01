from dataclasses import dataclass

from audio_segmentation.types.segment import Segment


@dataclass
class TranscriptionResult:
    transcript: str
    """
    Entire transcription of the audio.
    """

    segments: list[Segment]
    """
    List of segments with start and end times. These can be
    word level or sentence level segments.
    """
