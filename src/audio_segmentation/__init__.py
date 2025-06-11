import importlib.util

from audio_segmentation.refine import refine_segment_timestamps
from audio_segmentation.segment import Segment
from audio_segmentation.segmenter import SegmentationException
from audio_segmentation.transcribe import transcribe_audio
from audio_segmentation.transcriber.transcriber import (
    Transcriber,
    TranscriptionResult,
)


# Only import transcribers if their respective libraries are available
if importlib.util.find_spec("nemo"):
    from audio_segmentation.transcriber.nemo import (
        NemoTranscriber,
        NemoModel,
    )


if importlib.util.find_spec("whisperx"):
    from audio_segmentation.transcriber.whisperx import (
        WhisperxTranscriber,
        WhisperxModel,
    )


if importlib.util.find_spec("whisper_timestamped"):
    from audio_segmentation.transcriber.whisper_timestamped import (
        WhisperModel,
        WhisperTimestampedTranscriber,
    )


# Export all classes and functions
__all__ = [
    'refine_segment_timestamps',
    "Segment",
    "SegmentationException",
    "Transcriber",
    "TranscriptionResult",
    "transcribe_audio",

    # Nemo Transcriber
    "NemoTranscriber",
    "NemoModel",

    # Whisperx Transcriber
    "WhisperxTranscriber",
    "WhisperxModel",

    # Whisper Timestamped Transcriber
    "WhisperTimestampedTranscriber",
    "WhisperModel",
]
