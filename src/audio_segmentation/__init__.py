import importlib.util

from audio_segmentation.segment import Segment
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

# Export all classes and functions
__all__ = [
    "Segment",
    "Transcriber",
    "TranscriptionResult",
    "transcribe_audio",
    "NemoTranscriber",
    "NemoModel",
    "WhisperxTranscriber",
    "WhisperxModel",
]
