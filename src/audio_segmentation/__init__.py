import importlib.util

from audio_segmentation.refine import (
    refine_segment_timestamps,
    refine_sentence_segments,
)

from audio_segmentation.types.segment import Segment
from audio_segmentation.segmenter import SegmentationException
from audio_segmentation.transcribe import transcribe_audio
from audio_segmentation.transcriber.transcriber import Transcriber

from audio_segmentation.utility import load_audio


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
    "load_audio",
    "NemoModel",
    "NemoTranscriber",
    # "RawTranscriptionResult",
    "refine_segment_timestamps",
    "refine_sentence_segments",
    "Segment",
    "SegmentationException",
    "transcribe_audio",
    "Transcriber",
    "WhisperxModel",
    "WhisperxTranscriber",
]
