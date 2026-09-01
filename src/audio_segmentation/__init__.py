import importlib.util

from audio_segmentation.aligner.aligner import Aligner
from audio_segmentation.aligner.torchaudio import AlignerModel, TorchAudioAligner
from audio_segmentation.refine import (
    refine_segment_timestamps,
    refine_segment_timestamps_batch,
    refine_sentence_segments,
)
from audio_segmentation.types.audio import Audio
from audio_segmentation.types.segment import Segment
from audio_segmentation.segmenter import SegmentationException
from audio_segmentation.transcribe import transcribe_audio
from audio_segmentation.transcriber.transcriber import Transcriber
from audio_segmentation.utility import load_audio
from audio_segmentation.verifiers.verifier import SpeakerVerifier


# Export all classes and functions
__all__ = [
    "Aligner",
    "AlignerModel",
    "Audio",
    "load_audio",
    "refine_segment_timestamps",
    "refine_segment_timestamps_batch",
    "refine_sentence_segments",
    "Segment",
    "SegmentationException",
    "SpeakerVerifier",
    "TorchAudioAligner",
    "transcribe_audio",
    "Transcriber",
]


# Only import transcribers if their respective libraries are available; the
# names they provide are exported alongside the import so that `import *`
# doesn't reference a name that was never bound.
if importlib.util.find_spec("nemo"):
    from audio_segmentation.transcriber.nemo import (
        NemoTranscriber,
        NemoModel,
    )

    __all__ += ["NemoModel", "NemoTranscriber"]


if importlib.util.find_spec("whisperx"):
    from audio_segmentation.transcriber.whisperx import (
        WhisperxTranscriber,
        WhisperxModel,
    )

    __all__ += ["WhisperxModel", "WhisperxTranscriber"]


if importlib.util.find_spec("speechbrain"):
    from audio_segmentation.verifiers.speechbrain import SpeechBrainVerifier

    __all__ += ["SpeechBrainVerifier"]
