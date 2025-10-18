from enum import StrEnum
import warnings

import numpy as np
from transformers import Wav2Vec2ForCTC
import whisperx
from whisperx.asr import FasterWhisperPipeline

from audio_segmentation.types.segment import RawSegment
from audio_segmentation.transcriber.transcriber import Transcriber, RawTranscriptionResult


warnings.filterwarnings("ignore", module="pyannote")


class WhisperxModel(StrEnum):
    Tiny = 'tiny'
    Tiny_En = 'tiny.en'

    Small = 'small'
    Small_En = 'small.en'

    Medium = 'medium'
    Medium_En = 'medium.en'

    Turbo = 'turbo'
    Large_v3 = 'large-v3'

    DistilSmall_En = 'distill-small.en'
    DistilMedium_En = 'distill-medium.en'
    DistilLarge_v3 = 'distill-large-v3'


class WhisperxTranscriber(Transcriber):
    def __init__(
        self,
        model_name: WhisperxModel = WhisperxModel.Large_v3,
        device: str = 'cuda',
        device_index: int = 0,
        batch_size: int = 4,
        compute_type: str = 'float16',
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.align_device = device
        self.batch_size = batch_size

        if device_index is not None:
            self.align_device = f"{device}:{device_index}"

        self.model: FasterWhisperPipeline = whisperx.load_model(
            self.model_name.value,
            device=device,
            device_index=device_index,
            language='en',
            compute_type=compute_type,
        )

        aligner, metadata = whisperx.load_align_model(
            language_code='en',
            device=self.align_device,
        )

        self.aligner: Wav2Vec2ForCTC = aligner
        self.metadata: dict = metadata

        super().__init__()

    @property
    def supports_word_level_segmentation(self) -> bool:
        return True

    @property
    def required_sample_rate(self) -> int:
        return 16000

    @property
    def requires_mono_audio(self) -> bool:
        return True

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int,
        # word_level_segmentation: bool = True,
        **kwargs,
    ) -> RawTranscriptionResult:
        """
        Transcribes the given audio segment using WhisperX.

        Args:
            audio_segment (pydub.AudioSegment): The audio segment to transcribe.
            word_level_segmentation (bool): Whether to return word-level segmentation.

        Returns:
            list[Segment]: A list of segments containing the transcription.
        """
        # key = 'word_segments' if word_level_segmentation else 'segments'
        key = 'segments'

        # key = 'word_segments' if word_level_segmentation else 'segments'
        field = 'word' if key == 'word_segments' else 'text'

        transcription_result = self.model.transcribe(
            audio,
            batch_size=self.batch_size,
            language='en',
            **kwargs,
        )

        aligned = whisperx.align(
            transcription_result['segments'],
            self.aligner,
            self.metadata,
            audio,
            device=self.device,
        )

        transcription = " ".join([seg["text"] for seg in aligned["segments"]])
        raw_segments: list[RawSegment] = []

        # segments and word_segments
        for raw_segment in aligned[key]:
            raw_segments.append(
                RawSegment(
                    start=raw_segment['start'],
                    end=raw_segment['end'],
                    text=raw_segment[field],
                )
            )

        return RawTranscriptionResult(
            transcript=transcription,
            segments=raw_segments,
        )
