from enum import StrEnum
from typing import TypedDict

import numpy as np
import pydub
import whisper_timestamped as whisper

from audio_segmentation.segment import RawSegment
from audio_segmentation.transcriber.transcriber import Transcriber, TranscriptionResult


class WhisperModel(StrEnum):
    Tiny = 'tiny'
    Tiny_En = 'tiny.en'

    Small = 'small'
    Small_En = 'small.en'

    Medium = 'medium'
    Medium_En = 'medium.en'

    Turbo = 'turbo'
    Large_v2 = 'large-v2'
    Large_v3 = 'large-v3'


class Segment(TypedDict):
    start: float
    end: float
    text: str
    confidence: float | None


class SentenceSegment(Segment):
    start: float
    end: float
    text: str
    confidence: float | None
    words: list[Segment]


class WhisperTranscriptionResult(TypedDict):
    segments: list[SentenceSegment]
    text: str



class WhisperTimestampedTranscriber(Transcriber):
    def __init__(
        self,
        model_name: WhisperModel = WhisperModel.Large_v3,
        device: str = 'cuda',
        device_index: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device

        if device_index is not None:
            self.device = f"{device}:{device_index}"

        self.model = whisper.load_model(
            self.model_name.value,
            device=self.device,
        )

        super().__init__()

    @property
    def supports_word_level_segmentation(self) -> bool:
        return True

    @property
    def includes_punctuation(self) -> bool:
        return True

    def transcribe(
        self,
        audio_segment: pydub.AudioSegment,
        word_level_segmentation: bool = True,
        **kwargs,
    ) -> TranscriptionResult:
        data = np.array(audio_segment.get_array_of_samples())
        data = data.astype(np.float32) / np.iinfo(np.int16).max

        result: WhisperTranscriptionResult = whisper.transcribe_timestamped(
            self.model,
            data,
            beam_size=5,
            best_of=5,
            language='en',
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),

            # TODO: We may want to support detecting_disfluencies as it allegedly improves accuracy
            # The problem is, our naive sentence segmentation does not handle it well at all.
            # detect_disfluencies=True,
            vad=True,
            **kwargs,
        )

        segments: list[Segment] | list[SentenceSegment] = result['segments']

        # Flatten segments['words'] into a list of Segment
        if word_level_segmentation:
            segments = [
                word for segment in segments
                for word in segment['words']
            ]

        return TranscriptionResult(
            segments=[
                RawSegment(
                    start=segment['start'],
                    end=segment['end'],
                    text=segment['text'],
                ) for segment in segments
            ],
            transcript=result['text'],
        )
