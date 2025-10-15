import logging
from typing import cast

import numpy as np

from audio_segmentation.types.segment import Segment
from audio_segmentation.segmenter import default_segmenter, sentence_segmenter
from audio_segmentation.transcriber.transcriber import Transcriber
from audio_segmentation.types.transcription import TranscriptionResult
from audio_segmentation.utility import resample, convert_to_mono


logger = logging.getLogger(__name__)


DEFAULT_SEGMENT_LENGTH_MS = 60 * 60 * 1000  # 1 hour in milliseconds


def transcribe_audio_segment(
    audio: np.ndarray,
    sr: int,
    transcriber: Transcriber,
    # segment_length: int | None = None,
    raise_exception_on_mismatch: bool = False,
    transcriber_kwargs: dict | None = None,
) -> list[Segment]:
    use_sentence_segmentation: bool = transcriber.supports_word_level_segmentation
    kwargs = transcriber_kwargs or {}

    result = transcriber.transcribe(
        audio=audio,
        sr=sr,
        **kwargs,
    )

    if use_sentence_segmentation:
        return sentence_segmenter(
            segmented_transcription=result,
            raise_exception_on_mismatch=raise_exception_on_mismatch,
        )

    return default_segmenter(segments=result.segments)


def transcribe_audio(
    audio: np.ndarray,
    sr: int,
    transcriber: Transcriber,
    # segment_length: int | None = None,
    raise_exception_on_mismatch: bool = False,  # Only applies if use_sentence_segmentation is True
    transcriber_kwargs: dict | None = None,
) -> TranscriptionResult:
    audio_length = len(audio) * 1000 // sr  # in milliseconds
    segment_length: int = transcriber.ideal_segment_length or DEFAULT_SEGMENT_LENGTH_MS
    complete_segments: list[Segment] = []
    last_segment: bool = False
    start: int = 0

    while start < audio_length and not last_segment:
        end = start + segment_length

        if end >= audio_length:
            end = audio_length

            # We're at the end of the audio.
            last_segment = True

        adjusted_start = int(start * sr / 1000)
        adjusted_end = int(end * sr / 1000)
        audio_segment = audio[adjusted_start:adjusted_end]

        # if the segment sample_rate does not match that of the transcriber, we need to resample it
        if transcriber.required_sample_rate and sr != transcriber.required_sample_rate:
            audio_segment = resample(
                audio_segment,
                original_sr=sr,
                target_sr=transcriber.required_sample_rate,
            )

        # if the transcriber requires mono audio, we need to convert it
        if transcriber.requires_mono_audio:
            audio_segment = convert_to_mono(audio_segment)

        segments = transcribe_audio_segment(
            audio=audio_segment,
            sr=transcriber.required_sample_rate or sr,
            transcriber=transcriber,
            raise_exception_on_mismatch=raise_exception_on_mismatch,
            transcriber_kwargs=transcriber_kwargs or {},
        )

        logger.debug(
            f"Transcribed segment from {start / 1000:.2f}s to {end / 1000:.2f}s, found {len(segments)} segments.",
            extra={
                'start': start,
                'end': end,
                'segment_length': segment_length,
                'total_length': audio_length,
                'num_segments': len(segments),
                'last_segment': last_segment,
            },
        )

        # If, for whatever reason, we don't have any segments, we should just
        # skip this segment and move on to the next one.
        if not segments:
            if start + segment_length >= audio_length:
                break

            start = end
            continue

        # The segment timestamps are relative to the provided segment, so we add the start time
        # of the segment to each segment to make them absolute.
        for segment in segments:
            segment.start = segment.start + start
            segment.end = segment.end + start

        # If there's more segments to come, we should pop off the last segment
        # as it's likely to be a partial sentence
        if start + segment_length < audio_length and len(segments) > 1:
            if not last_segment:
                segments.pop(-1)

        complete_segments.extend(segments)

        # Move the start to the end of the last segment
        start = segments[-1].end

    return TranscriptionResult(
        transcript=" ".join([segment.text for segment in complete_segments]),
        segments=complete_segments,
    )
