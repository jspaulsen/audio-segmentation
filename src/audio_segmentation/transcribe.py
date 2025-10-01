import io
import logging
from pathlib import Path
from typing import cast

import pydub

from audio_segmentation.types.segment import Segment
from audio_segmentation.segmenter import default_segmenter, sentence_segmenter
from audio_segmentation.transcriber.transcriber import Transcriber
from audio_segmentation.types.transcription import TranscriptionResult


logger = logging.getLogger(__name__)


DEFAULT_SEGMENT_LENGTH_MS = 60 * 60 * 1000  # 1 hour in milliseconds


# Takes an audio segment and converts it into list[Segment]
def segment_audio_segment(
    audio_segment: pydub.AudioSegment,
    transcriber: Transcriber,
    use_sentence_segmentation: bool = True,
    raise_exception_on_mismatch: bool = False, # Only applies if word_level_segmentation is True
    transcriber_kwargs: dict | None = None,
) -> list[Segment]:
    if use_sentence_segmentation and (not transcriber.supports_word_level_segmentation or not transcriber.includes_punctuation):
        logger.warning(
            "Transcriber does not support word-level segmentation or punctuation; using the default segmenter instead.",
            extra={
                'transcriber': transcriber.__class__.__name__,
                'supports_word_level_segmentation': transcriber.supports_word_level_segmentation,
                'includes_punctuation': transcriber.includes_punctuation,
            },
        )

        use_sentence_segmentation = False

    result = transcriber.transcribe(
        audio_segment=audio_segment,
        word_level_segmentation=use_sentence_segmentation,
        **(transcriber_kwargs or {}),
    )

    if use_sentence_segmentation:
        return sentence_segmenter(
            segmented_transcription=result,
            raise_exception_on_mismatch=raise_exception_on_mismatch,
        )

    return default_segmenter(segments=result.segments)


def transcribe_audio_segment(
    audio_segment: pydub.AudioSegment,
    transcriber: Transcriber,
    # segment_length: int | None = None,
    raise_exception_on_mismatch: bool = False,
    transcriber_kwargs: dict | None = None,
) -> list[Segment]:
    use_sentence_segmentation: bool = transcriber.supports_word_level_segmentation
    kwargs = transcriber_kwargs or {}

    result = transcriber.transcribe(
        audio_segment=audio_segment,
        **kwargs,
    )

    if use_sentence_segmentation:
        return sentence_segmenter(
            segmented_transcription=result,
            raise_exception_on_mismatch=raise_exception_on_mismatch,
        )

    return default_segmenter(segments=result.segments)


def transcribe_audio(
    audio: pydub.AudioSegment,
    transcriber: Transcriber,
    # segment_length: int | None = None,
    raise_exception_on_mismatch: bool = False,  # Only applies if use_sentence_segmentation is True
    transcriber_kwargs: dict | None = None,
) -> TranscriptionResult:
    audio_length = len(audio)
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

        audio_segment: pydub.AudioSegment = cast(pydub.AudioSegment, audio[start:end])
        segments = transcribe_audio_segment(
            audio_segment=audio_segment,
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
