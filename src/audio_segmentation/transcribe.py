import io
import logging
from pathlib import Path
from typing import cast

import pydub

from audio_segmentation.segment import Segment
from audio_segmentation.segmenter import default_segmenter, sentence_segmenter
from audio_segmentation.transcriber.transcriber import Transcriber


logger = logging.getLogger(__name__)


DEFAULT_SEGMENT_LENGTH_MS = 60 * 60 * 1000  # 1 hour in milliseconds


def reformat_audio(audio: pydub.AudioSegment) -> pydub.AudioSegment:
    in_memory = io.BytesIO()
    audio.export(in_memory, format="wav")

    in_memory.seek(0)
    return pydub.AudioSegment.from_file(in_memory, format="wav")


# Takes an audio segment and converts it into list[Segment]
def segment_audio_segment(
    audio_segment: pydub.AudioSegment,
    transcriber: Transcriber,
    use_sentence_segmentation: bool = True,
    raise_exception: bool = False, # Only applies if word_level_segmentation is True
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
    )

    if use_sentence_segmentation:
        return sentence_segmenter(
            segmented_transcription=result,
            exception_if_no_words=raise_exception,
        )

    return default_segmenter(segments=result.segments)


# TODO: Rename me; this splits the audio into segments of a fixed length
# and then transcribes each segment; on the chance that the split 
# interrupts a sentence, it will try to fix that by using the n-1 location
# of the last segment as the end of the segment.
def segment_full_audio(
    full_audio: pydub.AudioSegment,
    transcriber: Transcriber,
    segment_length: int, # Length in milliseconds
    use_sentence_segmentation: bool = True,
    raise_exception: bool = False,  # Only applies if use_sentence_segmentation is True
) -> list[Segment]:
    total_length = len(full_audio)
    complete_segments: list[Segment] = []
    last_segment: bool = False
    start: int = 0

    while start < total_length and not last_segment:
        end = start + segment_length

        if end >= total_length:
            end = total_length

            # We're at the end of the audio.
            last_segment = True

        audio_segment: pydub.AudioSegment = cast(pydub.AudioSegment, full_audio[start:end])
        
        segments = segment_audio_segment(
            audio_segment=audio_segment,
            transcriber=transcriber,
            use_sentence_segmentation=use_sentence_segmentation,
            raise_exception=raise_exception,
        )

        logger.debug(
            f"Transcribed segment from {start / 1000:.2f}s to {end / 1000:.2f}s, found {len(segments)} segments.",
            extra={
                'start': start,
                'end': end,
                'segment_length': segment_length,
                'total_length': total_length,
                'num_segments': len(segments),
                'last_segment': last_segment,
            },
        )

        # If, for whatever reason, we don't have any segments, we should just
        # skip this segment and move on to the next one.
        if not segments:
            if start + segment_length >= len(full_audio):
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
        if start + segment_length < len(full_audio) and len(segments) > 1:
            if not last_segment:
                segments.pop(-1)

        complete_segments.extend(segments)

        # Move the start to the end of the last segment
        start = segments[-1].end
    
    return complete_segments


def transcribe_audio(
    audio: Path | str,
    transcriber: Transcriber,
    segment_length: int | None = None,
    use_sentence_segmentation: bool = True,
    raise_exception: bool = False,  # Only applies if use_sentence_segmentation is True
) -> list[Segment]:
    """
    Transcribe the audio file using the specified transcriber.

    Args:
        audio (Path | str): Path to the audio file.
        transcriber (Transcriber): The transcriber to use for transcription.
        segment_length (int | None): Length of each segment in milliseconds.
            NOTE: Unless you have a specific reason to change this, you should leave this as None
                and let the transcriber decide the ideal segment length.
        use_sentence_segmentation (bool): Whether to perform word-level segmentation.
        raise_exception (bool): If True, raises an exception if no words are found in a segment
            when use_sentence_segmentation is True. Defaults to False.

    Returns:
        list[Segment]: List of segments with transcription results.
    """
    audio = Path(audio)
    segment_length = segment_length or transcriber.ideal_segment_length or DEFAULT_SEGMENT_LENGTH_MS
    
    if not audio.exists():
        raise FileNotFoundError(f"Audio file {audio} does not exist.")
    
    audio_segment: pydub.AudioSegment = pydub.AudioSegment.from_file(audio)
    
    # if the file isn't wav, convert it to wav
    if audio.suffix.lower() != ".wav":
        audio_segment = reformat_audio(audio_segment)

    audio_segment: pydub.AudioSegment = audio_segment.set_frame_rate(16000)
    audio_segment: pydub.AudioSegment = audio_segment.set_channels(1)
    audio_segment: pydub.AudioSegment = audio_segment.set_sample_width(2)

    return segment_full_audio(
        full_audio=audio_segment,
        transcriber=transcriber,
        segment_length=segment_length,
        use_sentence_segmentation=use_sentence_segmentation,
        raise_exception=raise_exception,
    )
