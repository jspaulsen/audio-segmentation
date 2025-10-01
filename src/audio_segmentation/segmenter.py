import logging

from audio_segmentation.types.segment import RawSegment, Segment
from audio_segmentation.transcriber.transcriber import RawTranscriptionResult

import nltk


logger = logging.getLogger(__name__)


class SegmentationException(Exception):
    def __init__(
        self,
        transcription: RawTranscriptionResult,
        message: str = "An exception occurred during segmentation",
        *args,
        **kwargs
    ):
        super().__init__(message, *args, **kwargs)
        self.message = message
        self.transcription = transcription


def transform_word(word: str) -> str:
    """
    Transform the word to a standard format
    """
    return (
        word
            .strip("-.,!?\"'")
            .replace("'", " ")
            .replace('⁇', "")
            .lower()
            .strip()
    )


def transform_sentence(sentence: str) -> str:
    return (
        sentence
            .replace('⁇', "")
    )


def sentence_segmenter(
    segmented_transcription: RawTranscriptionResult,
    raise_exception_on_mismatch: bool = True
) -> list[Segment]:
    transcription = segmented_transcription.transcript.strip()
    sentences = nltk.tokenize.sent_tokenize(transcription)
    words: list[RawSegment] = segmented_transcription.segments
    segments: list[Segment] = []

    current_word_index = 0

    for sentence in sentences:
        sentence_words =  transform_sentence(sentence).split()

        sentence_len = len(sentence_words)
        matched_words = []

        while current_word_index < len(words) and len(matched_words) < sentence_len:
            segment = words[current_word_index]
            word = transform_word(segment.text)
            expected_word = transform_word(sentence_words[len(matched_words)])

            # Try peeking ahead and combining words if they don't match
            # if they do match, build a new segment and increment the index
            if word != expected_word:
                peek = current_word_index + 1

                if peek > len(words) - 1:
                    break

                peek_segment = words[peek]
                peek_word = transform_word(peek_segment.text)

                # This is a little flaky of a match; if _both_ word and peek are within the
                # expected word, we can combine them into a single segment
                if word in expected_word and peek_word in expected_word:
                    segment = RawSegment(
                        start=segment.start,
                        end=peek_segment.end,
                        text=segment.text + peek_segment.text
                    )

                    matched_words.append(segment)
                    current_word_index = current_word_index + 1
                else:
                    logger.error(f"Warning: Word '{word}' does not match expected word '{expected_word}' in sentence: {sentence}")
            else:
                matched_words.append(segment)

            current_word_index = current_word_index + 1

        if matched_words:
            starting_word: RawSegment = matched_words[0]
            ending_word: RawSegment = matched_words[-1]

            if starting_word.start is None or ending_word.end is None:
                logger.warning(f"Missing start or end time for sentence: {sentence}.")
                continue

            start_time: float = starting_word.start
            end_time: float = ending_word.end

            segments.append(
                Segment(
                    start=int(start_time * 1000),  # Convert to milliseconds
                    end=int(end_time * 1000),  # Convert to milliseconds
                    text=sentence
                )
            )
        else:
            logger.exception(f"No matching words found for sentence: {sentence}")

            if raise_exception_on_mismatch:
                raise SegmentationException(
                    transcription=segmented_transcription,
                    message=f"Mismatch in sentence segmentation for: {sentence}"
                )

    return segments


def default_segmenter(segments: list[RawSegment]) -> list[Segment]:
    """
    Default segmenter that returns the segments as they are.

    NOTE: This only works for sentence level segments. It does not handle word level segmentation.
    """
    return [
        Segment(
            start=int(segment.start * 1000),  # Convert to milliseconds
            end=int(segment.end * 1000),  # Convert to milliseconds
            text=segment.text
        ) for segment in segments if segment.start is not None and segment.end is not None
    ]
