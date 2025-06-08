import logging

from audio_segmentation.segment import RawSegment, Segment
from audio_segmentation.transcriber.transcriber import TranscriptionResult


logger = logging.getLogger(__name__)


def transform_word(word: str) -> str:
    """
    Transform the word to a standard format
    """
    return (
        word
            .strip("-.,!?\"'")
            .replace("'", " ")
            .lower()
    )


# def sentence_segmenter(transcription: str, words: list[RawSegment]) -> list[Segment]:
def sentence_segmenter(
    segmented_transcription: TranscriptionResult,
    exception_if_no_words: bool = True
) -> list[Segment]:
    import nltk
    nltk.download('punkt_tab', quiet=True)

    transcription = segmented_transcription.transcript.strip()
    sentences = nltk.tokenize.sent_tokenize(transcription)
    words: list[RawSegment] = segmented_transcription.segments
    segments: list[Segment] = []

    current_word_index = 0

    for sentence in sentences:
        sentence_words =  sentence.split()

        # split any dashed words
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

            if exception_if_no_words:
                raise ValueError(f"No matching words found for sentence: {sentence}")

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
