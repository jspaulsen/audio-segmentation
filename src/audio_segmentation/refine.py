import pydub
from pydub.silence import detect_silence

from audio_segmentation.segment import Segment


def refine_start(
    audio: pydub.AudioSegment,
    initial_start_ms: int,
    max_lookback_ms: int,
    min_silence_len: int,
    silence_thresh: int,
    padding: int = 0,
) -> int:
    lookback_start = max(0, initial_start_ms - max_lookback_ms)
    segment_to_check = audio[lookback_start:initial_start_ms]
    silences: list[list[int]] = detect_silence(segment_to_check, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    if silences: # Find the last silence that ends just before the original start
        return lookback_start + silences[-1][1] - padding

    # If no silence was found, return the original
    return initial_start_ms


def refine_end(
    audio: pydub.AudioSegment,
    initial_end_ms: int,
    max_lookahead_ms: int,
    min_silence_len: int,
    silence_thresh: int,
    padding: int = 0,
) -> int:
    lookahead_end = min(len(audio), initial_end_ms + max_lookahead_ms)
    segment_to_check = audio[initial_end_ms:lookahead_end]
    silences: list[list[int]] = detect_silence(segment_to_check, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    if silences: # Snap to the start of the first silence found
        return initial_end_ms + silences[0][0] + padding

    return initial_end_ms


def refine_segment_timestamps(
    audio: pydub.AudioSegment,
    segment: Segment,
    max_look_ms: int = 100,
    min_silence_len: int = 10,
    silence_thresh: int = -40,
    padding: int = 0,
) -> Segment:
    """
    Attempts to refine the start and end timestamps of a segment based on silence detection.

    Args:
        audio (pydub.AudioSegment): The audio segment to analyze.
        segment (Segment): The segment with initial start and end timestamps.
        max_look_ms (int): Maximum milliseconds to look back or ahead for silence.
        min_silence_len (int): Minimum length of silence to consider it valid.
        silence_thresh (int): Silence threshold in dBFS.
        padding (int): Padding in milliseconds to apply after refining.

    Returns:
        Segment: A new Segment with refined start and end timestamps.
    """
    start = refine_start(
        audio,
        segment.start,
        max_lookback_ms=max_look_ms,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        padding=padding,
    )

    end = refine_end(
        audio,
        segment.end,
        max_lookahead_ms=max_look_ms,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        padding=padding,
    )

    return Segment(start=start, end=end, text=segment.text)
