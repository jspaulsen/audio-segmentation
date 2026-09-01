import logging

import librosa
import numpy as np

from audio_segmentation.types.segment import Segment
from audio_segmentation.types.audio import Audio


logger = logging.getLogger(__name__)


def detect_edge_energy(
    data: np.ndarray,
    sr: int,
    threshold_db: float = 40,
    hop_length: int = 128,
    frame_length: int = 1024,
) -> int | None:
    """
    Detects the edge based on when energy rises above a threshold relative to the peak.

    Args:
        data: Audio data.
        sr: Sample rate.
        threshold_db: The threshold in decibels below the peak to consider 'silence'.
            40dB is a good standard for speech (1% of peak amplitude).
            Higher values (e.g., 50) are more sensitive, Lower (e.g., 30) cut more audio.

    Returns:
        int | None: The timestamp in milliseconds where the edge is detected, or None if no edge is found.
    """
    if len(data) == 0:
        return None

    rms = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)[0]

    if len(rms) == 0:
        return None

    # Convert to Decibels relative to the peak of this specific segment
    max_rms = np.max(rms)

    if max_rms == 0:
        return None

    rms_db = librosa.amplitude_to_db(rms, ref=max_rms)

    # Find the first frame that exceeds the threshold (e.g., > -40dB)
    mask = rms_db > -threshold_db

    # np.argmax on a boolean array returns the index of the first True
    if not np.any(mask):
        return None # The whole clip is silence

    first_frame_index = np.argmax(mask)

    # Convert frame index back to time (ms)
    sample_index = librosa.frames_to_samples(first_frame_index, hop_length=hop_length)
    return int((sample_index / sr) * 1000)


def refine_segment_timestamps(
    audio: np.ndarray,
    sr: int,
    segment: Segment,
    search_boundary: int = 200,
    hop_length: int = 128,
    frame_length: int = 1024,
    pad: int = 0,
    previous_segment: Segment | None = None,
    next_segment: Segment | None = None,
) -> Segment:
    """
    Attempts to refine the start and end timestamps of a segment based on silence detection.

    Args:
        audio (np.ndarray): The audio data.
        sr (int): Sample rate of the audio data.
        segment (Segment): The segment with initial start and end timestamps.
        search_boundary (int): Maximum milliseconds to look back or ahead for silence.
        hop_length (int): Hop length for the short-time Fourier transform.
        frame_length (int): Frame length for the short-time Fourier transform.
        pad (int): Padding in milliseconds to apply before and after the refined timestamps if needed.
            Default is 0.
        previous_segment (Segment | None): The segment immediately before this one, if any. The
            backwards search is clamped to the midpoint of the gap between the two so that the
            previous segment's speech never enters the search window.
        next_segment (Segment | None): The segment immediately after this one, if any. Clamped the
            same way as `previous_segment`.

    Returns:
        Segment: A new Segment with refined start and end timestamps.

    Note:
        On densely packed audio a `search_boundary` wider than the gap to a neighbour would
        otherwise pull the refined boundary across that gap and onto the neighbour's speech.
        Pass the neighbours -- or use `refine_segment_timestamps_batch`, which wires them up --
        to prevent that. Each clamp is logged at WARNING.
    """
    naudio = Audio(data=audio, sr=sr)

    # Clamp to the bounds of the audio; a negative lookback would be read as an
    # index from the end, and a lookforward past the end would be silently
    # truncated by the slice while still being used as the reference point below.
    #
    # Then clamp again to the midpoint of the gap to each neighbour. Stopping at
    # the midpoint rather than at the neighbour's boundary keeps the neighbour's
    # speech out of the window entirely -- detect_edge_energy would otherwise fire
    # on it immediately and snap this segment's edge across the gap -- and
    # guarantees two adjacent refined segments cannot overlap.
    #
    # The outer min/max pin the window to the segment's own bounds so that an
    # already-overlapping neighbour collapses the search rather than inverting it.
    floor = 0 if previous_segment is None else (previous_segment.end + segment.start) // 2
    ceiling = len(naudio) if next_segment is None else (segment.end + next_segment.start) // 2

    lookback = min(segment.start, max(0, segment.start - search_boundary, floor))
    lookforward = max(segment.end, min(len(naudio), segment.end + search_boundary, ceiling))

    if previous_segment is not None and lookback > max(0, segment.start - search_boundary):
        logger.warning(
            "search_boundary of %dms exceeds the %dms gap to the previous segment; "
            "clamped the start search to %dms (segment starts at %dms).",
            search_boundary,
            segment.start - previous_segment.end,
            lookback,
            segment.start,
        )

    if next_segment is not None and lookforward < min(len(naudio), segment.end + search_boundary):
        logger.warning(
            "search_boundary of %dms exceeds the %dms gap to the next segment; "
            "clamped the end search to %dms (segment ends at %dms).",
            search_boundary,
            next_segment.start - segment.end,
            lookforward,
            segment.end,
        )

    nsegment = naudio[lookback:lookforward]

    predicted_start = detect_edge_energy(nsegment.data, nsegment.sr, hop_length=hop_length, frame_length=frame_length)
    predicted_end = detect_edge_energy(nsegment.data[::-1], nsegment.sr, hop_length=hop_length, frame_length=frame_length)  # reversed

    if predicted_start is not None:
        predicted_start = max(0, predicted_start - pad)

    # predicted_end is a distance measured back from lookforward, so padding the
    # end of the segment means shrinking it.
    if predicted_end is not None:
        predicted_end = max(0, predicted_end - pad)

    start = lookback + predicted_start if predicted_start is not None else segment.start
    end = lookforward - predicted_end if predicted_end is not None else segment.end

    # `pad` can push a boundary back outside the window we actually searched, which
    # would undo the clamping above and let padded neighbours overlap again. Confine
    # the result to the window; inside it, `pad` is unaffected.
    return Segment(
        start=min(max(start, lookback), lookforward),
        end=max(min(end, lookforward), lookback),
        text=segment.text,
        speaker_id=segment.speaker_id
    )


def refine_segment_timestamps_batch(
    audio: np.ndarray,
    sr: int,
    segments: list[Segment],
    search_boundary: int = 200,
    hop_length: int = 128,
    frame_length: int = 1024,
    pad: int = 0,
) -> list[Segment]:
    """
    Refines the timestamps of every segment in a list, giving each one its neighbours.

    This is the preferred entry point over calling `refine_segment_timestamps` in a loop: it
    supplies each segment's neighbours so the search window is clamped to the gaps between
    them, which keeps adjacent refined segments from overlapping or absorbing each other's
    speech on densely packed audio.

    Segments are assumed to already be in chronological order and are not re-sorted. Each
    segment is refined against the *original* neighbours rather than the already-refined ones,
    so the result does not depend on the order the list is walked.

    Args:
        audio (np.ndarray): The audio data.
        sr (int): Sample rate of the audio data.
        segments (list[Segment]): Segments to refine, in chronological order.
        search_boundary (int): Maximum milliseconds to look back or ahead for silence.
        hop_length (int): Hop length for the short-time Fourier transform.
        frame_length (int): Frame length for the short-time Fourier transform.
        pad (int): Padding in milliseconds to apply before and after the refined timestamps.

    Returns:
        list[Segment]: The refined segments, in the same order as the input.
    """
    return [
        refine_segment_timestamps(
            audio=audio,
            sr=sr,
            segment=segment,
            search_boundary=search_boundary,
            hop_length=hop_length,
            frame_length=frame_length,
            pad=pad,
            previous_segment=segments[index - 1] if index > 0 else None,
            next_segment=segments[index + 1] if index + 1 < len(segments) else None,
        )
        for index, segment in enumerate(segments)
    ]


def refine_sentence_segments(
    segments: list[Segment],
    merge_threshold_ms: int = 500,
    max_segment_length_ms: int | None = None,
) -> list[Segment]:
    """
    Refines a list of sentence segments by merging segments that are close together.

    Args:
        segments (list[Segment]): List of segments to refine.
        merge_threshold_ms (int): Maximum gap in milliseconds between segments to consider merging.
        max_segment_length_ms (int | None): Optional maximum length for a segment. If merging
            two segments would exceed this length, they will not be merged.

    Returns:
        list[Segment]: Refined list of segments.
    """
    if not segments:
        return []

    current_segment: Segment | None = None
    refined_segments: list[Segment] = []

    for segment in segments:
        if current_segment is None:
            current_segment = segment
            continue

        identifiers = (
            current_segment.speaker_id is not None,
            segment.speaker_id is not None,
        )

        # If either of them are not null or both are not null but different, do not merge
        if any(identifiers) and not all(identifiers):
            refined_segments.append(current_segment)
            current_segment = segment
            continue

        if current_segment.speaker_id != segment.speaker_id:
            refined_segments.append(current_segment)
            current_segment = segment
            continue

        gap = segment.start - current_segment.end

        # If the gap is larger than the merge threshold, or if merging would exceed max length, finalize current segment
        if gap > merge_threshold_ms or (max_segment_length_ms is not None and current_segment.duration + segment.duration > max_segment_length_ms):
            refined_segments.append(current_segment)
            current_segment = segment
            continue

        # Otherwise, merge the segments
        current_segment = current_segment.combine(segment)

    if current_segment is not None:
        refined_segments.append(current_segment)

    return refined_segments
