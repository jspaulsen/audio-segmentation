import librosa
import numpy as np

from audio_segmentation.types.segment import Segment
from audio_segmentation.types.audio import Audio


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

    Returns:
        Segment: A new Segment with refined start and end timestamps.
    """
    naudio = Audio(data=audio, sr=sr)

    lookback = segment.start - search_boundary
    lookforward = segment.end + search_boundary
    nsegment = naudio[lookback:lookforward]

    predicted_start = detect_edge_energy(nsegment.data, nsegment.sr, hop_length=hop_length, frame_length=frame_length)
    predicted_end = detect_edge_energy(nsegment.data[::-1], nsegment.sr, hop_length=hop_length, frame_length=frame_length)  # reversed

    if predicted_start is not None:
        predicted_start = max(0, predicted_start - pad)

    if predicted_end is not None:
        predicted_end = max(0, predicted_end + pad)

    return Segment(
        start=lookback + predicted_start if predicted_start is not None else segment.start,
        end=lookforward - predicted_end if predicted_end is not None else segment.end,
        text=segment.text,
        speaker_id=segment.speaker_id
    )


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
