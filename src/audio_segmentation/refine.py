import librosa
import numpy as np

from audio_segmentation.types.segment import Segment
from audio_segmentation.types.audio import Audio


def detect_silence_librosa(
    audio_data: np.ndarray,
    sr: int,
    silence_thresh_db: float,
    min_silence_len_ms: int,
) -> list[list[int]]:
    non_silent_intervals = librosa.effects.split(
        y=audio_data,
        top_db=abs(silence_thresh_db), # Librosa expects positive dB relative to max
        frame_length=2048,
        hop_length=512
    )

    # Detect non-silent intervals
    # top_db: The threshold (in decibels) below reference to consider as silence
    # ref: The reference amplitude (default is np.max)
    # frame_length & hop_length: Determine time resolution

    # Invert intervals to find silence
    silence_intervals = []
    prev_end = 0

    for start, end in non_silent_intervals:
        if start > prev_end:  # Found a gap (silence)
            silence_intervals.append([prev_end, start])
        prev_end = end

    # Check for trailing silence after the last sound
    if prev_end < len(audio_data):
        silence_intervals.append([prev_end, len(audio_data)])

    # 3. Convert samples to milliseconds and filter by minimum length
    silence_ms = []
    min_samples = (min_silence_len_ms * sr) // 1000

    for start, end in silence_intervals:
        if (end - start) >= min_samples:
            start_ms = int(start * 1000 / sr)
            end_ms = int(end * 1000 / sr)
            silence_ms.append([start_ms, end_ms])

    return silence_ms


# TODO: Can we replace this function with an existing library function?
# def detect_silence(
#     audio: Audio,
#     min_silence_len: int,
#     silence_thresh: int,
# ) -> list[list[int]]:
#     """
#     Detect silence in audio based on RMS energy threshold.

#     Args:
#         audio (Audio): The audio to analyze.
#         min_silence_len (int): Minimum length of silence in milliseconds.
#         silence_thresh (int): Silence threshold in dBFS.

#     Returns:
#         list[list[int]]: List of [start_ms, end_ms] pairs for silent regions.
#     """
#     audio_data = audio.data

#     # Convert dBFS to amplitude (assuming audio is normalized to [-1, 1])
#     # dBFS = 20 * log10(amplitude)
#     # amplitude = 10^(dBFS/20)
#     amplitude_thresh = 10 ** (silence_thresh / 20.0)

#     # Handle both mono and stereo audio
#     if audio_data.ndim > 1:
#         audio_data = np.mean(np.abs(audio_data), axis=0)  # For multi-channel, take the mean across channels
#     else:
#         audio_data = np.abs(audio_data)

#     # Create boolean mask where audio is below threshold (silence)
#     is_silent = audio_data < amplitude_thresh

#     # Find contiguous regions of silence
#     # Pad with False to detect edges properly
#     padded = np.pad(is_silent, (1, 1), constant_values=False)
#     diff = np.diff(padded.astype(int))

#     # Find starts and ends of silent regions
#     starts = np.where(diff == 1)[0]
#     ends = np.where(diff == -1)[0]

#     # Convert sample indices to milliseconds and filter by minimum length
#     silences = []
#     min_silence_samples = int(min_silence_len * audio.sr / 1000)

#     for start_idx, end_idx in zip(starts, ends):
#         if end_idx - start_idx >= min_silence_samples:
#             start_ms = int(start_idx * 1000 / audio.sr)
#             end_ms = int(end_idx * 1000 / audio.sr)
#             silences.append([start_ms, end_ms])

#     return silences


def estimate_min_silence_len(
    audio: Audio,
    silence_thresh: int,
    min_silence_len: int = 50,
    percentile: float = 25.0,
) -> int:
    silences = detect_silence_librosa(
        audio.data,
        sr=audio.sr,
        silence_thresh_db=silence_thresh,
        min_silence_len_ms=min_silence_len
    )

    if not silences:
        return 150  # fallback

    # Calculate silence durations
    durations = [end - start for start, end in silences]

    # Use the median silence duration, or 25th percentile
    # This filters out very brief noise while catching real pauses
    min_silence = int(np.percentile(durations, percentile))

    # Clamp to reasonable range
    return max(100, min(min_silence, 500))


def estimate_thresholds(
    audio: np.ndarray,
    sr: int,
    silence_percentile: float = 10.0,
    minimum_silence_percentile: float = 25.0,
) -> tuple[int, int]:
    """
    Estimate silence detection parameters from audio characteristics.

    Analyzes the audio's amplitude distribution to automatically determine
    appropriate silence threshold and minimum silence duration. This is useful
    when you don't know the recording conditions ahead of time.

    The silence threshold is computed by taking a low percentile of non-zero
    amplitudes and converting to dBFS. The minimum silence length is then
    estimated by detecting silences at that threshold and taking a percentile
    of their durations.

    Args:
        audio: The audio waveform as a numpy array (mono or multi-channel).
        sr: Sample rate in Hz.
        silence_percentile: Percentile of amplitude distribution to use as
            silence threshold. Lower values (e.g., 5.0) are more conservative,
            only treating very quiet samples as silence. Higher values (e.g., 20.0)
            are more aggressive. Default is 10.0.
        minimum_silence_percentile: Percentile of detected silence durations to
            use as minimum silence length. Lower values capture shorter pauses,
            higher values require longer pauses. Default is 25.0.

    Returns:
        A tuple of (min_silence_len_ms, silence_thresh_dbfs) where:
        - min_silence_len_ms: Minimum silence duration in milliseconds (clamped to 100-500ms)
        - silence_thresh_dbfs: Silence threshold in dBFS (negative value)
    """
    data = np.abs(audio.flatten())

    silence_level = np.percentile(data[data > 0], silence_percentile)
    silence_thresh = int(20 * np.log10(silence_level + 1e-10))

    min_silence_len = estimate_min_silence_len(
        Audio(data=audio, sr=sr),
        silence_thresh,
        percentile=minimum_silence_percentile,
    )

    return min_silence_len, silence_thresh


def refine_start(
    audio: Audio,
    initial_start_ms: int,
    max_lookback_ms: int,
    min_silence_len: int,
    silence_thresh: int,
) -> int:
    """
    Refine the start timestamp of a segment by looking back for silence.

    Args:
        audio (Audio): The audio to analyze.
        initial_start_ms (int): Initial start time in milliseconds.
        max_lookback_ms (int): Maximum milliseconds to look back for silence.
        min_silence_len (int): Minimum length of silence to consider it valid.
        silence_thresh (int): Silence threshold in dBFS.

    Returns:
        int: Refined start time in milliseconds.
    """
    lookback_start = max(0, initial_start_ms - max_lookback_ms)
    segment_to_check = audio[lookback_start:initial_start_ms]
    silences: list[list[int]] = detect_silence_librosa(
        segment_to_check.data,
        sr=segment_to_check.sr,
        silence_thresh_db=silence_thresh,
        min_silence_len_ms=min_silence_len
    )

    if silences:  # Find the last silence that ends just before the original start
        return lookback_start + silences[-1][1]

    # If no silence was found, return the original
    return initial_start_ms


def refine_end(
    audio: Audio,
    initial_end_ms: int,
    max_lookahead_ms: int,
    min_silence_len: int,
    silence_thresh: int,
) -> int:
    """
    Refine the end timestamp of a segment by looking ahead for silence.

    Args:
        audio (Audio): The audio to analyze.
        initial_end_ms (int): Initial end time in milliseconds.
        max_lookahead_ms (int): Maximum milliseconds to look ahead for silence.
        min_silence_len (int): Minimum length of silence to consider it valid.
        silence_thresh (int): Silence threshold in dBFS.
        padding (int): Padding in milliseconds to apply after refining.

    Returns:
        int: Refined end time in milliseconds.
    """
    lookahead_end = min(len(audio), initial_end_ms + max_lookahead_ms)
    segment_to_check = audio[initial_end_ms:lookahead_end]
    silences: list[list[int]] = detect_silence_librosa(
        segment_to_check.data,
        sr=segment_to_check.sr,
        silence_thresh_db=silence_thresh,
        min_silence_len_ms=min_silence_len
    )

    if silences:  # Snap to the start of the first silence found
        return initial_end_ms + silences[0][0]

    return initial_end_ms


def refine_segment_timestamps(
    audio: np.ndarray,
    sr: int,
    segment: Segment,
    min_silence_len: int,
    silence_thresh: int,
    max_look_ms: int = 75,
) -> Segment:
    """
    Attempts to refine the start and end timestamps of a segment based on silence detection.

    It's recommended to use `estimate_thresholds` to calculate `min_silence_len` and `silence_thresh` globally or on a known "quiet" region,
    rather than estimating them on the active speech segment itself.

    Args:
        audio (np.ndarray): The audio data.
        sr (int): Sample rate of the audio data.
        segment (Segment): The segment with initial start and end timestamps.
        min_silence_len (int): Minimum length of silence to consider it valid.
        silence_thresh (int): Silence threshold in dBFS.
        max_look_ms (int): Maximum milliseconds to look back or ahead for silence.

    Returns:
        Segment: A new Segment with refined start and end timestamps.
    """
    naudio = Audio(data=audio, sr=sr)

    start = refine_start(
        naudio,
        segment.start,
        max_lookback_ms=max_look_ms,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    end = refine_end(
        naudio,
        segment.end,
        max_lookahead_ms=max_look_ms,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    return Segment(start=start, end=end, text=segment.text, speaker_id=segment.speaker_id)


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
