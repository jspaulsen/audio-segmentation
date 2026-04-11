import logging

import numpy as np

from audio_segmentation.aligner.aligner import Aligner
from audio_segmentation.refine import refine_segment_timestamps
from audio_segmentation.types.audio import Audio
from audio_segmentation.types.segment import Segment
from audio_segmentation.transcriber.transcriber import Transcriber
from audio_segmentation.types.transcription import TranscriptionResult
from audio_segmentation.verifiers.speaker import SpeakerIdentifier
from audio_segmentation.verifiers.verifier import SpeakerVerifier


logger = logging.getLogger(__name__)


DEFAULT_SEGMENT_LENGTH_MS = 60 * 60 * 1000  # 1 hour in milliseconds
ALIGNMENT_SEARCH_BOUNDARY_MS = 200  # Look 200ms before/after for silence
ALIGNMENT_PAD_MS = 50  # Add 50ms padding after refinement


def _align_segment(
    audio: Audio,
    segment: Segment,
    aligner: Aligner,
) -> list[Segment]:
    """
    Align a segment to get word-level boundaries.

    Uses silence detection to refine boundaries before alignment.
    """
    # Refine segment boundaries using silence detection
    refined = refine_segment_timestamps(
        audio=audio.to_numpy(),
        sr=audio.sr,
        segment=segment,
        search_boundary=ALIGNMENT_SEARCH_BOUNDARY_MS,
        pad=ALIGNMENT_PAD_MS,
    )

    # Extract audio for the refined segment
    segment_audio = audio[refined.start:refined.end]

    # Resample if needed for aligner
    if aligner.required_sample_rate and segment_audio.sr != aligner.required_sample_rate:
        segment_audio = segment_audio.resample(target_sr=aligner.required_sample_rate)

    # Run alignment
    try:
        raw_segments = aligner.align(
            audio=segment_audio.to_numpy(),
            sr=segment_audio.sr,
            transcript=refined.text,
        )
    except Exception as e:
        logger.warning(f"Alignment failed for segment '{refined.text[:50]}...': {e}")
        return [segment]  # Return original segment on failure

    if not raw_segments:
        return [segment]  # Return original if no alignment results

    # Convert to Segment with absolute timestamps
    word_segments = []
    for raw in raw_segments:
        if raw.start is None or raw.end is None:
            continue
        word_segments.append(
            Segment(
                start=refined.start + int(raw.start * 1000),
                end=refined.start + int(raw.end * 1000),
                text=raw.text,
                speaker_id=segment.speaker_id,  # Inherit speaker from parent
            )
        )

    return word_segments if word_segments else [segment]


def transcribe_audio(
    audio: np.ndarray,
    sr: int,
    transcriber: Transcriber,
    aligner: Aligner | None = None,
    speaker_verifier: SpeakerVerifier | None = None,
    similarity_threshold: float = 0.25,
    transcriber_kwargs: dict | None = None,
) -> TranscriptionResult:
    naudio = Audio(data=audio, sr=sr)
    audio_length = len(naudio)  # in milliseconds
    segment_length: int = transcriber.ideal_segment_length or DEFAULT_SEGMENT_LENGTH_MS
    complete_segments: list[Segment] = []
    last_segment: bool = False
    start: int = 0

    # Setup speaker identifier if we have a verifier
    speaker_identifier: SpeakerIdentifier | None = None

    if speaker_verifier:
        speaker_identifier = SpeakerIdentifier(
            verifier=speaker_verifier,
            similarity_threshold=similarity_threshold,
        )

    # Segment the whole audio into chunks of segment_length and transcribe each chunk.
    while start < audio_length and not last_segment:
        end = start + segment_length

        if end >= audio_length:
            end = audio_length

            # We're at the end of the audio.
            last_segment = True

        audio_segment: Audio = naudio[start:end]

        # if the segment sample_rate does not match that of the transcriber, we need to resample it
        if transcriber.required_sample_rate and sr != transcriber.required_sample_rate:
            audio_segment = audio_segment.resample(target_sr=transcriber.required_sample_rate)

        # if the transcriber requires mono audio, we need to convert it
        if transcriber.requires_mono_audio:
            audio_segment = audio_segment.to_mono()

        result = transcriber.transcribe(
            audio=audio_segment.to_numpy(),
            sr=audio_segment.sr,
            **(transcriber_kwargs or {}),
        )

        logger.debug(
            f"Transcribed segment from {start / 1000:.2f}s to {end / 1000:.2f}s, found {len(result.segments)} segments.",
            extra={
                'start': start,
                'end': end,
                'segment_length': segment_length,
                'total_length': audio_length,
                'num_segments': len(result.segments),
                'last_segment': last_segment,
            },
        )

        # If, for whatever reason, we don't have any segments, we should just
        # skip this segment and move on to the next one.
        if not result.segments:
            if start + segment_length >= audio_length:
                break

            start = end
            continue
        
        segments: list[Segment] = []

        # The segment timestamps are relative to the provided segment, so we add the start time
        # of the segment to each segment to make them absolute.
        for segment in result.segments:
            if not segment.start or not segment.end:
                continue
            
            nsegment = Segment(
                start=start + int(segment.start * 1000),  # Convert to milliseconds
                end=start + int(segment.end * 1000),  # Convert to milliseconds
                text=segment.text.strip(),
            )

            # If a speaker identifier is provided, identify the speaker for this segment
            if speaker_identifier:
                speaker_id = speaker_identifier.identify_or_register_speaker(
                    audio=naudio[nsegment.start:nsegment.end].to_numpy(),
                    sr=sr,
                )

                nsegment.speaker_id = speaker_id
            
            segments.append(nsegment)

        # If there's more segments to come, we should pop off the last segment
        # as it's likely to be a partial sentence
        if start + segment_length < audio_length and len(segments) > 1:
            if not last_segment:
                segments.pop(-1)

        # If an aligner is provided, refine word boundaries for each segment
        if aligner:
            aligned_segments: list[Segment] = []
            for seg in segments:
                word_segments = _align_segment(naudio, seg, aligner)
                aligned_segments.extend(word_segments)
            segments = aligned_segments if aligned_segments else segments

        complete_segments.extend(segments)

        # Move the start to the end of the last segment
        start = segments[-1].end

    return TranscriptionResult(
        transcript=" ".join([segment.text for segment in complete_segments]),
        segments=complete_segments,
    )
