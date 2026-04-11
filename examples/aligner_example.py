"""
Example demonstrating TorchAudioAligner for improved word boundary detection.

This shows how to use forced alignment to get more precise word boundaries
compared to standard ASR alignment, particularly for initial consonants
and final sounds that may otherwise be cut off.

Requires: nemo optional dependency (uv sync --extra nemo)
"""

from pathlib import Path

import librosa
import numpy as np

from audio_segmentation import (
    NemoTranscriber,
    NemoModel,
    TorchAudioAligner,
    load_audio,
    refine_segment_timestamps,
    Segment,
)


def main(device_index: int = 0):
    audio_fpath = Path("tests/fixtures/10m_segment.wav")

    # if the current cwd is examples, then we need to go up one level
    if Path.cwd().name == "examples":
        audio_fpath = Path("..") / audio_fpath

    # Load audio
    audio, sr = load_audio(audio_fpath)
    print(f"Loaded audio: {len(audio)} samples at {sr}Hz")

    # Transcribe with Parakeet
    print("\nTranscribing with Parakeet...")
    transcriber = NemoTranscriber(
        model_name=NemoModel.PARAKEET_TDT_V2,
        device_index=device_index,
    )

    # Resample for transcriber if needed
    transcribe_audio = audio
    transcribe_sr = sr
    if transcriber.required_sample_rate and sr != transcriber.required_sample_rate:
        transcribe_audio = librosa.resample(
            audio, orig_sr=sr, target_sr=transcriber.required_sample_rate
        )
        transcribe_sr = transcriber.required_sample_rate

    result = transcriber.transcribe(transcribe_audio, transcribe_sr)
    print(f"Transcript: {result.transcript[:200]}...")
    print(f"Transcriber segments: {len(result.segments)}")

    # Now align with TorchAudioAligner for better word boundaries
    print("\nAligning with TorchAudioAligner...")
    aligner = TorchAudioAligner(device_index=device_index)

    # Resample full audio for aligner if needed
    align_audio = audio
    align_sr = sr
    if aligner.required_sample_rate and sr != aligner.required_sample_rate:
        align_audio = librosa.resample(
            audio, orig_sr=sr, target_sr=aligner.required_sample_rate
        )
        align_sr = aligner.required_sample_rate

    # Pick a segment from the transcription to align
    raw_segment = result.segments[4]
    print(f"\nOriginal segment: \"{raw_segment.text}\"")
    print(f"  Transcriber timing: {raw_segment.start:.3f}s - {raw_segment.end:.3f}s")

    # Convert to Segment (milliseconds) for refinement
    segment = Segment(
        start=int(raw_segment.start * 1000),
        end=int(raw_segment.end * 1000),
        text=raw_segment.text,
    )

    # Use silence detection to find true speech boundaries
    # This expands the segment to capture any cut-off speech
    refined = refine_segment_timestamps(
        audio=align_audio,
        sr=align_sr,
        segment=segment,
        search_boundary=200,  # Look 200ms before/after
        pad=50,  # Add 50ms padding for safety
    )
    print(f"  Refined timing: {refined.start/1000:.3f}s - {refined.end/1000:.3f}s")

    # Extract the refined audio segment
    start_sample = int(refined.start / 1000 * align_sr)
    end_sample = int(refined.end / 1000 * align_sr)
    segment_audio = align_audio[start_sample:end_sample]

    # Align the segment
    segments = aligner.align(segment_audio, align_sr, refined.text)

    # Adjust timestamps to be relative to original audio
    segment_offset = refined.start / 1000  # Convert back to seconds

    print(f"\nAligned {len(segments)} words:\n")
    for seg in segments:
        abs_start = seg.start + segment_offset
        abs_end = seg.end + segment_offset
        duration_ms = (seg.end - seg.start) * 1000
        print(f'  "{seg.text}": {abs_start:.3f}s - {abs_end:.3f}s ({duration_ms:.0f}ms)')


if __name__ == "__main__":
    main()
