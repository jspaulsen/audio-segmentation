"""
Example demonstrating TorchAudioAligner for improved word boundary detection.

This shows how to use forced alignment to get more precise word boundaries
compared to standard ASR alignment, particularly for initial consonants
and final sounds that may otherwise be cut off.

Requires: nemo optional dependency (uv sync --extra nemo)
"""

from pathlib import Path

from audio_segmentation import (
    NemoTranscriber,
    NemoModel,
    TorchAudioAligner,
    load_audio,
    transcribe_audio,
)


def main(device_index: int = 0):
    audio_fpath = Path("tests/fixtures/10m_segment.wav")

    # if the current cwd is examples, then we need to go up one level
    if Path.cwd().name == "examples":
        audio_fpath = Path("..") / audio_fpath

    # Load audio
    audio, sr = load_audio(audio_fpath)
    print(f"Loaded audio: {len(audio)} samples at {sr}Hz")

    # Setup transcriber and aligner
    transcriber = NemoTranscriber(
        model_name=NemoModel.PARAKEET_TDT_V2,
        device_index=device_index,
    )
    aligner = TorchAudioAligner(device_index=device_index)

    # Transcribe with alignment - produces word-level segments with refined boundaries
    print("\nTranscribing with Parakeet + TorchAudio alignment...")
    result = transcribe_audio(
        audio=audio,
        sr=sr,
        transcriber=transcriber,
        aligner=aligner,
    )

    print(f"Transcript: {result.transcript[:200]}...")
    print(f"Word segments: {len(result.segments)}")

    # Show first 20 word segments
    print("\nFirst 20 aligned words:\n")
    for seg in result.segments[:20]:
        duration_ms = seg.end - seg.start
        print(f'  "{seg.text}": {seg.start}ms - {seg.end}ms ({duration_ms}ms)')


if __name__ == "__main__":
    main()
