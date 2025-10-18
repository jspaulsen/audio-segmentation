from pathlib import Path
import numpy as np

from audio_segmentation import (
    NemoTranscriber,
    NemoModel,
    WhisperxTranscriber,
    WhisperxModel,
    transcribe_audio,
    load_audio,
)
from audio_segmentation.verifiers.speechbrain import SpeechBrainVerifier


class TestTranscribeAudio:
    def test_transcribe_segmented_nemo(self, ten_minute_segment_path: Path) -> None:
        audio, sr = load_audio(
            ten_minute_segment_path,
            sr=16000,
            mono=True,
        )

        transcriber = NemoTranscriber(
            model_name=NemoModel.PARAKEET_TDT_V2,
            device_index=1, # NOTE: Change this based on your GPU availability
        )

        result = transcribe_audio(
            audio=audio,
            sr=sr,
            transcriber=transcriber,
        )

        assert result.segments[0].text == "This is Audible."

    def test_transcribe_segmented_whisperx(self, ten_minute_segment_path: Path) -> None:
        audio, sr = load_audio(ten_minute_segment_path)

        transcriber = WhisperxTranscriber(
            model_name=WhisperxModel.Tiny,
            device_index=1, # NOTE: Change this based on your GPU availability
        )

        result = transcribe_audio(
            audio,
            sr=sr,
            transcriber=transcriber,
        )

        assert result.segments[0].text.strip() == "This is audible."

    def test_transcribe_audio_with_nemo_and_speechbrain(
        self,
        ten_minute_segment: tuple[np.ndarray, int]
    ) -> None:
        """Test transcribe_audio with Nemo transcriber and SpeechBrain verifier."""
        audio, sr = ten_minute_segment

        # Initialize transcriber and verifier
        transcriber = NemoTranscriber(
            model_name=NemoModel.PARAKEET_TDT_V2,
            device_index=None,
        )
        verifier = SpeechBrainVerifier(device_index=None)

        # Transcribe the audio with speaker identification
        # Higher threshold = more strict (speakers must be more similar to match)
        # Lower threshold = more permissive (speakers can be less similar to match)
        result = transcribe_audio(
            audio=audio,
            sr=sr,
            transcriber=transcriber,
            speaker_verifier=verifier,
            similarity_threshold=0.50,
        )

        # Check that we got a transcription
        assert result.transcript
        assert len(result.segments) > 0

        # Count unique speakers
        unique_speakers = set()
        for segment in result.segments:
            if segment.speaker_id is not None:
                unique_speakers.add(segment.speaker_id)

        # Should have 2 speakers (or at most 3)
        num_speakers = len(unique_speakers)
        print(f"Found {num_speakers} unique speakers: {unique_speakers}")
        assert num_speakers >= 1, "Should have at least 1 speaker"
        assert num_speakers <= 3, f"Should have at most 3 speakers, but found {num_speakers}"

        # Ideally, we should have exactly 2 speakers
        assert num_speakers == 2, f"Expected 2 speakers, but found {num_speakers}. Try adjusting the threshold."
