from pathlib import Path

import librosa
import pytest

from audio_segmentation import Aligner, TorchAudioAligner, load_audio


class TestTorchAudioAligner:
    @pytest.fixture
    def aligner(self) -> TorchAudioAligner:
        return TorchAudioAligner(device="cpu")

    def test_implements_protocol(self, aligner: TorchAudioAligner) -> None:
        """Verify TorchAudioAligner implements the Aligner protocol."""
        assert isinstance(aligner, Aligner)

    def test_properties(self, aligner: TorchAudioAligner) -> None:
        """Verify aligner properties are set correctly."""
        assert aligner.required_sample_rate == 16000

    def test_align_returns_segments(
        self, aligner: TorchAudioAligner, five_second_segment_path: Path
    ) -> None:
        """Verify alignment produces word-level segments."""
        audio, sr = load_audio(five_second_segment_path)

        # Resample to required sample rate
        if sr != aligner.required_sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=aligner.required_sample_rate
            )
            sr = aligner.required_sample_rate

        transcript = "I wish I could"
        segments = aligner.align(audio, sr, transcript)

        # Should return one segment per word
        assert len(segments) == 4

        # Verify segment structure
        for seg in segments:
            assert seg.start is not None
            assert seg.end is not None
            assert seg.start < seg.end
            assert seg.text in transcript

    def test_align_preserves_word_order(
        self, aligner: TorchAudioAligner, five_second_segment_path: Path
    ) -> None:
        """Verify segments are returned in chronological order."""
        audio, sr = load_audio(five_second_segment_path)

        if sr != aligner.required_sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=aligner.required_sample_rate
            )
            sr = aligner.required_sample_rate

        transcript = "I wish I could"
        segments = aligner.align(audio, sr, transcript)

        # Segments should be in order
        for i in range(len(segments) - 1):
            assert segments[i].start <= segments[i + 1].start

    def test_align_sample_rate_mismatch_raises(
        self, aligner: TorchAudioAligner
    ) -> None:
        """Verify error is raised for incorrect sample rate."""
        import numpy as np

        audio = np.zeros(16000, dtype=np.float32)
        wrong_sr = 44100

        with pytest.raises(ValueError, match="Sample rate mismatch"):
            aligner.align(audio, wrong_sr, "test")

    def test_align_empty_transcript(
        self, aligner: TorchAudioAligner, five_second_segment_path: Path
    ) -> None:
        """Verify empty transcript returns empty segments."""
        audio, sr = load_audio(five_second_segment_path)

        if sr != aligner.required_sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=aligner.required_sample_rate
            )
            sr = aligner.required_sample_rate

        segments = aligner.align(audio, sr, "")
        assert segments == []
