from pathlib import Path
import torch

from audio_segmentation.verifiers.speechbrain import SpeechBrainVerifier
from audio_segmentation.utility import load_audio


class TestSpeechBrainVerifier:
    def test_verify_same_speakers(
        self,
        glimpsed_14_path: Path,
        glimpsed_45_path: Path,
    ) -> None:
        """
        Test that the SpeechBrainVerifier correctly identifies that the two
        'glimpsed_' audio files contain the same speaker.
        """
        verifier = SpeechBrainVerifier(device_index=None)

        # Load the audio files
        audio_14, _ = load_audio(glimpsed_14_path, sr=verifier.required_sample_rate, mono=True)
        audio_45, _ = load_audio(glimpsed_45_path, sr=verifier.required_sample_rate, mono=True)

        # Create embeddings
        embedding_14 = verifier.create_embedding(audio_14)
        embedding_45 = verifier.create_embedding(audio_45)

        # Compute similarity
        similarity= torch.nn.functional.cosine_similarity(
            embedding_14,
            embedding_45,
        ).item()

        # Assert that the speakers are identified as the same
        assert similarity > 0.25, f"Expected similarity score > 0.25 but got {similarity}"

    def test_verify_different_speakers(
        self,
        glimpsed_14_path: Path,
        five_second_segment_path: Path,
    ) -> None:
        """
        Test that the SpeechBrainVerifier correctly identifies that the
        'glimpsed_' and '5s_segment.wav' audio files contain different speakers.
        """
        verifier = SpeechBrainVerifier(device_index=None)

        # Load the audio files
        audio_glimpsed, _ = load_audio(glimpsed_14_path, sr=verifier.required_sample_rate, mono=True)
        audio_5s, _ = load_audio(five_second_segment_path, sr=verifier.required_sample_rate, mono=True)

        # Create embeddings
        embedding_glimpsed = verifier.create_embedding(audio_glimpsed)
        embedding_5s = verifier.create_embedding(audio_5s)

        # Compute similarity
        similarity= torch.nn.functional.cosine_similarity(
            embedding_glimpsed,
            embedding_5s,
        ).item()

        # Assert that the speakers are identified as different
        assert similarity < 0.25, f"Expected similarity score < 0.25 but got {similarity}"
