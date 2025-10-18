from pathlib import Path
import numpy as np
import torch
import pytest

from audio_segmentation.verifiers.profile import SpeakerProfile
from audio_segmentation.verifiers.speaker import SpeakerIdentifier
from audio_segmentation.verifiers.speechbrain import SpeechBrainVerifier
from audio_segmentation.utility import load_audio


class TestSpeakerProfile:
    def test_initialization(self) -> None:
        """Test that SpeakerProfile initializes correctly."""
        speaker_id = 42
        initial_embedding = torch.randn(192)

        profile = SpeakerProfile(speaker_id, initial_embedding)

        assert profile.speaker_id == speaker_id
        assert torch.equal(profile.centroid, initial_embedding)
        assert profile.count == 0

    def test_add_embedding_standard_average(self) -> None:
        """Test that add_embedding correctly updates the centroid using running average."""
        initial_embedding = torch.ones(192)
        profile = SpeakerProfile(0, initial_embedding)

        # Add a second embedding (count=0 -> count=1)
        second_embedding = torch.zeros(192)
        profile.add_embedding(second_embedding)

        # With count=1, the formula centroid + (embedding - centroid) / 1 replaces the centroid
        assert torch.allclose(profile.centroid, second_embedding)
        assert profile.count == 1

        # Add a third embedding (count=1 -> count=2)
        third_embedding = torch.ones(192) * 2
        profile.add_embedding(third_embedding)

        # With count=2, centroid = zeros + (2*ones - zeros) / 2 = zeros + ones = ones
        expected_centroid = torch.ones(192)
        assert torch.allclose(profile.centroid, expected_centroid)
        assert profile.count == 2

    def test_add_embedding_ema(self) -> None:
        """Test that add_embedding_ema correctly updates the centroid using EMA."""
        initial_embedding = torch.ones(192)
        profile = SpeakerProfile(0, initial_embedding)

        # Add a new embedding with alpha=0.1
        new_embedding = torch.zeros(192)
        alpha = 0.1
        profile.add_embedding_ema(new_embedding, alpha)

        # Expected centroid: (1-alpha) * old + alpha * new
        expected_centroid = (1 - alpha) * initial_embedding + alpha * new_embedding
        assert torch.allclose(profile.centroid, expected_centroid)

    def test_add_embedding_raises_on_wrong_dimension(self) -> None:
        """Test that add_embedding raises ValueError for non-1D embeddings."""
        profile = SpeakerProfile(0, torch.randn(192))

        # Test with 2D embedding
        with pytest.raises(ValueError, match="must be a 1D tensor"):
            profile.add_embedding(torch.randn(1, 192))

    def test_add_embedding_ema_raises_on_wrong_dimension(self) -> None:
        """Test that add_embedding_ema raises ValueError for non-1D embeddings."""
        profile = SpeakerProfile(0, torch.randn(192))

        # Test with 2D embedding
        with pytest.raises(ValueError, match="must be a 1D tensor"):
            profile.add_embedding_ema(torch.randn(1, 192))

    def test_compare_same_embedding(self) -> None:
        """Test that compare returns high similarity for identical embeddings."""
        embedding = torch.randn(192)
        profile = SpeakerProfile(0, embedding)

        similarity, is_match = profile.compare(embedding, threshold=0.75)

        # Comparing an embedding to itself should give similarity of 1.0
        assert similarity > 0.99
        assert is_match is True

    def test_compare_similar_embeddings(self) -> None:
        """Test that compare returns high similarity for similar embeddings."""
        embedding1 = torch.randn(192)
        embedding2 = embedding1 + torch.randn(192) * 0.01  # Very similar

        profile = SpeakerProfile(0, embedding1)
        similarity, is_match = profile.compare(embedding2, threshold=0.75)

        # Should be very similar
        assert similarity > 0.9
        assert is_match is True

    def test_compare_different_embeddings(self) -> None:
        """Test that compare returns low similarity for different embeddings."""
        embedding1 = torch.randn(192)
        embedding2 = torch.randn(192)

        profile = SpeakerProfile(0, embedding1)
        similarity, is_match = profile.compare(embedding2, threshold=0.75)

        # Random embeddings should have low similarity
        assert similarity is not None
        assert is_match is False

    def test_compare_raises_on_wrong_dimension(self) -> None:
        """Test that compare raises ValueError for non-1D embeddings."""
        profile = SpeakerProfile(0, torch.randn(192))

        # Test with 2D embedding
        with pytest.raises(ValueError, match="must be a 1D tensor"):
            profile.compare(torch.randn(1, 192))


class TestSpeakerIdentifier:
    def test_initialization(self) -> None:
        """Test that SpeakerIdentifier initializes correctly."""
        verifier = SpeechBrainVerifier(device_index=None)
        similarity_threshold = 0.65

        identifier = SpeakerIdentifier(verifier, similarity_threshold)

        assert identifier.verifier == verifier
        assert identifier.similarity_threshold == similarity_threshold
        assert identifier.profiles == []
        assert identifier.next_speaker_id == 0

    def test_register_new_speaker(
        self,
        glimpsed_14_path: Path,
    ) -> None:
        """Test that identify_or_register_speaker registers a new speaker."""
        verifier = SpeechBrainVerifier(device_index=None)
        identifier = SpeakerIdentifier(verifier, similarity_threshold=0.60)

        # Load audio
        audio, sr = load_audio(glimpsed_14_path, sr=16000, mono=True)

        # First call should register a new speaker with ID 0
        speaker_id = identifier.identify_or_register_speaker(audio, sr)

        assert speaker_id == 0
        assert len(identifier.profiles) == 1
        assert identifier.next_speaker_id == 1
        assert identifier.profiles[0].speaker_id == 0

    def test_identify_same_speaker(
        self,
        glimpsed_14_path: Path,
        glimpsed_45_path: Path,
    ) -> None:
        """Test that the same speaker is identified across different audio clips."""
        verifier = SpeechBrainVerifier(device_index=None)
        identifier = SpeakerIdentifier(verifier, similarity_threshold=0.20)

        # Load audio files (both contain the same speaker)
        audio_14, sr_14 = load_audio(glimpsed_14_path, sr=16000, mono=True)
        audio_45, sr_45 = load_audio(glimpsed_45_path, sr=16000, mono=True)

        # Register first speaker
        speaker_id_1 = identifier.identify_or_register_speaker(audio_14, sr_14)

        # Second audio should identify as the same speaker
        speaker_id_2 = identifier.identify_or_register_speaker(audio_45, sr_45)

        assert speaker_id_1 == speaker_id_2
        assert len(identifier.profiles) == 1

    def test_identify_different_speakers(
        self,
        glimpsed_14_path: Path,
        five_second_segment_path: Path,
    ) -> None:
        """Test that different speakers are registered separately."""
        verifier = SpeechBrainVerifier(device_index=None)
        identifier = SpeakerIdentifier(verifier, similarity_threshold=0.60)

        # Load audio files (different speakers)
        audio_glimpsed, sr_glimpsed = load_audio(glimpsed_14_path, sr=16000, mono=True)
        audio_5s, sr_5s = load_audio(five_second_segment_path, sr=16000, mono=True)

        # Register first speaker
        speaker_id_1 = identifier.identify_or_register_speaker(audio_glimpsed, sr_glimpsed)

        # Second audio should be registered as a different speaker
        speaker_id_2 = identifier.identify_or_register_speaker(audio_5s, sr_5s)

        assert speaker_id_1 != speaker_id_2
        assert len(identifier.profiles) == 2
        assert speaker_id_1 == 0
        assert speaker_id_2 == 1

    def test_short_audio_returns_none(self) -> None:
        """Test that very short audio segments return None."""
        verifier = SpeechBrainVerifier(device_index=None)
        identifier = SpeakerIdentifier(verifier, similarity_threshold=0.60)

        # Create very short audio (0.3 seconds at 16kHz)
        short_audio = np.random.randn(int(16000 * 0.3))
        sr = 16000

        # Should return None for audio shorter than 0.5 seconds
        speaker_id = identifier.identify_or_register_speaker(short_audio, sr)

        assert speaker_id is None
        assert len(identifier.profiles) == 0

    def test_profile_updates_on_match(
        self,
        glimpsed_14_path: Path,
        glimpsed_45_path: Path,
    ) -> None:
        """Test that profiles are updated when matching speakers are found."""
        verifier = SpeechBrainVerifier(device_index=None)
        identifier = SpeakerIdentifier(verifier, similarity_threshold=0.20)

        # Load audio files (same speaker)
        audio_14, sr_14 = load_audio(glimpsed_14_path, sr=16000, mono=True)
        audio_45, sr_45 = load_audio(glimpsed_45_path, sr=16000, mono=True)

        # Register first speaker
        identifier.identify_or_register_speaker(audio_14, sr_14)

        # Store the original centroid
        original_centroid = identifier.profiles[0].centroid.clone()

        # Identify same speaker again
        identifier.identify_or_register_speaker(audio_45, sr_45)

        # Centroid should have been updated
        updated_centroid = identifier.profiles[0].centroid
        assert not torch.equal(original_centroid, updated_centroid)

    def test_multiple_speakers_registration(
        self,
        glimpsed_14_path: Path,
        glimpsed_45_path: Path,
        five_second_segment_path: Path,
    ) -> None:
        """Test registering multiple speakers and re-identifying them."""
        verifier = SpeechBrainVerifier(device_index=None)
        identifier = SpeakerIdentifier(verifier, similarity_threshold=0.60)

        # Load audio files
        audio_14, sr_14 = load_audio(glimpsed_14_path, sr=16000, mono=True)
        audio_45, sr_45 = load_audio(glimpsed_45_path, sr=16000, mono=True)
        audio_5s, sr_5s = load_audio(five_second_segment_path, sr=16000, mono=True)

        # Register speaker 1 (glimpsed_14)
        speaker_1 = identifier.identify_or_register_speaker(audio_14, sr_14)

        # Register speaker 2 (5s_segment - different speaker)
        speaker_2 = identifier.identify_or_register_speaker(audio_5s, sr_5s)

        # Identify speaker 1 again (glimpsed_45 - same as glimpsed_14)
        speaker_1_again = identifier.identify_or_register_speaker(audio_45, sr_45)

        assert speaker_1 == 0
        assert speaker_2 == 1
        assert speaker_1_again == 0
        assert len(identifier.profiles) == 2
