import numpy as np
import torch

from audio_segmentation.verifiers.profile import SpeakerProfile
from audio_segmentation.verifiers.verifier import SpeakerVerifier
from audio_segmentation.utility import convert_to_mono, resample


class SpeakerIdentifier:
    """
    Stateful speaker identification system that tracks and identifies speakers
    across audio chunks using embeddings from a SpeakerVerifier.
    """

    def __init__(
        self,
        verifier: SpeakerVerifier,
        similarity_threshold: float,
    ) -> None:
        """
        Initialize the SpeakerIdentifier.

        Args:
            verifier: SpeakerVerifier instance for creating embeddings and computing similarity
            similarity_threshold: Threshold above which two embeddings are considered the same speaker
            max_embeddings_per_speaker: Maximum number of embeddings to store per speaker for robustness
        """
        self.verifier = verifier
        self.similarity_threshold = similarity_threshold
        self.profiles: list[SpeakerProfile] = []
        self.next_speaker_id: int = 0

    def _preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess audio to match verifier requirements.

        Args:
            audio: Audio data as numpy array
            sr: Sample rate of the audio

        Returns:
            Preprocessed audio array
        """
        if self.verifier.requires_mono_audio:  # Convert to mono if required
            if audio.ndim == 2 and audio.shape[0] == 2:
                audio = convert_to_mono(audio)  # Check if audio is stereo (2D with shape [2, samples])
            elif audio.ndim > 2:  # If audio is already 1D, it's already mono
                audio = convert_to_mono(audio) # For other multi-channel formats, convert to mono

        # Resample if needed
        return resample(audio, sr, self.verifier.required_sample_rate)

    def identify_or_register_speaker(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> int | None:
        """
        Identify an existing speaker or register a new one based on audio.

        Args:
            audio: Audio data as numpy array
            sr: Sample rate of the audio

        Returns:
            Speaker ID (int) if audio is long enough, None if audio is too short
        """
        min_audio_length = 0.5
        audio_duration = len(audio) / sr

        # Return None for segments that are too short for reliable speaker verification
        if audio_duration < min_audio_length:
            return None

        # Preprocess audio
        processed_audio = self._preprocess_audio(audio, sr)

        # Create embedding for the input audio
        embedding: torch.Tensor = self.verifier.create_embedding(processed_audio) # [B, D]
        embedding = embedding.squeeze(0)

        # Try to match against known speakers
        best_match_id: int | None = None
        best_similarity: float = -1.0

        for profile in self.profiles:
            similarity, is_match = profile.compare(embedding, self.similarity_threshold)

            if is_match and similarity > best_similarity:
                best_similarity = similarity
                best_match_id = profile.speaker_id

        # If we found a match, update that profile's centroid with the new embedding
        if best_match_id is not None:
            for profile in self.profiles:
                if profile.speaker_id == best_match_id:
                    profile.add_embedding_ema(embedding)
                    break

            return best_match_id

        # No match found - register new speaker
        new_profile = SpeakerProfile(self.next_speaker_id, embedding)
        self.profiles.append(new_profile)
        self.next_speaker_id = self.next_speaker_id + 1

        return new_profile.speaker_id
