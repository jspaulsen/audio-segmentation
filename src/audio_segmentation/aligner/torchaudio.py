import string
from enum import Enum

import numpy as np
import torch
import torchaudio
from torchaudio.functional import forced_align

from audio_segmentation.types.segment import RawSegment


class AlignerModel(Enum):
    """Available models for TorchAudioAligner."""

    MMS_FA = "MMS_FA"
    """Multilingual model specifically designed for forced alignment. Best accuracy."""

    WAV2VEC2_ASR_BASE_960H = "WAV2VEC2_ASR_BASE_960H"
    """Wav2Vec2 base model trained on 960h LibriSpeech. Fast, good accuracy."""

    WAV2VEC2_ASR_LARGE_960H = "WAV2VEC2_ASR_LARGE_960H"
    """Wav2Vec2 large model trained on 960h LibriSpeech. Better accuracy, slower."""


class TorchAudioAligner:
    """
    Forced alignment using torchaudio's forced_align with Wav2Vec2.

    Uses CTC-based forced alignment to produce accurate word boundaries,
    particularly improving alignment of initial consonants and final sounds
    that may be cut off with standard ASR-based alignment.
    """

    def __init__(
        self,
        model: AlignerModel = AlignerModel.MMS_FA,
        device: str = "cuda",
        device_index: int = 0,
    ):
        """
        Initialize the TorchAudio aligner.

        Args:
            model: Model to use for alignment. Defaults to MMS_FA for best accuracy.
            device: Device to run the model on ('cuda' or 'cpu').
            device_index: GPU index if using CUDA.
        """
        if device == "cuda" and torch.cuda.is_available():
            self._device = torch.device(f"cuda:{device_index}")
        else:
            self._device = torch.device("cpu")

        # Load model bundle
        bundle = getattr(torchaudio.pipelines, model.value)
        self._model = bundle.get_model().to(self._device)
        self._model.eval()
        self._labels = bundle.get_labels()
        self._sample_rate = bundle.sample_rate

        # Build label to index mapping
        self._label_to_idx = {label: idx for idx, label in enumerate(self._labels)}
        self._blank_idx = self._label_to_idx.get("-", 0)

    @property
    def required_sample_rate(self) -> int:
        """Required sample rate for the aligner (16kHz for Wav2Vec2)."""
        return self._sample_rate

    def align(
        self,
        audio: np.ndarray,
        sr: int,
        transcript: str,
    ) -> list[RawSegment]:
        """
        Align transcript to audio, returning word-level segments.

        Args:
            audio: Audio waveform as numpy array (mono, float32).
            sr: Sample rate of the audio.
            transcript: Text transcript to align to the audio.

        Returns:
            List of RawSegment with word-level timing boundaries.
        """
        if sr != self._sample_rate:
            raise ValueError(
                f"Sample rate mismatch: got {sr}, expected {self._sample_rate}"
            )

        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]

        # Convert to tensor and add batch dimension
        waveform = torch.from_numpy(audio).float().unsqueeze(0).to(self._device)

        # Get emission probabilities from Wav2Vec2
        with torch.no_grad():
            emissions, _ = self._model(waveform)
            # Log softmax for forced_align
            emissions = torch.log_softmax(emissions, dim=-1)

        # Tokenize transcript to target indices
        words, targets = self._tokenize(transcript)

        if not targets:
            return []

        # Convert targets to tensor (batch dimension required)
        targets_tensor = (
            torch.tensor(targets, dtype=torch.int32).unsqueeze(0).to(self._device)
        )

        # Run forced alignment
        aligned_tokens, scores = forced_align(
            emissions, targets_tensor, blank=self._blank_idx
        )

        # Convert frame indices to word-level RawSegments
        return self._frames_to_segments(
            aligned_tokens[0],  # Remove batch dimension
            words,
            transcript,
            waveform.size(-1),
            emissions.size(1),
        )

    def _tokenize(self, transcript: str) -> tuple[list[str], list[int]]:
        """
        Tokenize transcript to model label indices.

        Returns:
            Tuple of (words list, token indices list).
        """
        # Determine case based on model labels (MMS_FA uses lowercase, others uppercase)
        has_lowercase = "a" in self._label_to_idx

        # Normalize case and remove punctuation
        clean_transcript = transcript.lower() if has_lowercase else transcript.upper()
        clean_transcript = clean_transcript.translate(
            str.maketrans("", "", string.punctuation)
        )

        words = clean_transcript.split()
        tokens = []

        # Determine word separator token
        word_separator_idx = None
        for sep in ["|", "*", " "]:
            if sep in self._label_to_idx:
                word_separator_idx = self._label_to_idx[sep]
                break

        for word in words:
            for char in word:
                if char in self._label_to_idx:
                    tokens.append(self._label_to_idx[char])
            # Add word boundary
            if word_separator_idx is not None:
                tokens.append(word_separator_idx)

        # Remove trailing word boundary
        if tokens and word_separator_idx is not None and tokens[-1] == word_separator_idx:
            tokens.pop()

        return words, tokens

    def _frames_to_segments(
        self,
        aligned_tokens: torch.Tensor,
        words: list[str],
        original_transcript: str,
        num_samples: int,
        num_frames: int,
    ) -> list[RawSegment]:
        """
        Convert aligned frame indices to word-level RawSegments.

        Args:
            aligned_tokens: Tensor of token indices per frame.
            words: List of words from tokenization.
            original_transcript: Original transcript for getting original word forms.
            num_samples: Number of audio samples.
            num_frames: Number of emission frames.

        Returns:
            List of RawSegment with word-level timing.
        """
        # Calculate frame duration in seconds
        frame_duration = num_samples / num_frames / self._sample_rate

        # Get original words (preserving case and punctuation context)
        original_words = original_transcript.split()

        segments = []
        aligned_tokens_list = aligned_tokens.tolist()

        # Find word boundaries by tracking when we see word separator tokens
        word_separator = None
        for sep in ["|", "*", " "]:
            if sep in self._label_to_idx:
                word_separator = self._label_to_idx[sep]
                break

        current_word_idx = 0
        word_start_frame = None
        word_end_frame = None

        for frame_idx, token_idx in enumerate(aligned_tokens_list):
            # Skip blank tokens for boundary detection
            if token_idx == self._blank_idx:
                continue

            # Check if this is a word separator
            if token_idx == word_separator:
                # End of current word
                if word_start_frame is not None and current_word_idx < len(words):
                    word_end_frame = frame_idx
                    start_time = word_start_frame * frame_duration
                    end_time = word_end_frame * frame_duration

                    # Use original word if available, otherwise use cleaned word
                    word_text = (
                        original_words[current_word_idx]
                        if current_word_idx < len(original_words)
                        else words[current_word_idx]
                    )

                    segments.append(
                        RawSegment(
                            start=start_time,
                            end=end_time,
                            text=word_text,
                        )
                    )

                    current_word_idx += 1
                    word_start_frame = None
                    word_end_frame = None
            else:
                # Character token - track word boundaries
                if word_start_frame is None:
                    word_start_frame = frame_idx
                word_end_frame = frame_idx

        # Handle last word (no trailing separator)
        if word_start_frame is not None and current_word_idx < len(words):
            word_end_frame = len(aligned_tokens_list) - 1
            start_time = word_start_frame * frame_duration
            end_time = (word_end_frame + 1) * frame_duration

            word_text = (
                original_words[current_word_idx]
                if current_word_idx < len(original_words)
                else words[current_word_idx]
            )

            segments.append(
                RawSegment(
                    start=start_time,
                    end=end_time,
                    text=word_text,
                )
            )

        return segments
