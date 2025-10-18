import numpy as np
from torch import Tensor
from speechbrain.inference.speaker import SpeakerRecognition

from audio_segmentation.verifiers.verifier import SpeakerVerifier


class SpeechBrainVerifier(SpeakerVerifier):
    def __init__(
        self,
        device_index: int | None = None,
    ) -> None:
        self.sample_rate = 16000  # SpeechBrain model requires 16kHz audio

        model: SpeakerRecognition | None = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",

            # TODO: This is not respecting the device_index properly
            run_opts={"device_index": f"cuda:{device_index}"} if device_index is not None else None,
        )

        if not model:
            raise ValueError("Failed to load SpeechBrain model")

        self.model = model

    @property
    def requires_mono_audio(self) -> bool:
        return True

    @property
    def required_sample_rate(self) -> int:
        return 16000

    def create_embedding(
        self,
        audio: np.ndarray | Tensor,
    ) -> Tensor: # [B, D]
        if isinstance(audio, np.ndarray):
            audio = Tensor(audio)

        # If audio has > 1 dimension, assume first dimension is batch
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        return (
            self
                .model
                .encode_batch(audio)  # encode_batch expects [B, T]
                .squeeze(0)
        )
