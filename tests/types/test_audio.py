import numpy as np
from audio_segmentation.types import Audio


class TestAudio:
    def test_audio(self) -> None:
        sample_rate = 16000
        duration = 5.25  # seconds

        audio = Audio(
            data=np.random.randn(int(sample_rate * duration)),
            sr=sample_rate,
        )

        assert len(audio) == 5250  # length in milliseconds

        # get a slice of the audio
        segment = audio[1000:2000]
        assert len(segment) == 16000  # 1 second of audio at 16kHz
