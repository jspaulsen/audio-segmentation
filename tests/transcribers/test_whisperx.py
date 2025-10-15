import numpy as np

from audio_segmentation.transcriber.whisperx import WhisperxModel, WhisperxTranscriber


class TestWhisperxTranscriber:
    def test_transcribe_ten_minute_segment(self, ten_minute_segment: tuple[np.ndarray, int]) -> None:
        audio, sr = ten_minute_segment
        transcriber = WhisperxTranscriber(model_name=WhisperxModel.Tiny, device_index=1)  # NOTE: Change this based on your GPU availability
        results = transcriber.transcribe(
            audio,
            sr,
        )

        del transcriber

        simple_transcript = ' '.join(segment.text for segment in results.segments[:3])

        assert simple_transcript == 'This is audible.'
