from pathlib import Path

from audio_segmentation.transcriber.whisperx import WhisperxModel, WhisperxTranscriber
from audio_segmentation.utility import load_audio


class TestWhisperxTranscriber:
    def test_transcribe_ten_minute_segment(self, ten_minute_segment_path: Path) -> None:
        audio, sr = load_audio(ten_minute_segment_path, sr=16000, mono=True)
        transcriber = WhisperxTranscriber(model_name=WhisperxModel.Tiny, device_index=1)  # NOTE: Change this based on your GPU availability
        results = transcriber.transcribe(
            audio,
            sr,
        )

        del transcriber
        assert results.segments[0].text.strip() == 'This is audible.'
