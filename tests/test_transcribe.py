from pathlib import Path

from audio_segmentation import (
    NemoTranscriber,
    NemoModel,
    WhisperxTranscriber,
    WhisperxModel,
    transcribe_audio,
    load_audio,
)


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
