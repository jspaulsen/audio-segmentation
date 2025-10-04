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
        audio = load_audio(ten_minute_segment_path)

        transcriber = NemoTranscriber(
            model_name=NemoModel.PARAKEET_TDT_V2,
            device_index=0, # NOTE: Change this based on your GPU availability
        )

        result = transcribe_audio(
            audio=audio,
            transcriber=transcriber,
            # use_sentence_segmentation=True,
            # segment_length=4 * 60 * 1000,  # 4 minutes in milliseconds
            raise_exception_on_mismatch=True
        )

        assert result.segments[0].text == "This is Audible."

    def test_transcribe_segmented_whisperx(self, ten_minute_segment_path: Path) -> None:
        audio = load_audio(ten_minute_segment_path)

        transcriber = WhisperxTranscriber(
            model_name=WhisperxModel.Tiny,
            device_index=0, # NOTE: Change this based on your GPU availability
        )

        result = transcribe_audio(
            audio,
            transcriber=transcriber,
            # use_sentence_segmentation=True,
            # segment_length=4 * 60 * 1000,  # 4 minutes in milliseconds
            raise_exception_on_mismatch=True
        )

        assert result.segments[0].text == "This is Audible."
