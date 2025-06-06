from pathlib import Path

from audio_segmentation import NemoTranscriber, NemoModel, WhisperxTranscriber, WhisperxModel, transcribe_audio


class TestTranscribeAudio:
    def test_transcribe_segmented_nemo(self, ten_minute_segment_path: Path) -> None:
        transcriber = NemoTranscriber(model_name=NemoModel.PARAKEET_TDT_V2)

        result = transcribe_audio(
            ten_minute_segment_path,
            transcriber=transcriber,
            use_sentence_segmentation=True,
            segment_length=4 * 60 * 1000,  # 4 minutes in milliseconds
            raise_exception=True
        )
        
        assert result
        assert result[0].text == "This is Audible."

    def test_transcribe_segmented_whisperx(self, ten_minute_segment_path: Path) -> None:
        transcriber = WhisperxTranscriber(model_name=WhisperxModel.Tiny)

        result = transcribe_audio(
            ten_minute_segment_path,
            transcriber=transcriber,
            use_sentence_segmentation=True,
            segment_length=4 * 60 * 1000,  # 4 minutes in milliseconds
            raise_exception=True
        )
        
        assert result
        assert result[0].text == "This is audible."
