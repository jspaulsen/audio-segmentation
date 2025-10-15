import numpy as np

from audio_segmentation.transcriber.nemo import NemoTranscriber, NemoModel


class TestNemoTranscriber:
    def test_transcribe_ten_minute_segment_parakeet(self, ten_minute_segment: tuple[np.ndarray, int]) -> None:
        audio, sr = ten_minute_segment
        transcriber = NemoTranscriber(model_name=NemoModel.PARAKEET_TDT_V2, device_index=1)  # NOTE: Change this based on your GPU availability

        result_tdt = transcriber.transcribe(
            audio,
            sr,
        )

        del transcriber

        simple_transcript_tdt = ' '.join(segment.text for segment in result_tdt.segments[:3])

        assert simple_transcript_tdt == 'This is Audible.'
