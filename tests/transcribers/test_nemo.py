from pathlib import Path

from audio_segmentation.transcriber.nemo import NemoTranscriber, NemoModel
from audio_segmentation.types.audio import Audio
from audio_segmentation.utility import load_audio


class TestNemoTranscriber:
    def test_transcribe_ten_minute_segment_parakeet(self, ten_minute_segment_path: Path) -> None:
        audio, sr = load_audio(ten_minute_segment_path, sr=16000, mono=True)
        transcriber = NemoTranscriber(model_name=NemoModel.PARAKEET_TDT_V2, device_index=1)  # NOTE: Change this based on your GPU availability
    
        naudio = Audio(audio, sr=sr)
        naudio_segment = naudio[0:8 * 60 * 1000]

        result_tdt = transcriber.transcribe(
            naudio_segment.to_numpy(),
            sr=naudio_segment.sr,
        )

        del transcriber

        assert result_tdt.segments[0].text == 'This is Audible.'
