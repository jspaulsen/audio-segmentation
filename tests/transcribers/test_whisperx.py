import pydub

from audio_segmentation.transcriber.whisperx import WhisperxModel, WhisperxTranscriber


class TestWhisperxTranscriber:
    def test_transcribe_ten_minute_segment(self, ten_minute_segment: pydub.AudioSegment):
        transcriber = WhisperxTranscriber(model_name=WhisperxModel.Tiny)
        results = transcriber.transcribe(ten_minute_segment)

        del transcriber

        simple_transcript = ' '.join(segment.text for segment in results.segments[:3])

        assert simple_transcript == 'This is audible.'
