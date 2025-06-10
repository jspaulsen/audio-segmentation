import pydub

from audio_segmentation.transcriber.whisper_timestamped import WhisperModel, WhisperTimestampedTranscriber


class TestWhisperTimestampedTranscriber:
    def test_transcribe_ten_minute_segment(self, ten_minute_segment: pydub.AudioSegment):
        transcriber = WhisperTimestampedTranscriber(model_name=WhisperModel.Tiny)
        results = transcriber.transcribe(ten_minute_segment, word_level_segmentation=True)

        del transcriber

        simple_transcript = ' '.join(segment.text for segment in results.segments[:3])
        assert simple_transcript == 'This is audible.'
