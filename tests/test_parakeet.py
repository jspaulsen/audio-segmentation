import pydub

from audio_segmentation.transcriber.parakeet import ParakeetTranscriber, ParakeetModel


class TestParakeetTranscription:
    def test_transcribe_ten_minute_segment(self, ten_minute_segment: pydub.AudioSegment):
        transcriber = ParakeetTranscriber(model_name=ParakeetModel.TDT_V2)
        result_tdt = transcriber.transcribe(ten_minute_segment, word_level_segmentation=True)
        
        del transcriber

        # transcriber = ParakeetTranscriber(model_name=ParakeetModel.RNNT)
        # result_rnnt = transcriber.transcribe(ten_minute_segment, word_level_segmentation=True)
        
        # del transcriber

        simple_transcript_tdt = ' '.join(segment.text for segment in result_tdt.segments[:3])
        # simple_transcript_rnnt = ' '.join(segment.text for segment in result_rnnt.segments[:3])

        assert simple_transcript_tdt == 'This is Audible.'
