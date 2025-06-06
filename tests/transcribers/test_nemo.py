import pydub

from audio_segmentation.transcriber.nemo import NemoTranscriber, NemoModel


class TestNemoTranscriber:
    def test_transcribe_ten_minute_segment_parakeet(self, ten_minute_segment: pydub.AudioSegment):
        transcriber = NemoTranscriber(model_name=NemoModel.PARAKEET_TDT_V2)
        result_tdt = transcriber.transcribe(ten_minute_segment, word_level_segmentation=True)
        
        del transcriber

        simple_transcript_tdt = ' '.join(segment.text for segment in result_tdt.segments[:3])

        assert simple_transcript_tdt == 'This is Audible.'

    # TODO: Enable this test when the canary model is stable for longer segments
    # def test_transcribe_ten_minute_segment_canary(self, ten_minute_segment: pydub.AudioSegment):
    #     transcriber = NemoTranscriber(model_name=NemoModel.CANARY_1B_FLASH)
    #     result_canary = transcriber.transcribe(ten_minute_segment, word_level_segmentation=True)
        
    #     simple_transcript_canary = ' '.join(segment.text for segment in result_canary.segments[:3])

    #     assert simple_transcript_canary == 'This is Audible.'
