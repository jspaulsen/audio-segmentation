import json
import pytest

from audio_segmentation.segmenter import sentence_segmenter, RawTranscriptionResult
from audio_segmentation.types.segment import RawSegment


@pytest.fixture
def transcription_result():
    with open('tests/fixtures/transcription_result.json', 'r') as f:
        data = json.load(f)

    return RawTranscriptionResult(
        segments=[
            RawSegment(start=segment['start'], end=segment['end'], text=segment['text'])
            for segment in data['segments']
        ],
        transcript=data['transcript']
    )


class TestSegmentation:
    def test_sentence_segmenter(self, transcription_result):
        segmented_result = sentence_segmenter(
            transcription_result,
            raise_exception_on_mismatch=True,
        )

        assert segmented_result is not None
        assert segmented_result[0].text == 'I wish I could, hun.'
