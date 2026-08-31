import numpy as np

from audio_segmentation.transcribe import transcribe_audio
from audio_segmentation.transcriber.transcriber import RawTranscriptionResult
from audio_segmentation.types.segment import RawSegment


class FakeTranscriber:
    """
    Minimal Transcriber protocol implementation which replays canned results,
    one per call to transcribe().
    """

    def __init__(
        self,
        results: list[RawTranscriptionResult],
        ideal_segment_length: int | None = None,
    ) -> None:
        self.results = results
        self.calls = 0
        self._ideal_segment_length = ideal_segment_length

    def transcribe(self, audio: np.ndarray, sr: int, **kwargs) -> RawTranscriptionResult:
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1

        return result

    @property
    def ideal_segment_length(self) -> int | None:
        return self._ideal_segment_length

    @property
    def required_sample_rate(self) -> int | None:
        return None

    @property
    def requires_mono_audio(self) -> bool:
        return False

    @property
    def supports_word_level_segmentation(self) -> bool:
        return False


def silence(seconds: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


class TestTranscribeLoop:
    def test_keeps_segment_starting_at_zero(self) -> None:
        transcriber = FakeTranscriber([
            RawTranscriptionResult(
                transcript="first second",
                segments=[
                    RawSegment(start=0.0, end=1.0, text="first"),
                    RawSegment(start=1.0, end=2.0, text="second"),
                ],
            ),
        ])

        result = transcribe_audio(audio=silence(2), sr=16000, transcriber=transcriber)

        assert [segment.text for segment in result.segments] == ["first", "second"]
        assert result.segments[0].start == 0

    def test_returns_empty_result_when_every_segment_is_dropped(self) -> None:
        transcriber = FakeTranscriber([
            RawTranscriptionResult(
                transcript="unusable",
                segments=[RawSegment(start=None, end=None, text="unusable")],
            ),
        ])

        result = transcribe_audio(audio=silence(2), sr=16000, transcriber=transcriber)

        assert result.segments == []

    def test_advances_past_a_chunk_whose_segments_were_all_dropped(self) -> None:
        transcriber = FakeTranscriber(
            [
                RawTranscriptionResult(
                    transcript="unusable",
                    segments=[RawSegment(start=None, end=None, text="unusable")],
                ),
                RawTranscriptionResult(
                    transcript="hello",
                    segments=[RawSegment(start=0.0, end=0.5, text="hello")],
                ),
            ],
            ideal_segment_length=1000,
        )

        result = transcribe_audio(audio=silence(2), sr=16000, transcriber=transcriber)

        assert [segment.text for segment in result.segments] == ["hello"]
        assert result.segments[0].start == 1000
