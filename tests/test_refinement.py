import numpy as np

from audio_segmentation.types.segment import Segment
from audio_segmentation.types.audio import Audio
from audio_segmentation.refine import refine_sentence_segments, refine_segment_timestamps


def _synthetic_audio(
    sr: int,
    duration_ms: int,
    speech_start_ms: int,
    speech_end_ms: int,
) -> np.ndarray:
    """Near-silence for the whole file, with louder 'speech' over the given range."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.0001, int(duration_ms * sr / 1000))

    start = int(speech_start_ms * sr / 1000)
    end = int(speech_end_ms * sr / 1000)
    data[start:end] = rng.normal(0, 0.1, end - start)

    return data


class TestRefinement:
    def test_refine_segments_combines_with_short_gap(self):
        segments = [
            Segment(start=0, end=1000, text="First segment."),
            Segment(start=1200, end=2000, text="Second segment.")  # 200ms gap (< 500ms threshold)
        ]

        refined = refine_sentence_segments(segments)

        assert len(refined) == 1
        assert refined[0].start == 0
        assert refined[0].end == 2000
        assert refined[0].text == "First segment. Second segment."

    def test_refine_segments_does_not_combine_with_long_gap(self):
        segments = [
            Segment(start=0, end=1000, text="First segment."),
            Segment(start=2000, end=3000, text="Second segment.")  # 1000ms gap (> 500ms threshold)
        ]

        refined = refine_sentence_segments(segments)

        assert len(refined) == 2
        assert refined[0].text == "First segment."
        assert refined[1].text == "Second segment."

    def test_refine_segments_does_not_combine_when_duration_too_long(self):
        segments = [
            Segment(start=0, end=6000, text="Long first segment."),
            Segment(start=6200, end=11000, text="Long second segment.")  # Combined duration would be 11000ms (> 10000ms max)
        ]

        refined = refine_sentence_segments(segments, max_segment_length_ms=10000)

        assert len(refined) == 2
        assert refined[0].text == "Long first segment."
        assert refined[1].text == "Long second segment."

    def test_refine_segments_edge_cases(self):
        # Test empty list
        assert refine_sentence_segments([]) == []

        # Test single segment
        single_segment = [Segment(start=0, end=1000, text="Only segment.")]
        refined = refine_sentence_segments(single_segment)
        assert len(refined) == 1
        assert refined[0] == single_segment[0]

    # Add a test with multiple segments; some should combine, others should not
    def test_refine_segments_mixed(self):
        segments = [
            Segment(start=0, end=1000, text="Segment 1."),
            Segment(start=1200, end=2000, text="Segment 2."),  # Combine with Segment 1
            Segment(start=3000, end=4000, text="Segment 3."),  # Do not combine (gap too large)
            Segment(start=4100, end=5000, text="Segment 4."),  # Combine with Segment 3
            Segment(start=6000, end=17000, text="Segment 5."),  # Do not combine (too long)
            Segment(start=17200, end=18000, text="Segment 6.")   # Combine with Segment 5
        ]

        refined = refine_sentence_segments(segments, max_segment_length_ms=10000)

        assert len(refined) == 4
        assert refined[0].text == "Segment 1. Segment 2."
        assert refined[1].text == "Segment 3. Segment 4."
        assert refined[2].text == "Segment 5."
        assert refined[3].text == "Segment 6."

    def test_refine_segments_multispeaker(self):
        segments = [
            Segment(start=0, end=1000, text="Segment 1.", speaker_id=None),
            Segment(start=1200, end=2500, text="Segment 2.", speaker_id=1),
            Segment(start=3000, end=3200, text="Segment 3.", speaker_id=1),  # Combine with Segment 2
            Segment(start=3400, end=5000, text="Segment 4.", speaker_id=1),
            Segment(start=5300, end=6000, text="Segment 5.", speaker_id=2),
            Segment(start=6200, end=6800, text="Segment 6.", speaker_id=2),
            Segment(start=7000, end=8000, text="Segment 7.", speaker_id=None),
        ]

        refined = refine_sentence_segments(segments, max_segment_length_ms=10000)

        assert len(refined) == 4
        assert refined[0].text == "Segment 1."
        assert refined[1].text == "Segment 2. Segment 3. Segment 4."
        assert refined[2].text == "Segment 5. Segment 6."
        assert refined[3].text == "Segment 7."

    def test_refine_segment_timestamps_detects_silence(self):
        """Test that refine_segment_timestamps finds true speech boundaries."""
        # Create synthetic audio: silence (100ms) + speech (200ms) + silence (100ms)
        sr = 16000  # 16kHz sample rate

        # Silence: very low amplitude
        silence_samples = int(0.1 * sr)  # 100ms
        silence = np.random.normal(0, 0.001, silence_samples)

        # Speech: higher amplitude
        speech_samples = int(0.2 * sr)  # 200ms
        speech = np.random.normal(0, 0.1, speech_samples)

        # Combine: silence + speech + silence
        audio_data = np.concatenate([silence, speech, silence])

        # Create a segment that's too narrow (misses some speech)
        # Speech actually runs from ~100ms to ~300ms
        # Segment from 150ms to 250ms cuts off start and end
        segment = Segment(start=150, end=250, text="Test speech")

        # Refine the segment timestamps - looks in expanded window to find true boundaries
        refined = refine_segment_timestamps(
            audio=audio_data,
            sr=sr,
            segment=segment,
            search_boundary=100,  # Look 100ms before/after
            pad=0,
        )

        # The refined segment should find the actual speech boundaries
        # Start should move earlier (toward actual speech start ~100ms)
        # End should move later (toward actual speech end ~300ms)
        assert refined.start < segment.start, "Start should be adjusted earlier to capture speech"
        assert refined.end > segment.end, "End should be adjusted later to capture speech"
        assert refined.start >= 50, "Start should be within search window"
        assert refined.end <= 350, "End should be within search window"
        assert refined.text == segment.text, "Text should be preserved"

    def test_refine_segment_timestamps_near_audio_start(self):
        """Segments within search_boundary of t=0 should still be refined."""
        # Speech runs from 50ms to 400ms in a 1s file.
        audio_data = _synthetic_audio(sr=16000, duration_ms=1000, speech_start_ms=50, speech_end_ms=400)

        # lookback would be -100ms, which must not be treated as an index from the end.
        segment = Segment(start=100, end=300, text="Test speech")

        refined = refine_segment_timestamps(
            audio=audio_data,
            sr=16000,
            segment=segment,
            search_boundary=200,
            pad=0,
        )

        assert refined.start < segment.start, "Start should be adjusted earlier to capture speech"
        assert 0 <= refined.start <= 60, "Start should land on the speech onset at ~50ms"

    def test_refine_segment_timestamps_near_audio_end(self):
        """The refined end must not run past the end of the audio."""
        # Speech runs from 400ms to 600ms in a 1s file.
        audio_data = _synthetic_audio(sr=16000, duration_ms=1000, speech_start_ms=400, speech_end_ms=600)

        # lookforward would be 1100ms, past the end of a 1000ms file.
        segment = Segment(start=500, end=900, text="Test speech")

        refined = refine_segment_timestamps(
            audio=audio_data,
            sr=16000,
            segment=segment,
            search_boundary=200,
            pad=0,
        )

        assert refined.end <= 1000, "End should not exceed the length of the audio"
        assert abs(refined.end - 600) <= 40, "End should land on the speech offset at ~600ms"

    def test_refine_segment_timestamps_pad_widens_both_edges(self):
        """pad should extend the segment on both sides, not shift it."""
        # Speech runs from 400ms to 600ms in a 1s file; the search window stays in bounds.
        audio_data = _synthetic_audio(sr=16000, duration_ms=1000, speech_start_ms=400, speech_end_ms=600)
        segment = Segment(start=300, end=700, text="Test speech")

        kwargs = dict(audio=audio_data, sr=16000, segment=segment, search_boundary=200)
        unpadded = refine_segment_timestamps(**kwargs, pad=0)
        padded = refine_segment_timestamps(**kwargs, pad=50)

        assert padded.start == unpadded.start - 50
        assert padded.end == unpadded.end + 50
