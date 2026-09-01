import numpy as np

from audio_segmentation.types.segment import Segment
from audio_segmentation.types.audio import Audio
from audio_segmentation.refine import (
    refine_segment_timestamps,
    refine_segment_timestamps_batch,
    refine_sentence_segments,
)


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


def _multi_speech_audio(
    sr: int,
    duration_ms: int,
    spans_ms: list[tuple[int, int]],
) -> np.ndarray:
    """Near-silence for the whole file, with louder 'speech' over each given range."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.0001, int(duration_ms * sr / 1000))

    for speech_start_ms, speech_end_ms in spans_ms:
        start = int(speech_start_ms * sr / 1000)
        end = int(speech_end_ms * sr / 1000)
        data[start:end] = rng.normal(0, 0.1, end - start)

    return data


class TestNeighbourAwareRefinement:
    def test_end_does_not_run_into_the_next_segment(self):
        """A search_boundary wider than the gap must not snap the end onto the neighbour."""
        # Two utterances with only a 40ms gap between them.
        audio_data = _multi_speech_audio(sr=16000, duration_ms=2000, spans_ms=[(200, 800), (840, 1500)])

        segment = Segment(start=250, end=750, text="First")
        following = Segment(start=840, end=1500, text="Second")

        # 200ms of lookforward from end=750 reaches 950ms, well inside the second utterance.
        unaware = refine_segment_timestamps(audio=audio_data, sr=16000, segment=segment, search_boundary=200)
        aware = refine_segment_timestamps(
            audio=audio_data,
            sr=16000,
            segment=segment,
            search_boundary=200,
            next_segment=following,
        )

        # Without neighbour awareness the reversed detector fires on the second
        # utterance and drags the end across the gap.
        assert unaware.end > following.start, "precondition: the unaware call is expected to corrupt the end"

        assert aware.end <= (segment.end + following.start) // 2, "End must stay on this side of the gap midpoint"
        assert aware.end < following.start, "End must never reach the next segment's start"
        assert abs(aware.end - 800) <= 40, "End should land on this utterance's offset at ~800ms"

    def test_start_does_not_run_into_the_previous_segment(self):
        audio_data = _multi_speech_audio(sr=16000, duration_ms=2000, spans_ms=[(200, 800), (840, 1500)])

        preceding = Segment(start=200, end=800, text="First")
        segment = Segment(start=900, end=1450, text="Second")

        aware = refine_segment_timestamps(
            audio=audio_data,
            sr=16000,
            segment=segment,
            search_boundary=200,
            previous_segment=preceding,
        )

        assert aware.start >= (preceding.end + segment.start) // 2, "Start must stay on this side of the gap midpoint"
        assert aware.start > preceding.end, "Start must never reach into the previous segment"
        assert abs(aware.start - 840) <= 40, "Start should land on this utterance's onset at ~840ms"

    def test_clamping_emits_a_warning(self, caplog):
        audio_data = _multi_speech_audio(sr=16000, duration_ms=2000, spans_ms=[(200, 800), (840, 1500)])

        segment = Segment(start=250, end=750, text="First")
        following = Segment(start=840, end=1500, text="Second")

        with caplog.at_level("WARNING", logger="audio_segmentation.refine"):
            refine_segment_timestamps(
                audio=audio_data,
                sr=16000,
                segment=segment,
                search_boundary=200,
                next_segment=following,
            )

        assert any("search_boundary" in record.message for record in caplog.records), (
            "Clamping against a neighbour must not be silent"
        )

    def test_no_warning_when_the_gap_is_wide_enough(self, caplog):
        audio_data = _multi_speech_audio(sr=16000, duration_ms=3000, spans_ms=[(200, 800), (1800, 2400)])

        segment = Segment(start=250, end=750, text="First")
        following = Segment(start=1800, end=2400, text="Second")

        with caplog.at_level("WARNING", logger="audio_segmentation.refine"):
            refine_segment_timestamps(
                audio=audio_data,
                sr=16000,
                segment=segment,
                search_boundary=200,
                next_segment=following,
            )

        assert not caplog.records, "A gap wider than the search boundary should not warn"

    def test_overlapping_neighbours_collapse_the_window_instead_of_inverting(self):
        """If neighbours already overlap the segment, the window must not invert."""
        audio_data = _multi_speech_audio(sr=16000, duration_ms=2000, spans_ms=[(200, 1500)])

        preceding = Segment(start=200, end=900, text="First")
        segment = Segment(start=800, end=1200, text="Second")
        following = Segment(start=1100, end=1500, text="Third")

        refined = refine_segment_timestamps(
            audio=audio_data,
            sr=16000,
            segment=segment,
            search_boundary=200,
            previous_segment=preceding,
            next_segment=following,
        )

        assert refined.start >= segment.start, "Window must not expand past an overlapping previous segment"
        assert refined.end <= segment.end, "Window must not expand past an overlapping next segment"
        assert refined.start <= refined.end, "The refined segment must not be inverted"

    def test_neighbourless_call_is_unchanged(self):
        """Omitting neighbours must behave exactly as before."""
        audio_data = _synthetic_audio(sr=16000, duration_ms=1000, speech_start_ms=400, speech_end_ms=600)
        segment = Segment(start=300, end=700, text="Test speech")

        kwargs = dict(audio=audio_data, sr=16000, segment=segment, search_boundary=200)

        assert refine_segment_timestamps(**kwargs) == refine_segment_timestamps(
            **kwargs,
            previous_segment=None,
            next_segment=None,
        )


class TestBatchRefinement:
    def test_batch_wires_neighbours_and_prevents_overlap(self):
        audio_data = _multi_speech_audio(
            sr=16000,
            duration_ms=3000,
            spans_ms=[(200, 800), (840, 1400), (1440, 2000)],
        )

        segments = [
            Segment(start=250, end=750, text="First"),
            Segment(start=900, end=1350, text="Second"),
            Segment(start=1500, end=1950, text="Third"),
        ]

        refined = refine_segment_timestamps_batch(
            audio=audio_data,
            sr=16000,
            segments=segments,
            search_boundary=200,
        )

        assert len(refined) == len(segments)
        assert [s.text for s in refined] == ["First", "Second", "Third"]

        for earlier, later in zip(refined, refined[1:]):
            assert earlier.end <= later.start, f"{earlier} overlaps {later}"

    def test_batch_uses_original_neighbours_not_refined_ones(self):
        """Refinement is order-independent: each segment sees the original neighbours."""
        audio_data = _multi_speech_audio(
            sr=16000,
            duration_ms=3000,
            spans_ms=[(200, 800), (840, 1400), (1440, 2000)],
        )

        segments = [
            Segment(start=250, end=750, text="First"),
            Segment(start=900, end=1350, text="Second"),
            Segment(start=1500, end=1950, text="Third"),
        ]

        batched = refine_segment_timestamps_batch(
            audio=audio_data,
            sr=16000,
            segments=segments,
            search_boundary=200,
        )

        individually = [
            refine_segment_timestamps(
                audio=audio_data,
                sr=16000,
                segment=segment,
                search_boundary=200,
                previous_segment=segments[index - 1] if index > 0 else None,
                next_segment=segments[index + 1] if index + 1 < len(segments) else None,
            )
            for index, segment in enumerate(segments)
        ]

        assert batched == individually

    def test_batch_on_empty_input(self):
        assert refine_segment_timestamps_batch(audio=np.zeros(16000), sr=16000, segments=[]) == []

    def test_batch_on_a_single_segment_matches_the_neighbourless_call(self):
        audio_data = _synthetic_audio(sr=16000, duration_ms=1000, speech_start_ms=400, speech_end_ms=600)
        segment = Segment(start=300, end=700, text="Only")

        batched = refine_segment_timestamps_batch(audio=audio_data, sr=16000, segments=[segment], search_boundary=200)
        single = refine_segment_timestamps(audio=audio_data, sr=16000, segment=segment, search_boundary=200)

        assert batched == [single]

    def test_pad_cannot_push_boundaries_outside_the_clamped_window(self):
        """pad must not undo neighbour clamping and re-introduce overlap."""
        audio_data = _multi_speech_audio(
            sr=16000,
            duration_ms=3000,
            spans_ms=[(200, 800), (840, 1400), (1440, 2000)],
        )

        segments = [
            Segment(start=250, end=750, text="First"),
            Segment(start=900, end=1350, text="Second"),
            Segment(start=1500, end=1950, text="Third"),
        ]

        refined = refine_segment_timestamps_batch(
            audio=audio_data,
            sr=16000,
            segments=segments,
            search_boundary=200,
            pad=100,  # wider than every gap in this fixture
        )

        for earlier, later in zip(refined, refined[1:]):
            assert earlier.end <= later.start, f"{earlier} overlaps {later} once padded"

    def test_pad_cannot_push_the_start_below_zero(self):
        audio_data = _synthetic_audio(sr=16000, duration_ms=1000, speech_start_ms=0, speech_end_ms=400)
        segment = Segment(start=10, end=300, text="Test speech")

        refined = refine_segment_timestamps_batch(
            audio=audio_data,
            sr=16000,
            segments=[segment],
            search_boundary=200,
            pad=200,
        )

        assert refined[0].start >= 0, "A padded start must not become negative"
