from audio_segmentation.types.segment import Segment
from audio_segmentation.refine import refine_sentence_segments


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
