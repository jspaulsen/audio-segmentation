from audio_segmentation.types.segment import Segment


class TestSegmentCombine:
    def test_combine_adjacent_segments(self):
        combined = Segment(start=0, end=1000, text="hello").combine(
            Segment(start=1200, end=2000, text="world")
        )

        assert combined.start == 0
        assert combined.end == 2000
        assert combined.text == "hello world"

    def test_combine_is_ordered_by_start_time(self):
        combined = Segment(start=1200, end=2000, text="world").combine(
            Segment(start=0, end=1000, text="hello")
        )

        assert combined.start == 0
        assert combined.end == 2000
        assert combined.text == "hello world"

    def test_combine_keeps_both_texts_when_one_segment_contains_the_other(self):
        combined = Segment(start=0, end=1000, text="hello").combine(
            Segment(start=100, end=500, text="world")
        )

        assert combined.start == 0
        assert combined.end == 1000
        assert combined.text == "hello world"
