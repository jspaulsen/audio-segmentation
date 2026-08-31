from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    start: int
    """
    Start time in milliseconds
    """

    end: int
    """
    End time in milliseconds
    """

    text: str
    """
    Transcribed text for this segment
    """

    speaker_id: int | None = None
    """
    Optional speaker ID for this segment
    """

    @property
    def duration(self) -> int:
        """
        Duration in milliseconds
        """
        return self.end - self.start

    def combine(self, other: Segment) -> Segment:
        """
        Combine two segments into one, merging their text and adjusting start/end times.
        """
        first, second = (self, other) if self.start <= other.start else (other, self)

        return Segment(
            start=first.start,
            end=max(self.end, other.end),
            text=first.text + " " + second.text,
            speaker_id=self.speaker_id,
        )


@dataclass
class RawSegment:
    """
    Raw segment data with optional start and end times.
    """

    start: float | None
    end: float | None
    text: str
