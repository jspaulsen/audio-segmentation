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


@dataclass
class RawSegment:
    """
    Raw segment data with optional start and end times.
    """
    
    start: float | None
    end: float | None
    text: str
