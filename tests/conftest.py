from pathlib import Path
import pydub
import pytest


@pytest.fixture
def ten_minute_segment_path() -> Path:
    return Path('tests/fixtures/10m_segment.wav')


@pytest.fixture
def ten_minute_segment(ten_minute_segment_path) -> pydub.AudioSegment:
    """
    Fixture that provides a 10-minute audio segment.
    This is used to test the transcriber with a longer audio segment.
    """
    audio = pydub.AudioSegment.from_file(
        ten_minute_segment_path,
        format='wav',
    )

    audio = audio.set_frame_rate(16000)  # Set to 16kHz
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)

    return audio
