from pathlib import Path
import numpy as np
import pytest

from audio_segmentation.utility import load_audio


@pytest.fixture
def ten_minute_segment_path() -> Path:
    return Path('tests/fixtures/10m_segment.wav')


@pytest.fixture
def ten_minute_segment(ten_minute_segment_path) -> tuple[np.ndarray, int]:
    """
    Fixture that provides a 10-minute audio segment.
    This is used to test the transcriber with a longer audio segment.
    """
    return load_audio(ten_minute_segment_path, sr=16000, mono=True)


@pytest.fixture
def glimpsed_14_path() -> Path:
    return Path('tests/fixtures/glimpsed_14.wav')


@pytest.fixture
def glimpsed_45_path() -> Path:
    return Path('tests/fixtures/glimpsed_45.wav')


@pytest.fixture
def five_second_segment_path() -> Path:
    return Path('tests/fixtures/5s_segment.wav')
