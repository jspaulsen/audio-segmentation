from pathlib import Path
import pytest


@pytest.fixture
def ten_minute_segment_path() -> Path:
    return Path('tests/fixtures/10m_segment.wav')

@pytest.fixture
def glimpsed_14_path() -> Path:
    return Path('tests/fixtures/glimpsed_14.wav')


@pytest.fixture
def glimpsed_45_path() -> Path:
    return Path('tests/fixtures/glimpsed_45.wav')


@pytest.fixture
def five_second_segment_path() -> Path:
    return Path('tests/fixtures/5s_segment.wav')
