import json
from pathlib import Path

import librosa
import numpy as np
import pytest

from audio_segmentation import Aligner, TorchAudioAligner, load_audio


TEN_MINUTE_PATH = Path("tests/fixtures/10m_segment.wav")
REFERENCE_PATH = Path("tests/fixtures/transcription_result.json")


def _prepare_audio_window(
    path: Path, target_sr: int, end_seconds: float
) -> np.ndarray:
    audio, sr = load_audio(path, mono=True)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio[: int(end_seconds * target_sr)]


def _select_reference_window(
    reference_segments: list[dict], max_end: float
) -> list[dict]:
    return [s for s in reference_segments if s["end"] <= max_end]


@pytest.fixture(scope="module")
def aligner() -> TorchAudioAligner:
    return TorchAudioAligner(device="cpu")


@pytest.fixture(scope="module")
def reference_segments() -> list[dict]:
    with open(REFERENCE_PATH) as f:
        return json.load(f)["segments"]


class TestProtocol:
    def test_implements_protocol(self, aligner: TorchAudioAligner) -> None:
        assert isinstance(aligner, Aligner)

    def test_required_sample_rate(self, aligner: TorchAudioAligner) -> None:
        assert aligner.required_sample_rate == 16000

    def test_align_sample_rate_mismatch_raises(
        self, aligner: TorchAudioAligner
    ) -> None:
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            aligner.align(audio, 44100, "test")

    def test_align_empty_transcript(self, aligner: TorchAudioAligner) -> None:
        audio = _prepare_audio_window(
            TEN_MINUTE_PATH, aligner.required_sample_rate, 5.0
        )
        assert aligner.align(audio, aligner.required_sample_rate, "") == []


class TestStructuralProperties:
    """Invariants that should hold for every alignment, regardless of accuracy."""

    @pytest.fixture(
        scope="class",
        params=[5.0, 60.0],
        ids=lambda v: f"{int(v)}s_window",
    )
    def windowed_alignment(
        self,
        request,
        aligner: TorchAudioAligner,
        reference_segments: list[dict],
    ):
        end = request.param
        ref = _select_reference_window(reference_segments, end)
        transcript = " ".join(s["text"] for s in ref)
        audio = _prepare_audio_window(
            TEN_MINUTE_PATH, aligner.required_sample_rate, end
        )
        segments = aligner.align(audio, aligner.required_sample_rate, transcript)
        return segments, ref, transcript, end

    def test_segment_count_matches_word_count(self, windowed_alignment) -> None:
        segments, _, transcript, _ = windowed_alignment
        assert len(segments) == len(transcript.split())

    def test_segments_preserve_original_words(self, windowed_alignment) -> None:
        segments, _, transcript, _ = windowed_alignment
        assert [s.text for s in segments] == transcript.split()

    def test_segments_have_positive_duration(self, windowed_alignment) -> None:
        segments, *_ = windowed_alignment
        for seg in segments:
            assert seg.start is not None and seg.end is not None
            assert seg.start < seg.end

    def test_segments_are_monotonic(self, windowed_alignment) -> None:
        segments, *_ = windowed_alignment
        for prev, cur in zip(segments, segments[1:]):
            assert prev.start <= cur.start
            assert prev.end <= cur.end

    def test_segments_lie_within_audio(self, windowed_alignment) -> None:
        segments, _, _, end = windowed_alignment
        # frame quantization can push the final boundary marginally past the
        # window end; allow one frame (~20ms at 50Hz) of slack.
        slack = 0.05
        for seg in segments:
            assert seg.start >= 0.0
            assert seg.end <= end + slack

    def test_segments_do_not_overlap(self, windowed_alignment) -> None:
        segments, *_ = windowed_alignment
        for prev, cur in zip(segments, segments[1:]):
            assert prev.end <= cur.start + 1e-6


LIBRITTS_DIR = Path("tests/fixtures/libritts")


class TestAlignmentAccuracy:
    """
    Boundary accuracy against MFA word alignments from LibriTTS-R.

    Errors are pooled across all fixtures so the assertions reflect aggregate
    accuracy rather than any single sample. MFA itself is not human ground
    truth, but it's a stable, well-validated reference; thresholds here gauge
    "stays in line with MFA" — tight enough to catch regressions, loose enough
    to allow normal model variance.
    """

    @pytest.fixture(scope="class")
    def boundary_errors(self, aligner: TorchAudioAligner) -> dict:
        manifest = json.loads((LIBRITTS_DIR / "manifest.json").read_text())
        pred_starts: list[float] = []
        pred_ends: list[float] = []
        ref_starts: list[float] = []
        ref_ends: list[float] = []

        for entry in manifest:
            sample_id = entry["id"]
            meta = json.loads((LIBRITTS_DIR / f"{sample_id}.json").read_text())
            audio, sr = load_audio(
                LIBRITTS_DIR / f"{sample_id}.flac",
                sr=aligner.required_sample_rate,
                mono=True,
            )
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
            transcript = " ".join(w["text"] for w in meta["words"])
            segments = aligner.align(audio.astype(np.float32), sr, transcript)
            assert len(segments) == len(meta["words"]), (
                f"sample {sample_id}: aligner produced {len(segments)} words, "
                f"reference has {len(meta['words'])}"
            )
            pred_starts.extend(s.start for s in segments)
            pred_ends.extend(s.end for s in segments)
            ref_starts.extend(w["start"] for w in meta["words"])
            ref_ends.extend(w["end"] for w in meta["words"])

        return {
            "start_err": np.abs(np.array(pred_starts) - np.array(ref_starts)),
            "end_err": np.abs(np.array(pred_ends) - np.array(ref_ends)),
            "n_words": len(pred_starts),
        }

    def test_aggregated_word_count(self, boundary_errors: dict) -> None:
        # Guard against silently shrinking the test sample set.
        assert boundary_errors["n_words"] >= 150

    # Thresholds set ~40% above observed values (start ~53ms median, end ~27ms;
    # start consistently runs higher than end due to where each tool places the
    # leading boundary relative to silence/coarticulation).
    def test_median_start_error_under_75ms(self, boundary_errors: dict) -> None:
        assert float(np.median(boundary_errors["start_err"])) < 0.075

    def test_median_end_error_under_50ms(self, boundary_errors: dict) -> None:
        assert float(np.median(boundary_errors["end_err"])) < 0.050

    def test_mean_start_error_under_80ms(self, boundary_errors: dict) -> None:
        assert float(boundary_errors["start_err"].mean()) < 0.080

    def test_mean_end_error_under_60ms(self, boundary_errors: dict) -> None:
        assert float(boundary_errors["end_err"].mean()) < 0.060

    def test_p90_boundary_error_under_150ms(self, boundary_errors: dict) -> None:
        all_err = np.concatenate(
            [boundary_errors["start_err"], boundary_errors["end_err"]]
        )
        assert float(np.percentile(all_err, 90)) < 0.150
