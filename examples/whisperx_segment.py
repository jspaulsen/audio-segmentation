from pathlib import Path

from audio_segmentation import WhisperxTranscriber, WhisperxModel, transcribe_audio


def main():
    audio_fpath = Path("tests/fixtures/10m_segment.wav")

    # if the current cwd is examples, then we need to go up one level
    if Path.cwd().name == "examples":
        audio_fpath = Path("..") / audio_fpath

    if not audio_fpath.exists():
        raise FileNotFoundError(f"Audio file {audio_fpath} does not exist.")

    transcriber = WhisperxTranscriber(model_name=WhisperxModel.Large_v3, device_index=1, compute_type="float32")
    results = transcribe_audio(
        audio=audio_fpath,
        transcriber=transcriber,
        use_sentence_segmentation=True,
    )

    for segment in results:
        print(f"Segment: {segment.start} - {segment.end}, Text: {segment.text}")


if __name__ == "__main__":
    main()
