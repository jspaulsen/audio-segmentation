from pathlib import Path

from audio_segmentation.transcribe import transcribe_audio
from audio_segmentation import (
    load_audio,
    NemoTranscriber,
    NemoModel,
    transcribe_audio,
)


def main(device_index: int = 1):
    audio_fpath = Path("tests/fixtures/10m_segment.wav")

    # if the current cwd is examples, then we need to go up one level
    if Path.cwd().name == "examples":
        audio_fpath = Path("..") / audio_fpath

    audio = load_audio(audio_fpath)

    transcriber = NemoTranscriber(model_name=NemoModel.PARAKEET_TDT_V2, device_index=device_index)

    results = transcribe_audio(
        audio=audio,
        transcriber=transcriber,
        # use_sentence_segmentation=True,
    )

    # Output the segment to a file
    with open("example_transcription.txt", "w") as f:
        for segment in results.segments:
            f.write(segment.text + "\n")
            print(f"Segment: {segment.start} - {segment.end}, Text: {segment.text}")

    # Refine segments by combining those with short gaps
    refined_segments = refine_sentence_segments(results.segments, max_segment_length_ms=15000)
    with open("example_refined_transcription.txt", "w") as f:
        for segment in refined_segments:
            f.write(segment.text + "\n")
            print(f"Refined Segment: {segment.start} - {segment.end}, Text: {segment.text}")


if __name__ == "__main__":
    main()
