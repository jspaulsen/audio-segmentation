from pathlib import Path

from audio_segmentation.transcribe import transcribe_audio
from audio_segmentation import (
    load_audio,
    NemoTranscriber,
    NemoModel,
    refine_segment_timestamps,
    transcribe_audio,
)
from audio_segmentation.types.segment import Segment
from audio_segmentation.verifiers.speechbrain import SpeechBrainVerifier


def main(device_index: int = 1):
    audio_fpath = Path("tests/fixtures/10m_segment.wav")

    # if the current cwd is examples, then we need to go up one level
    if Path.cwd().name == "examples":
        audio_fpath = Path("..") / audio_fpath

    audio, sr = load_audio(audio_fpath)
    transcriber = NemoTranscriber(model_name=NemoModel.PARAKEET_TDT_V2, device_index=device_index)
    verifier = SpeechBrainVerifier(device_index=device_index)

    results = transcribe_audio(
        audio=audio,
        sr=sr,
        transcriber=transcriber,
        speaker_verifier=verifier,
        # use_sentence_segmentation=True,
    )
    
    segments: list[Segment] = []

    for segment in results.segments:
        refined_segment = refine_segment_timestamps(
            audio=audio,
            sr=sr,
            segment=segment,
            max_look_ms=500,
            min_silence_len=200,
            silence_thresh=-40,
            padding=100,
        )

        segments.append(refined_segment)

    # Output the segment to a file
    with open("example_transcription.txt", "w") as f:
        for segment in segments:
            f.write(f"{segment.speaker_id}, {segment.text}\n")
            print(f"Segment: {segment.start} - {segment.end}, Text: {segment.text}")



if __name__ == "__main__":
    main()
