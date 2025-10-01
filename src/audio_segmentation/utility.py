import io
from pathlib import Path

import pydub


def reformat_audio(audio: pydub.AudioSegment) -> pydub.AudioSegment:
    in_memory = io.BytesIO()
    audio.export(in_memory, format="wav")

    in_memory.seek(0)
    return pydub.AudioSegment.from_file(in_memory, format="wav")


def load_audio(audio: Path | str) -> pydub.AudioSegment:
    audio = Path(audio)

    if not audio.exists():
        raise FileNotFoundError(f"Audio file {audio} does not exist.")

    audio_segment: pydub.AudioSegment = pydub.AudioSegment.from_file(audio)

    # if the file isn't wav, convert it to wav
    if audio.suffix.lower() != ".wav":
        audio_segment = reformat_audio(audio_segment)

    audio_segment: pydub.AudioSegment = audio_segment.set_frame_rate(16000)
    audio_segment: pydub.AudioSegment = audio_segment.set_channels(1)
    audio_segment: pydub.AudioSegment = audio_segment.set_sample_width(2)

    return audio_segment
