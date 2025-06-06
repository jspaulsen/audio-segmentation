# Audio Segmentation

A python library for transcribing and segmenting audio files by sentences using word (or segment) level timestamps and ntlk sentence tokenization.

This library was spun out from work done to generate training data (audio segments) from audio files for TTS (Text-to-Speech) models.

## Use Case(s)

This library is designed to help with the segmentation of audio files into smaller segments based on sentences. It is particularly useful for generating training data for TTS models, where you need to have audio segments that correspond to specific sentences or phrases.

## Accuracy

The accuracy of the segmentation depends on the quality of the transcription model used. The library is designed to work with word or segment-level transcription models that provide timestamps for each word in the audio file. The segmentation is done based on these timestamps and the text is tokenized into sentences using the NLTK library.

## How it works

`audio_segmentation` generates a list of segments from an audio file, where each segment contains the start time, end time, and text of the segment. It uses a word or segment-level transcription model to get timestamps for each word in the audio file, and then uses the NLTK library to tokenize the text into sentences. The segments are then created based on the timestamps of the words in each sentence.

## Caveats

This library is still in development, undertested and may not work as expected. It also is biased towards the author's use case and may not work well for other audio files. Use at your own risk.

## Installation

To be announced. Should be able to install via uv and git.

### Dependencies

- `ffmpeg` - for audio processing

## Usage

Refer to the examples directory for more concrete usage examples.

```python
from audio_segmentation import ParakeetTranscriber, ParakeetModel, transcribe_audio

transcriber = ParakeetTranscriber(model_name=ParakeetModel.TDT_V2)
results = transcribe_audio(
    audio=audio_fpath,
    transcriber=transcriber,
)
```

## Known (Potential) Issues

- Optional dependencies aren't handled correctly and imports may fail if the optional dependencies are not installed.
