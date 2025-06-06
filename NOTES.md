We should look at using Parakeet RNNT 1.1B instead of the 0.6B TDT model for word level transcription: https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/

Additionally: If we're going to continue to use the TDT model, we should reduce the segmentation length to 8 minutes and not use the "Limited Context Attention" option.
