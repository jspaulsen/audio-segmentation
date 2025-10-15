import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # NOTE: Change this based on your GPU availability


import torchaudio

from audio_segmentation.verifiers.speechbrain import SpeechBrainVerifier


def main():
    verifier = SpeechBrainVerifier(device_index=0) # NOTE: Change this based on your GPU availability

    audio, sr = torchaudio.load("tests/fixtures/glimpsed_14.wav")
    second_audio, nsr = torchaudio.load("tests/fixtures/glimpsed_45.wav")

    # Make mono and 16kHz if needed
    if verifier.requires_mono_audio and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if sr != verifier.required_sample_rate:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=verifier.required_sample_rate)

    if verifier.requires_mono_audio and second_audio.shape[0] > 1:
        second_audio = second_audio.mean(dim=0, keepdim=True)

    if nsr != verifier.required_sample_rate:
        second_audio = torchaudio.functional.resample(second_audio, orig_freq=nsr, new_freq=verifier.required_sample_rate)

    embedding = verifier.create_embedding(audio)
    second_embedding = verifier.create_embedding(second_audio)

    print(embedding.shape)
    print(second_embedding.shape)
    score, result = verifier.compute_similarity(embedding, second_embedding)
    # result = verifier.model.verify_batch(audio, second_audio)

    print(f"Similarity score: {score}")
    print(f"Verification result: {result}")

if __name__ == "__main__":
    main()
