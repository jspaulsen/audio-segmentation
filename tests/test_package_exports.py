import importlib.util

import audio_segmentation


class TestPackageExports:
    def test_star_import_does_not_raise(self):
        namespace: dict = {}

        exec("from audio_segmentation import *", namespace)

        assert "transcribe_audio" in namespace

    def test_every_exported_name_is_available(self):
        missing = [
            name for name in audio_segmentation.__all__
            if not hasattr(audio_segmentation, name)
        ]

        assert missing == []

    def test_optional_names_are_exported_only_when_their_library_is_installed(self):
        optional = {
            "nemo": ("NemoModel", "NemoTranscriber"),
            "whisperx": ("WhisperxModel", "WhisperxTranscriber"),
            "speechbrain": ("SpeechBrainVerifier",),
        }

        for library, names in optional.items():
            installed = importlib.util.find_spec(library) is not None

            for name in names:
                assert (name in audio_segmentation.__all__) == installed, (
                    f"{name} exported={name in audio_segmentation.__all__}, {library} installed={installed}"
                )
