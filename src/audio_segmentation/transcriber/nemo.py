from enum import StrEnum
import logging
from typing import Protocol, TypedDict, cast

from nemo.collections.asr.models import EncDecRNNTBPEModel, EncDecMultiTaskModel
import numpy as np
import pydub

from audio_segmentation.types.segment import RawSegment
from audio_segmentation.transcriber.transcriber import Transcriber, RawTranscriptionResult


logger = logging.getLogger(__name__)
nemo_logger = logging.getLogger("nemo_logger")
nemo_logger.setLevel(logging.ERROR)  # Disable Nemo logging


class NemoModel(StrEnum):
    PARAKEET_TDT_V2 = "nvidia/parakeet-tdt-0.6b-v2"

    # TODO: canary-1b-flash seems to have issues. In testing, with a simplified script,
    # it hallucinates and does not produce correct transcriptions.
    # https://huggingface.co/nvidia/canary-1b-flash/discussions/10
    # It _might_ work with 10 second audio segments, but it is not reliable.
    CANARY_1B_FLASH = "nvidia/canary-1b-flash"


class NemoWordResult(TypedDict):
    word: str
    start: float
    end: float


class NemoSegmentResult(TypedDict):
    start: float
    end: float
    segment: str


class NemoTimestampResult(TypedDict):
    word: list[NemoWordResult]
    segment: list[NemoSegmentResult]


class NemoTranscriptResult(Protocol):
    timestamp: NemoTimestampResult
    text: str


class NemoTranscribeProtocol(Protocol):
    def transcribe(
        self,
        audio: np.ndarray,
    ) -> list[NemoTranscriptResult]:
        ...

    @property
    def includes_punctuation(self) -> bool:
        ...


class InternalParakeetTranscriber:
    def __init__(
        self,
        model_name: NemoModel = NemoModel.PARAKEET_TDT_V2,
        device_index: int | None = None,
        batch_size: int = 1,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model: EncDecRNNTBPEModel = EncDecRNNTBPEModel.from_pretrained(
            model_name=model_name.value,
            map_location=f"cuda:{device_index}" if device_index is not None else None,  # type: ignore
        )

        self.includes_punctuation = True

        # NOTE: parakeet-tdt-v2
        # I think the accuracy is lower (I forget the details) when using this method:
        # This (apparently) sets up parakeet tdt for inference on long (> 8min) audio segments.
        # asr_model = asr_model.to(device)
        # asr_model = asr_model.to(torch.float32)

        # asr_model.change_attention_model("rel_pos_local_attn", [256, 256])
        # asr_model.change_subsampling_conv_chunking_factor(1)  # 1
        # asr_model = asr_model.to(torch.float16)

    def transcribe(
        self,
        audio: np.ndarray,
        verbose: bool = False,
    ) -> list[NemoTranscriptResult]:
        return cast(
            list[NemoTranscriptResult],
            self.model.transcribe(
                audio,
                timestamps=True,
                verbose=verbose,
                batch_size=self.batch_size,
            )
        )


class InternalCanaryTranscriber:
    def __init__(
        self,
        model_name: NemoModel = NemoModel.CANARY_1B_FLASH,
        device_index: int | None = None,
        batch_size: int = 1,
    ) -> None:
        self.model_name = model_name
        self.device_index = device_index
        self.batch_size = batch_size

        self.model: EncDecMultiTaskModel = EncDecMultiTaskModel.from_pretrained(
            model_name=model_name.value,
            map_location=f"cuda:{device_index}" if device_index is not None else None,  # type: ignore
        )

        # TODO: This only works for 10 second audio segments and requires a more complex setup for longer segments.
        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        self.model.change_decoding_strategy(decode_cfg)

    @property
    def includes_punctuation(self) -> bool:
        return True

    def transcribe(
        self,
        audio: np.ndarray,
        verbose: bool = False,
    ) -> list[NemoTranscriptResult]:
        """
        Transcribe the audio using the Canary model.
        """
        return cast(
            list[NemoTranscriptResult],
            self.model.transcribe(
                audio,
                timestamps=True,
                verbose=verbose,
                batch_size=self.batch_size,

                source_lang='en',  # en: English, es: Spanish, fr: French, de: German
                target_lang='en',  # should be same as "source_lang" for 'asr'

                # Intentionally a string, as the model expects it [why?]
                pnc='yes',  # Punctuation and capitalization
            )
        )


class NemoTranscriber(Transcriber):
    def __init__(
        self,
        model_name: NemoModel = NemoModel.PARAKEET_TDT_V2,
        device_index: int | None = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.verbose = verbose
        self.device_index = device_index

        match model_name:
            case NemoModel.PARAKEET_TDT_V2:
                self.model = InternalParakeetTranscriber(
                    model_name=model_name,
                    device_index=device_index,
                )

            case NemoModel.CANARY_1B_FLASH:
                self.model = InternalCanaryTranscriber(
                    model_name=model_name,
                    device_index=device_index,
                )

                logger.warning(
                    "Canary model is not recommended for long audio segments. "
                    "It is known to hallucinate and produce incorrect transcriptions. "
                    "Use Parakeet TDT V2 for longer segments."
                )

            case _:
                raise ValueError(f"Unsupported model name: {model_name}")

        super().__init__()

    @property
    def ideal_segment_length(self) -> int | None:
        if self.model_name == NemoModel.PARAKEET_TDT_V2:
            return 60 * 8 * 1000 # 8 minutes in seconds

        if self.model_name == NemoModel.CANARY_1B_FLASH:
            return 10 * 1000 # 10 seconds in milliseconds

        return None

    @property
    def requires_mono_audio(self) -> bool:
        return True

    @property
    def required_sample_rate(self) -> int | None:
        return 16000

    @property
    def supports_word_level_segmentation(self) -> bool:
        """
        Returns whether the transcriber supports word level segmentation.
        """
        return True

    @property
    def includes_punctuation(self) -> bool:
        """
        Returns whether the transcriber includes punctuation in the transcription.
        """
        return self.model.includes_punctuation

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int,
        **kwargs,
     ) -> RawTranscriptionResult:
        key = 'word'  # Always use word-level segmentation
        # key = 'word' if word_level_segmentation else 'segment'

        outputs = self.model.transcribe(audio, verbose=self.verbose)
        output = outputs[0]
        ret = []

        for segment in output.timestamp[key]:
            ret.append(
                RawSegment(
                    start=segment['start'],
                    end=segment['end'],
                    text=segment[key], # type: ignore
                )
            )

        return RawTranscriptionResult(
            transcript=output.text,
            segments=ret,
        )
