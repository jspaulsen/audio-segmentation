from enum import StrEnum
import logging
from typing import Protocol, TypedDict, cast

from nemo.collections.asr.models import ASRModel, EncDecRNNTBPEModel
import numpy as np
import pydub

from audio_segmentation.segment import RawSegment
from audio_segmentation.transcriber.transcriber import Transcriber, TranscriptionResult


nemo_logger = logging.getLogger("nemo_logger")
nemo_logger.setLevel(logging.ERROR)  # Disable Nemo logging


class ParakeetModel(StrEnum):
    RNNT = "nvidia/parakeet-rnnt-1.1b"
    TDT_V2 = "nvidia/parakeet-tdt-0.6b-v2"


class ParakeetWordResult(TypedDict):
    word: str
    start: float
    end: float


class ParakeetSegmentResult(TypedDict):
    start: float
    end: float
    segment: str


class ParakeetTimestampResult(TypedDict):
    # char: list[dict]
    word: list[ParakeetWordResult]
    segment: list[ParakeetSegmentResult]


class ParakeetTranscriptResult(Protocol):
    timestamp: ParakeetTimestampResult
    text: str


# NOTE:
# This (apparently) sets up parakeet for inference on long (> 8min) audio segments.
# asr_model = asr_model.to(device)
# asr_model = asr_model.to(torch.float32)

# asr_model.change_attention_model("rel_pos_local_attn", [256, 256])
# asr_model.change_subsampling_conv_chunking_factor(1)  # 1
# asr_model = asr_model.to(torch.float16)


class ParakeetTranscriber(Transcriber):
    def __init__(
        self,
        model_name: ParakeetModel = ParakeetModel.TDT_V2,
        device_index: int | None = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.model: EncDecRNNTBPEModel = cast(
            EncDecRNNTBPEModel,
            ASRModel.from_pretrained(
                model_name=model_name.value,
                map_location=f"cuda:{device_index}" if device_index is not None else None, # type: ignore
            ),
        )

        self.verbose = verbose
    
    @property
    def ideal_segment_length(self) -> int | None:
        if self.model_name == ParakeetModel.TDT_V2:
            return 60 * 8 * 1000 # 8 minutes in seconds

        return None
    
    def transcribe(
        self, 
        audio_segment: pydub.AudioSegment,
        word_level_segmentation: bool = True,
     ) -> TranscriptionResult:
        data = np.array(audio_segment.get_array_of_samples())
        data = data.astype(np.float32) / np.iinfo(np.int16).max
        key = 'word' if word_level_segmentation else 'segment'

        outputs: list[ParakeetTranscriptResult] = cast(
            list[ParakeetTranscriptResult], 
            self.model.transcribe(
                data, 
                timestamps=True,
                verbose=self.verbose,
            )
        )

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
        
        return TranscriptionResult(
            transcript=output.text,
            segments=ret,
        )
