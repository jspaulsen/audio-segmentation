import copy
from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchMultiTaskAED, AudioFeatureIterator
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    get_buffered_pred_feat_multitaskAED,
    setup_model,
    write_transcription,
    wrap_transcription
)
from nemo.collections.asr.parts.utils import rnnt_utils
import numpy as np
import pydub
import torch


torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")


class ExtendedFrameBatchMultiTaskAED(FrameBatchMultiTaskAED):
    def populate_frame_reader(
        self, 
        segment: pydub.AudioSegment,
        sample_rate: int,
        model_stride_in_secs: float, # TODO: Verify
        delay: float = 0.0,  # TODO: Verify
    ) -> None:
        segment = segment.set_frame_rate(sample_rate)
        samples = segment.get_array_of_samples()
        samples = np.array(samples, dtype=np.float32) / np.iinfo(np.int16).max
        samples = samples.transpose()
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * sample_rate)))
        frame_reader = AudioFeatureIterator(
            samples, 
            self.frame_len, 
            self.raw_preprocessor, 
            self.asr_model.device, 
            pad_to_frame_len=False
        )

        self.set_frame_reader(frame_reader)
    
    def set_input_tokens(self, meta_data: dict) -> None:
        self.input_tokens = self.get_input_tokens(meta_data)


def get_buffered_pred_feat(
    asr: ExtendedFrameBatchMultiTaskAED,
    preprocessor_cfg: dict,
    model_stride_in_secs: int,
    device: str,
    audio_segment: pydub.AudioSegment,
    delay: float = 0.0,
    timestamps: bool = False,
) -> list[rnnt_utils.Hypothesis]:
    # TODO: Is this preprocessor actually used?
    preprocessor_cfg.normalize = "None"
    preprocessor = EncDecMultiTaskModel.from_config_dict(preprocessor_cfg)
    preprocessor.to(device)

    hyps = []
    refs = []

    meta = {
        # 'audio_filepath': audio_file,
        # 'duration': 100000,
        'source_lang': 'en',
        'target_lang': 'en',
        'taskname': 'asr',
        'pnc': 'yes',
        'answer': 'nothing',
        'timestamp': 'yes' if timestamps else 'no',
    }

    asr.reset()
    asr.populate_frame_reader(
        audio_segment,
        sample_rate=preprocessor_cfg.sample_rate,
        model_stride_in_secs=model_stride_in_secs,
        delay=delay,
    )

    asr.set_input_tokens(meta_data=meta)

    # asr.read_audio_file(audio_file, delay, model_stride_in_secs, meta_data=meta)
    hyp = asr.transcribe()
    hyps.append(hyp)

    wrapped_hyps = wrap_transcription(hyps)
    return wrapped_hyps

    


# TODO: When we configure / reset the model, we need to call both
# populate_frame_reader and set_input_tokens 
# to ensure the model is ready for inference with the new audio segment.


# load model
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-flash')

#     torch.set_float32_matmul_precision(cfg.matmul_precision)

# update decode params
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 1
canary_model.change_decoding_strategy(decode_cfg)

canary_model.eval()
canary_model = canary_model.to(canary_model.device)


model_cfg = copy.deepcopy(canary_model._cfg)
# some changes for streaming scenario
model_cfg.preprocessor.dither = 0.0
model_cfg.preprocessor.pad_to = 0

feature_stride = model_cfg.preprocessor['window_stride']
model_stride_in_secs = feature_stride * 8 # cfg.model_stride


# TODO: Need to replace read_audio_file with a version
# that already has the audio data.
frame_asr = ExtendedFrameBatchMultiTaskAED(
    asr_model=canary_model,
    frame_len=10,
    total_buffer=10,
    batch_size=1 # 8 in the config
)

amp_dtype = torch.float16 # if cfg.amp_dtype == "float16" else torch.bfloat16


with torch.amp.autocast(canary_model.device.type, enabled=False, dtype=amp_dtype):
    with torch.no_grad():
        results = get_buffered_pred_feat(
            asr=frame_asr,
            preprocessor_cfg=model_cfg.preprocessor,
            model_stride_in_secs=model_stride_in_secs,
            device=canary_model.device,
            audio_segment=pydub.AudioSegment.from_file('tests/fixtures/10m_segment.wav'),
            delay=0.0,  # TODO: Verify if this is needed
            timestamps=True,  # TODO: Verify if this is needed
        )

print(results)

# def read_audio_file(self, audio_filepath: str, delay, model_stride_in_secs, meta_data):
#     self.input_tokens = self.get_input_tokens(meta_data)
#     samples = get_samples(audio_filepath)
#     samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
#     frame_reader = AudioFeatureIterator(
#         samples, self.frame_len, self.raw_preprocessor, self.asr_model.device, pad_to_frame_len=False
#     )
#     self.set_frame_reader(frame_reader)



# hyps = get_buffered_pred_feat_multitaskAED(
#     frame_asr,
#     model_cfg.preprocessor,
#     model_stride_in_secs,
#     asr_model.device,
#     manifest,
#     filepaths,
#     timestamps=cfg.timestamps,
# )



# TODO:
# Need to replace this function with one that isn't using tqdm lolol
# def get_buffered_pred_feat_multitaskAED(
#     asr: FrameBatchMultiTaskAED,
#     preprocessor_cfg: DictConfig,
#     model_stride_in_secs: int,
#     device: Union[List[int], int],
#     manifest: str = None,
#     filepaths: List[list] = None,
#     delay: float = 0.0,
#     timestamps: bool = False,
# ) -> List[rnnt_utils.Hypothesis]:
#     # Create a preprocessor to convert audio samples into raw features,
#     # Normalization will be done per buffer in frame_bufferer
#     # Do not normalize whatever the model's preprocessor setting is
#     preprocessor_cfg.normalize = "None"
#     preprocessor = EncDecMultiTaskModel.from_config_dict(preprocessor_cfg)
#     preprocessor.to(device)
#     hyps = []
#     refs = []

#     if filepaths and manifest:
#         raise ValueError("Please select either filepaths or manifest")
#     if filepaths is None and manifest is None:
#         raise ValueError("Either filepaths or manifest shoud not be None")

#     if filepaths:
#         logging.info(
#             "Deteced audio files as input, default to English ASR with Punctuation and Capitalization output. \
#                 Please use manifest input for other options."
#         )
#         for audio_file in tqdm(filepaths, desc="Transcribing:", total=len(filepaths), ncols=80):
#             meta = {
#                 'audio_filepath': audio_file,
#                 'duration': 100000,
#                 'source_lang': 'en',
#                 'taskname': 'asr',
#                 'target_lang': 'en',
#                 'pnc': 'yes',
#                 'answer': 'nothing',
#                 'timestamp': 'yes' if timestamps else 'no',
#             }
#             asr.reset()
#             asr.read_audio_file(audio_file, delay, model_stride_in_secs, meta_data=meta)
#             hyp = asr.transcribe()
#             hyps.append(hyp)
#     else:
#         with open(manifest, "r", encoding='utf_8') as fin:
#             lines = list(fin.readlines())
#             for line in tqdm(lines, desc="Transcribing:", total=len(lines), ncols=80):
#                 asr.reset()
#                 line = line.strip()
#                 if not line:
#                     continue
#                 sample = json.loads(line)
#                 if (
#                     timestamps
#                 ):  # user convenience so that they don't need to make another manifest with timestamp field or modify the existing one
#                     sample['timestamp'] = 'yes'
#                 if 'text' in sample:
#                     refs.append(sample['text'])
#                 audio_file = get_full_path(audio_file=sample['audio_filepath'], manifest_file=manifest)
#                 # do not support partial audio
#                 asr.read_audio_file(audio_file, delay, model_stride_in_secs, meta_data=sample)
#                 hyp = asr.transcribe()
#                 hyps.append(hyp)

#     wrapped_hyps = wrap_transcription(hyps)
#     return wrapped_hyps


# hyps = get_buffered_pred_feat_multitaskAED(
#     frame_asr,
#     model_cfg.preprocessor,
#     model_stride_in_secs,
#     canary_model.device,
#     None,
#     # manifest,
#     filepaths,
#     timestamps=cfg.timestamps,
# )

# run inference
# audio = 'tests/fixtures/10m_segment.wav'


# transcription = canary_model.transcribe([audio], batch_size=1, pnc='yes')
# print(transcription)
