import tempfile
from io import BytesIO
from typing import Optional

import torch
from cog import BasePredictor, Input, Path
from torch.cuda import is_available as is_cuda_available

from demucs.api import Separator
from demucs.apply import BagOfModels
from demucs.audio import save_audio
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model

# 预加载的 Demucs 模型列表
DEMUCS_MODELS = [
    "htdemucs",
]


class PreloadedSeparator(Separator):
    """
    为了提高效率，此类将模型保存在内存中，避免每次请求都需要加载模型。
    """

    def __init__(
        self,
        model: BagOfModels,
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: Optional[int] = None,
        jobs: int = 0,
    ):
        self._model = model
        self._audio_channels = model.audio_channels
        self._samplerate = model.samplerate

        self.update_parameter(
            device="cuda" if is_cuda_available() else "cpu",
            shifts=shifts,
            overlap=overlap,
            split=split,
            segment=segment,
            jobs=jobs,
            progress=True,
            callback=None,
            callback_arg=None,
        )


class Predictor(BasePredictor):
    """
    实现 Cog API 的预测器，用于推理 Demucs 模型。
    """

    def setup(self):
        """
        预先加载模型以提高连续请求时的预测速度。
        """
        self.models = {model: get_model(model) for model in DEMUCS_MODELS}

    def predict(
        self,
        audio: Path = Input(description="上传需要处理的音频文件。"),
    ) -> dict:
        # 固定的参数值
        model_name = "htdemucs"
        stem = "vocals"
        output_format = "mp3"
        mp3_bitrate = 64
        mp3_preset = 4
        clip_mode = "rescale"
        shifts = 1
        overlap = 0.25
        split = True
        segment = None
        jobs = 0

        # 使用预加载的模型
        model = self.models[model_name]

        if stem != "none" and stem not in model.sources:
            raise ValueError(
                f"选择的 stem '{stem}' 不支持所选的模型。"
            )

        max_allowed_segment = float("inf")
        if isinstance(model, HTDemucs):
            max_allowed_segment = float(model.segment)
        elif isinstance(model, BagOfModels):
            max_allowed_segment = model.max_allowed_segment

        if segment is not None and segment > max_allowed_segment:
            raise ValueError(
                f"不能使用比模型训练时更长的 segment。最大允许的 segment 是 {max_allowed_segment}。"
            )

        separator = PreloadedSeparator(
            model=model,
            shifts=shifts,
            overlap=overlap,
            segment=segment,
            split=split,
            jobs=jobs,
        )

        _, outputs = separator.separate_audio_file(audio)

        kwargs = {
            "samplerate": separator.samplerate,
            "bitrate": mp3_bitrate,
            "preset": mp3_preset,
            "clip": clip_mode,
            "as_float": False,
            "bits_per_sample": 24,
        }

        output_stems = {}

        if stem == "none":
            for name, source in outputs.items():
                with tempfile.NamedTemporaryFile(suffix=f".{output_format}") as f:
                    save_audio(source.cpu(), f.name, **kwargs)
                    output_stems[name] = BytesIO(open(f.name, "rb").read())
        else:
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}") as f:
                save_audio(outputs[stem].cpu(), f.name, **kwargs)
                output_stems[stem] = BytesIO(open(f.name, "rb").read())

            other_stem = torch.zeros_like(outputs[stem])
            for source, audio in outputs.items():
                if source != stem:
                    other_stem += audio

            with tempfile.NamedTemporaryFile(suffix=f".{output_format}") as f:
                save_audio(other_stem.cpu(), f.name, **kwargs)
                output_stems["no_" + stem] = BytesIO(open(f.name, "rb").read())

        return output_stems
