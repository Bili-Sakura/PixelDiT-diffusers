# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from ...models.transformers.transformer_pixeldit import PixelDiTTransformer2DModel

RECOMMENDED_GUIDANCE_BY_SIZE = {
    256: 3.25,
    512: 3.75,
}

RECOMMENDED_SCHEDULER_SHIFT_BY_SIZE = {
    256: 1.0,
    512: 3.0,
}


class PixelDiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation using PixelDiT (Pixel Diffusion Transformer).

    Parameters:
        transformer ([`PixelDiTTransformer2DModel`]):
            A class-conditioned `PixelDiTTransformer2DModel` that predicts flow-matching velocity in pixel space.
        scheduler ([`KarrasDiffusionSchedulers`] or [`FlowMatchEulerDiscreteScheduler`]):
            Diffusers scheduler interface for PixelDiT generation (defaults to deterministic flow-matching Euler).
        id2label (`dict[int, str]`, *optional*):
            ImageNet class id to English label mapping. Values may contain comma-separated synonyms.
    """

    @staticmethod
    def prepare_extra_step_kwargs(
        scheduler,
        generator=None,
        eta: float | None = None,
    ):
        kwargs = {}
        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        if "generator" in step_params:
            kwargs["generator"] = generator
        if eta is not None and "eta" in step_params:
            kwargs["eta"] = eta
        return kwargs

    model_cpu_offload_seq = "transformer"

    def __init__(
        self,
        transformer: PixelDiTTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        scheduler = scheduler or FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            stochastic_sampling=False,
        )
        self.register_modules(transformer=transformer, scheduler=scheduler)
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model_kwargs = dict(kwargs)
        transformer_subfolder = model_kwargs.pop("transformer_subfolder", None)
        scheduler_subfolder = model_kwargs.pop("scheduler_subfolder", None)
        scheduler_kwargs = model_kwargs.pop("scheduler_kwargs", {})
        base_path = Path(pretrained_model_name_or_path)

        if transformer_subfolder is None and (base_path / "transformer").exists():
            transformer_subfolder = "transformer"
        if scheduler_subfolder is None and (base_path / "scheduler").exists():
            scheduler_subfolder = "scheduler"

        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            if transformer_subfolder is not None:
                transformer_path = str(base_path / transformer_subfolder)
            else:
                transformer_path = pretrained_model_name_or_path

            transformer = PixelDiTTransformer2DModel.from_pretrained(transformer_path, **model_kwargs)
            try:
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=scheduler_subfolder,
                    **scheduler_kwargs,
                )
            except Exception:
                scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    shift=1.0,
                    stochastic_sampling=False,
                    **scheduler_kwargs,
                )

            id2label = cls._read_id2label_from_model_index(str(base_path))
            return cls(
                transformer=transformer,
                scheduler=scheduler,
                id2label=id2label,
            )

    def _ensure_labels_loaded(self) -> None:
        if self._labels_loaded_from_model_index:
            return
        loaded = self._read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

    @staticmethod
    def _normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
        if not id2label:
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _read_id2label_from_model_index(variant_path: Optional[str]) -> Dict[int, str]:
        if not variant_path:
            return {}
        variant_dir = Path(variant_path).resolve()
        model_index_path = variant_dir / "model_index.json"
        if not model_index_path.exists():
            return {}
        raw = json.loads(model_index_path.read_text(encoding="utf-8"))
        id2label = raw.get("id2label")
        if not isinstance(id2label, dict):
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
        label2id: Dict[str, int] = {}
        for class_id, value in id2label.items():
            for synonym in value.split(","):
                synonym = synonym.strip()
                if synonym:
                    label2id[synonym] = int(class_id)
        return dict(sorted(label2id.items()))

    @property
    def id2label(self) -> Dict[int, str]:
        self._ensure_labels_loaded()
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        self._ensure_labels_loaded()
        label2id = self.labels
        if not label2id:
            raise ValueError(
                "No English labels loaded. Ensure `id2label` exists in model_index.json."
            )

        if isinstance(label, str):
            label = [label]

        missing = [item for item in label if item not in label2id]
        if missing:
            preview = ", ".join(list(label2id.keys())[:8])
            raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
        return [label2id[item] for item in label]

    def _normalize_class_labels(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
    ) -> List[int]:
        if isinstance(class_labels, int):
            return [class_labels]

        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)

        if class_labels and isinstance(class_labels[0], str):
            return self.get_label_ids(class_labels)

        return list(class_labels)

    @staticmethod
    def _resolve_timeshift(scheduler, image_size: int) -> float:
        shift = getattr(scheduler.config, "shift", None)
        if shift is not None:
            return float(shift)
        return RECOMMENDED_SCHEDULER_SHIFT_BY_SIZE.get(image_size, 1.0)

    @staticmethod
    def _build_flow_timesteps(
        num_inference_steps: int,
        timeshift: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        last_step = 1.0 / num_inference_steps if num_inference_steps > 1 else 1.0
        timesteps = torch.linspace(0.0, 1.0 - last_step, num_inference_steps, device=device, dtype=dtype)
        timesteps = torch.cat([timesteps, torch.ones(1, device=device, dtype=dtype)], dim=0)
        if timeshift != 1.0:
            timesteps = timesteps / (timesteps + (1.0 - timesteps) * timeshift)
        return timesteps

    @staticmethod
    def _apply_classifier_free_guidance(model_output: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        model_output_uncond, model_output_cond = model_output.chunk(2, dim=0)
        return model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
        guidance_scale: Optional[float] = None,
        guidance_interval_min: float = 0.0,
        guidance_interval_max: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")
        if output_type not in {"pil", "np", "pt"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt'.")

        class_label_ids = self._normalize_class_labels(class_labels)
        do_classifier_free_guidance = guidance_scale is not None and guidance_scale > 1.0

        batch_size = len(class_label_ids)
        image_size = int(getattr(self.transformer.config, "sample_size", 256))
        patch_size = int(self.transformer.config.patch_size)
        height = int(height or image_size)
        width = int(width or image_size)
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive integers.")
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"height and width must be divisible by patch_size={patch_size}. Got {(height, width)}."
            )
        channels = int(self.transformer.config.in_channels)
        null_class_val = int(
            getattr(self.transformer.config, "num_classes", getattr(self.transformer.config, "num_class_embeds", 1000))
        )

        if guidance_scale is None:
            guidance_scale = RECOMMENDED_GUIDANCE_BY_SIZE.get(image_size, 3.25)

        latents = randn_tensor(
            shape=(batch_size, channels, height, width),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )

        class_labels_t = torch.tensor(class_label_ids, device=self._execution_device, dtype=torch.long).reshape(-1)
        class_labels_t = class_labels_t.clamp(0, null_class_val - 1)
        class_null = torch.full_like(class_labels_t, null_class_val)

        timeshift = self._resolve_timeshift(self.scheduler, image_size)
        flow_timesteps = self._build_flow_timesteps(
            num_inference_steps,
            timeshift,
            device=self._execution_device,
            dtype=torch.float32,
        )
        velocity_dtype = self.transformer.dtype
        v_prev = None

        for t_cur, t_next in self.progress_bar(list(zip(flow_timesteps[:-1], flow_timesteps[1:]))):
            dt = t_next - t_cur
            flow_time = float(t_cur)
            effective_guidance = (
                guidance_scale
                if do_classifier_free_guidance
                and guidance_interval_min < flow_time < guidance_interval_max
                else 1.0
            )

            latent_model_input = torch.cat([latents, latents], dim=0)
            labels = torch.cat([class_null, class_labels_t], dim=0)
            timesteps = torch.full(
                (latent_model_input.shape[0],),
                flow_time,
                device=self._execution_device,
                dtype=velocity_dtype,
            )
            model_output = self.transformer(
                latent_model_input,
                timestep=timesteps,
                class_labels=labels,
            ).sample
            velocity = self._apply_classifier_free_guidance(model_output, effective_guidance)

            if v_prev is None:
                latents = latents + velocity * dt
            else:
                latents = latents + dt * (1.5 * velocity - 0.5 * v_prev)
            v_prev = velocity

        images_pt = ((latents.float().clamp(-1, 1) + 1.0) / 2.0).cpu()
        if output_type == "pt":
            images = images_pt
        elif output_type == "np":
            images = images_pt.permute(0, 2, 3, 1).numpy()
        else:
            images = self.numpy_to_pil(images_pt.permute(0, 2, 3, 1).numpy())

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)


PixelDiTPipelineOutput = ImagePipelineOutput
