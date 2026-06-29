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

import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...models.transformers.transformer_pixeldit_t2i import PixelDiTT2ITransformer2DModel


def _load_create_flow_dpm_solver():
    try:
        from diffusers.schedulers.flow_dpm import create_flow_dpm_solver

        return create_flow_dpm_solver
    except ImportError:
        repo_root = Path(__file__).resolve().parents[5]
        flow_dpm_path = repo_root / "libs/diffusers/src/diffusers/schedulers/flow_dpm.py"
        spec = importlib.util.spec_from_file_location("pixeldit_flow_dpm", flow_dpm_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load PixelDiT flow DPM solver from {flow_dpm_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.create_flow_dpm_solver


DEFAULT_TEXT_ENCODER_REPO = "google/gemma-2-2b-it"
DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality, over-saturated, blurry, deformed, watermark"
DEFAULT_CHI_PROMPT = "\n".join(
    [
        'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ]
)


class PixelDiTT2IPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using PixelDiT (Pixel Diffusion Transformer).

    Parameters:
        transformer ([`PixelDiTT2ITransformer2DModel`]):
            Text-conditioned PixelDiT transformer that predicts flow-matching velocity in pixel space.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Stores flow-shift settings used by the PixelDiT flow DPM sampler.
        text_encoder (`transformers.PreTrainedModel`, *optional*):
            Gemma decoder used to embed prompts.
        tokenizer (`transformers.PreTrainedTokenizer`, *optional*):
            Tokenizer paired with the text encoder.
    """

    model_cpu_offload_seq = "text_encoder->transformer"
    _optional_components = ["text_encoder", "tokenizer"]

    def __init__(
        self,
        transformer: PixelDiTT2ITransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder=None,
        tokenizer=None,
        model_max_length: int = 300,
        default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ):
        super().__init__()
        scheduler = scheduler or FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=4.0,
            stochastic_sampling=False,
        )
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.model_max_length = int(model_max_length)
        self.default_negative_prompt = default_negative_prompt

    def _load_text_stack(
        self,
        base_path: Path,
        text_encoder_subfolder: Optional[str],
        tokenizer_subfolder: Optional[str],
        text_encoder_repo: str,
        model_kwargs: dict,
    ) -> None:
        if self.text_encoder is not None and self.tokenizer is not None:
            return

        text_encoder = self.text_encoder
        tokenizer = self.tokenizer
        if text_encoder_subfolder is not None and text_encoder is None:
            text_encoder = AutoModelForCausalLM.from_pretrained(
                str(base_path / text_encoder_subfolder), **model_kwargs
            ).get_decoder()
        if tokenizer_subfolder is not None and tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(str(base_path / tokenizer_subfolder))
            tokenizer.padding_side = "right"

        if text_encoder is None or tokenizer is None:
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(text_encoder_repo)
            tokenizer.padding_side = "right"
            text_encoder = text_encoder or AutoModelForCausalLM.from_pretrained(
                text_encoder_repo, **model_kwargs
            ).get_decoder()

        self.register_modules(text_encoder=text_encoder, tokenizer=tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model_kwargs = dict(kwargs)
        transformer_subfolder = model_kwargs.pop("transformer_subfolder", None)
        scheduler_subfolder = model_kwargs.pop("scheduler_subfolder", None)
        text_encoder_subfolder = model_kwargs.pop("text_encoder_subfolder", None)
        tokenizer_subfolder = model_kwargs.pop("tokenizer_subfolder", None)
        scheduler_kwargs = model_kwargs.pop("scheduler_kwargs", {})
        text_encoder_repo = model_kwargs.pop("text_encoder_repo", DEFAULT_TEXT_ENCODER_REPO)
        base_path = Path(pretrained_model_name_or_path)

        if transformer_subfolder is None and (base_path / "transformer").exists():
            transformer_subfolder = "transformer"
        if scheduler_subfolder is None and (base_path / "scheduler").exists():
            scheduler_subfolder = "scheduler"
        if text_encoder_subfolder is None and (base_path / "text_encoder").exists():
            text_encoder_subfolder = "text_encoder"
        if tokenizer_subfolder is None and (base_path / "tokenizer").exists():
            tokenizer_subfolder = "tokenizer"

        try:
            pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            if transformer_subfolder is not None:
                transformer_path = str(base_path / transformer_subfolder)
            else:
                transformer_path = pretrained_model_name_or_path

            transformer = PixelDiTT2ITransformer2DModel.from_pretrained(transformer_path, **model_kwargs)
            try:
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=scheduler_subfolder,
                    **scheduler_kwargs,
                )
            except Exception:
                scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    shift=4.0,
                    stochastic_sampling=False,
                    **scheduler_kwargs,
                )
            pipe = cls(transformer=transformer, scheduler=scheduler)

        pipe._load_text_stack(
            base_path=base_path,
            text_encoder_subfolder=text_encoder_subfolder,
            tokenizer_subfolder=tokenizer_subfolder,
            text_encoder_repo=text_encoder_repo,
            model_kwargs=model_kwargs,
        )
        return pipe

    @property
    def _select_index(self) -> List[int]:
        return [0] + list(range(-self.model_max_length + 1, 0))

    def _ensure_text_stack_loaded(self) -> None:
        if self.text_encoder is not None and self.tokenizer is not None:
            return
        variant_path = getattr(self.config, "_name_or_path", None)
        base_path = Path(variant_path) if variant_path else Path(".")
        text_encoder_subfolder = "text_encoder" if (base_path / "text_encoder").exists() else None
        tokenizer_subfolder = "tokenizer" if (base_path / "tokenizer").exists() else None
        self._load_text_stack(
            base_path=base_path,
            text_encoder_subfolder=text_encoder_subfolder,
            tokenizer_subfolder=tokenizer_subfolder,
            text_encoder_repo=DEFAULT_TEXT_ENCODER_REPO,
            model_kwargs={"torch_dtype": self.transformer.dtype},
        )

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        do_classifier_free_guidance: bool = True,
        use_chi_prompt: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._ensure_text_stack_loaded()
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Text-to-image generation requires `text_encoder` and `tokenizer`.")

        device = device or self._execution_device
        dtype = dtype or self.transformer.dtype

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        if use_chi_prompt and DEFAULT_CHI_PROMPT:
            prompts_all = [DEFAULT_CHI_PROMPT + item for item in prompt]
            chi_prompt_tokens = len(self.tokenizer.encode(DEFAULT_CHI_PROMPT))
            max_length_all = chi_prompt_tokens + self.model_max_length - 2
        else:
            prompts_all = prompt
            max_length_all = self.model_max_length

        tokenized = self.tokenizer(
            prompts_all,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        prompt_embeds = self.text_encoder(input_ids, attention_mask=attention_mask)[0]
        select_index = self._select_index
        prompt_embeds = prompt_embeds[:, select_index]
        prompt_attention_mask = attention_mask[:, select_index]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        if not do_classifier_free_guidance:
            return prompt_embeds, prompt_attention_mask

        if negative_prompt is None:
            negative_prompt = self.default_negative_prompt
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size

        null_tokenized = self.tokenizer(
            negative_prompt,
            max_length=self.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        null_input_ids = null_tokenized.input_ids.to(device)
        null_attention_mask = null_tokenized.attention_mask.to(device)
        negative_embeds = self.text_encoder(null_input_ids, attention_mask=null_attention_mask)[0].to(dtype=dtype)
        negative_attention_mask = null_attention_mask

        prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_attention_mask, prompt_attention_mask], dim=0)
        return prompt_embeds, prompt_attention_mask

    @staticmethod
    def _resolve_flow_shift(scheduler, image_size: int) -> float:
        flow_shift = getattr(scheduler.config, "flow_shift", None)
        if flow_shift is not None:
            return float(flow_shift)
        shift = getattr(scheduler.config, "shift", None)
        if shift is not None:
            return float(shift)
        return 4.0 if image_size >= 1024 else 1.0

    def _make_dpm_transformer_fn(self):
        transformer = self.transformer

        def forward(x, t_input, y, **model_kwargs):
            if y.dim() == 4:
                y = y.squeeze(1)
            return transformer(
                x,
                timestep=t_input.to(dtype=transformer.dtype),
                encoder_hidden_states=y.to(dtype=transformer.dtype),
                encoder_attention_mask=None,
            ).sample

        return forward

    @torch.inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: Optional[float] = None,
        guidance_interval_min: float = 0.0,
        guidance_interval_max: float = 1.0,
        use_chi_prompt: bool = True,
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

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        image_size = int(getattr(self.transformer.config, "sample_size", 1024))
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

        if guidance_scale is None:
            guidance_scale = 2.75

        prompt_embeds, _ = self.encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            use_chi_prompt=use_chi_prompt,
        )

        latents = randn_tensor(
            shape=(batch_size, channels, height, width),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )

        flow_shift = self._resolve_flow_shift(self.scheduler, image_size)
        condition = prompt_embeds[batch_size:].unsqueeze(1)
        uncondition = prompt_embeds[:batch_size].unsqueeze(1)
        create_flow_dpm_solver = _load_create_flow_dpm_solver()
        dpm_solver = create_flow_dpm_solver(
            self._make_dpm_transformer_fn(),
            condition=condition,
            uncondition=uncondition,
            cfg_scale=float(guidance_scale),
            model_kwargs={},
            interval_guidance=[guidance_interval_min, guidance_interval_max],
        )
        os.environ["DPM_TQDM"] = "True"
        latents = dpm_solver.sample(
            latents,
            steps=num_inference_steps,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=flow_shift,
        )

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


PixelDiTT2IPipelineOutput = ImagePipelineOutput
