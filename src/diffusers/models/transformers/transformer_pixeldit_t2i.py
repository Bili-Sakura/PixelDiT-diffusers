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

from collections.abc import Mapping
from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm

from .transformer_pixeldit import (
    FinalLayer,
    FeedForward,
    PatchTokenEmbedder,
    PiTBlock,
    PixelTokenEmbedder,
    TimestepConditioner,
    apply_adaln,
    apply_rotary_emb,
    precompute_freqs_cis_2d,
)


PIXELDIT_T2I_PRESET_CONFIGS: Dict[str, Dict[str, object]] = {
    "pixeldit-t2i-1300m": {
        "sample_size": 1024,
        "num_groups": 24,
        "hidden_size": 1536,
        "pixel_hidden_size": 16,
        "pixel_attn_hidden_size": 1152,
        "pixel_num_groups": 16,
        "patch_depth": 14,
        "pixel_depth": 2,
        "num_text_blocks": 4,
        "patch_size": 16,
        "txt_embed_dim": 2304,
        "txt_max_length": 300,
        "use_text_rope": True,
        "text_rope_theta": 10000.0,
        "repa_encoder_index": 6,
    },
}


def remap_t2i_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map legacy T2I checkpoint keys to native PixelDiTT2ITransformer2DModel keys."""
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("_repa_projector."):
            continue
        new_key = key[5:] if key.startswith("core.") else key
        remapped[new_key] = value
    return remapped


def config_from_legacy_t2i(config: Dict[str, object]) -> Dict[str, object]:
    """Build native T2I config kwargs from a legacy config.json dict."""
    model_type = config.get("model_type")
    if model_type == "pixeldit" and config.get("architectures") == ["PixDiT_T2I"]:
        model_type = "pixeldit-t2i-1300m"
    if model_type not in PIXELDIT_T2I_PRESET_CONFIGS:
        raise ValueError(
            f"Unknown PixelDiT T2I preset '{model_type}'. Known: {list(PIXELDIT_T2I_PRESET_CONFIGS)}"
        )

    preset = dict(PIXELDIT_T2I_PRESET_CONFIGS[model_type])
    preset["in_channels"] = int(config.get("in_channels", 3))
    preset["use_pixel_abs_pos"] = bool(config.get("use_pixel_abs_pos", True))
    preset["model_type"] = model_type

    for key in preset:
        if config.get(key) is not None:
            preset[key] = config[key]
    if config.get("image_size") is not None:
        preset["sample_size"] = int(config["image_size"])
    return preset


class MMDiTJointAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim should be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_x = RMSNorm(self.head_dim, eps=eps)
        self.k_norm_x = RMSNorm(self.head_dim, eps=eps)
        self.q_norm_y = RMSNorm(self.head_dim, eps=eps)
        self.k_norm_y = RMSNorm(self.head_dim, eps=eps)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.proj_drop_y = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pos_img: torch.Tensor,
        pos_txt: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_img_tokens, channels = x.shape
        _, num_txt_tokens, txt_channels = y.shape
        if batch_size != y.shape[0] or channels != txt_channels:
            raise ValueError("x and y must share batch and channel dims")

        qkv_x = (
            self.qkv_x(x)
            .reshape(batch_size, num_img_tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 1, 3, 4)
        )
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        qkv_y = (
            self.qkv_y(y)
            .reshape(batch_size, num_txt_tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 1, 3, 4)
        )
        qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)

        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        qx = qx.transpose(1, 2)
        kx = kx.transpose(1, 2)
        vx = vx.transpose(1, 2)
        qy = qy.transpose(1, 2)
        ky = ky.transpose(1, 2)
        vy = vy.transpose(1, 2)

        q_joint = torch.cat([qy, qx], dim=2)
        k_joint = torch.cat([ky, kx], dim=2)
        v_joint = torch.cat([vy, vx], dim=2)

        out_joint = F.scaled_dot_product_attention(
            q_joint, k_joint, v_joint, dropout_p=0.0, attn_mask=attn_mask
        )
        out_y = out_joint[:, :, :num_txt_tokens, :]
        out_x = out_joint[:, :, num_txt_tokens:, :]

        out_y = out_y.transpose(1, 2).reshape(batch_size, num_txt_tokens, channels)
        out_x = out_x.transpose(1, 2).reshape(batch_size, num_img_tokens, channels)
        out_x = self.proj_drop_x(self.proj_x(out_x))
        out_y = self.proj_drop_y(self.proj_y(out_y))
        return out_x, out_y


class MMDiTBlockT2I(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        groups: int,
        mlp_ratio: float = 4.0,
        adaLN_modulation_img=None,
        adaLN_modulation_txt=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm_x1 = RMSNorm(hidden_size, eps=eps)
        self.norm_y1 = RMSNorm(hidden_size, eps=eps)
        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False, eps=eps)
        self.norm_x2 = RMSNorm(hidden_size, eps=eps)
        self.norm_y2 = RMSNorm(hidden_size, eps=eps)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation_img = adaLN_modulation_img or nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_modulation_txt = adaLN_modulation_txt or nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        conditioning: torch.Tensor,
        pos_img: torch.Tensor,
        pos_txt: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(
            conditioning
        ).chunk(6, dim=-1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(
            conditioning
        ).chunk(6, dim=-1)

        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y
        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y


class PixelDiTT2ITransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["pos", "_pos_cache", "y_pos_embedding"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 1024,
        in_channels: int = 3,
        num_groups: int = 24,
        hidden_size: int = 1536,
        pixel_hidden_size: int = 16,
        pixel_attn_hidden_size: int = 1152,
        pixel_num_groups: int = 16,
        patch_depth: int = 14,
        pixel_depth: int = 2,
        num_text_blocks: int = 4,
        patch_size: int = 16,
        txt_embed_dim: int = 2304,
        txt_max_length: int = 300,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        repa_encoder_index: int = 6,
        use_pixel_abs_pos: bool = True,
        norm_eps: float = 1e-6,
        model_type: str | None = None,
    ):
        super().__init__()
        if model_type in PIXELDIT_T2I_PRESET_CONFIGS:
            preset = PIXELDIT_T2I_PRESET_CONFIGS[model_type]
            sample_size = int(preset["sample_size"])
            num_groups = int(preset["num_groups"])
            hidden_size = int(preset["hidden_size"])
            pixel_hidden_size = int(preset["pixel_hidden_size"])
            pixel_attn_hidden_size = int(preset["pixel_attn_hidden_size"])
            pixel_num_groups = int(preset["pixel_num_groups"])
            patch_depth = int(preset["patch_depth"])
            pixel_depth = int(preset["pixel_depth"])
            num_text_blocks = int(preset["num_text_blocks"])
            patch_size = int(preset["patch_size"])
            txt_embed_dim = int(preset["txt_embed_dim"])
            txt_max_length = int(preset["txt_max_length"])
            use_text_rope = bool(preset["use_text_rope"])
            text_rope_theta = float(preset["text_rope_theta"])
            repa_encoder_index = int(preset["repa_encoder_index"])

        self.sample_size = int(sample_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.num_text_blocks = int(num_text_blocks)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.pixel_attn_hidden_size = int(pixel_attn_hidden_size)
        self.pixel_num_groups = int(pixel_num_groups)
        self.txt_embed_dim = int(txt_embed_dim)
        self.txt_max_length = int(txt_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)
        self.repa_encoder_index = int(repa_encoder_index)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.norm_eps = float(norm_eps)
        self.gradient_checkpointing = False

        if self.pixel_depth <= 0:
            raise ValueError("PixelDiT T2I expects pixel_depth > 0 to preserve the pixel pathway")

        self.pixel_embedder = PixelTokenEmbedder(
            self.in_channels, self.pixel_hidden_size, use_pixel_abs_pos=self.use_pixel_abs_pos
        )
        self.s_embedder = PatchTokenEmbedder(self.in_channels * self.patch_size**2, self.hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.y_embedder = PatchTokenEmbedder(
            self.txt_embed_dim,
            self.hidden_size,
            bias=True,
            norm_layer=lambda dim: RMSNorm(dim, eps=self.norm_eps),
        )
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, self.hidden_size))

        self.patch_blocks = nn.ModuleList(
            [MMDiTBlockT2I(self.hidden_size, self.num_groups, eps=self.norm_eps) for _ in range(self.patch_depth)]
        )
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    patch_size=self.patch_size,
                    num_heads=self.num_groups,
                    mlp_ratio=4.0,
                    attn_hidden_size=self.pixel_attn_hidden_size,
                    attn_num_heads=self.pixel_num_groups,
                    rope_fn=precompute_freqs_cis_2d,
                    eps=self.norm_eps,
                )
                for _ in range(self.pixel_depth)
            ]
        )
        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels, eps=self.norm_eps)
        self._precompute_pos: Dict[tuple[int, int], torch.Tensor] = {}
        self._precompute_pos_txt: Dict[int, torch.Tensor] = {}
        self._initialize_weights()

    def _fetch_pos(self, height: int, width: int, device: torch.device):
        key = (height, width)
        if key in self._precompute_pos:
            return self._precompute_pos[key].to(device)
        pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
        self._precompute_pos[key] = pos
        return pos

    def _fetch_pos_text(self, length: int, device: torch.device):
        if length in self._precompute_pos_txt:
            return self._precompute_pos_txt[length].to(device)
        head_dim = self.hidden_size // self.num_groups
        freqs = 1.0 / (
            self.text_rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        positions = torch.arange(0, length, device=device).float().unsqueeze(1)
        angles = positions * freqs.unsqueeze(0)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self._precompute_pos_txt[length] = freqs_cis
        return freqs_cis

    def _initialize_weights(self) -> None:
        weight = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _build_joint_attn_mask(
        self,
        encoder_attention_mask: Optional[torch.Tensor],
        batch_size: int,
        num_img_tokens: int,
        num_txt_tokens: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if encoder_attention_mask is None:
            return None
        mask = encoder_attention_mask
        while mask.dim() > 2 and mask.size(1) == 1:
            mask = mask.squeeze(1)
        if mask.dim() == 3 and mask.size(1) == 1:
            mask = mask.squeeze(1)
        if mask.dim() != 2:
            return None
        pad = mask == 0
        pad_img = torch.zeros((batch_size, num_img_tokens), dtype=torch.bool, device=device)
        return torch.cat([pad[:, :num_txt_tokens], pad_img], dim=1).view(batch_size, 1, 1, num_txt_tokens + num_img_tokens)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float],
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        if sample.dim() != 4:
            raise ValueError("PixelDiTT2ITransformer2DModel expects sample of shape [B,C,H,W]")
        batch_size, _, height, width = sample.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Image height and width must be divisible by patch_size.")

        timestep = torch.as_tensor(timestep, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.repeat(batch_size)
        else:
            timestep = timestep.reshape(-1)
            if timestep.shape[0] == 1 and batch_size > 1:
                timestep = timestep.repeat(batch_size)

        if encoder_hidden_states.dim() == 4:
            encoder_hidden_states = encoder_hidden_states.squeeze(1)
        if encoder_hidden_states.dim() != 3:
            raise ValueError("encoder_hidden_states must be [B, L, D]")

        height_tokens = height // self.patch_size
        width_tokens = width // self.patch_size
        num_img_tokens = height_tokens * width_tokens

        pos = self._fetch_pos(height_tokens, width_tokens, sample.device)
        x_patches = F.unfold(sample, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t_emb = self.t_embedder(timestep.view(-1)).view(batch_size, -1, self.hidden_size)

        txt_length = min(encoder_hidden_states.shape[1], self.txt_max_length)
        text_states = encoder_hidden_states[:, :txt_length, :]
        text_states = self.y_embedder(text_states).view(batch_size, txt_length, self.hidden_size)
        text_states = text_states + self.y_pos_embedding[:, :txt_length, :].to(text_states.dtype)
        conditioning = F.silu(t_emb)

        pos_txt = self._fetch_pos_text(txt_length, sample.device) if self.use_text_rope else None
        attn_mask = self._build_joint_attn_mask(
            encoder_attention_mask, batch_size, num_img_tokens, txt_length, sample.device
        )

        patch_states = self.s_embedder(x_patches)
        for block in self.patch_blocks:
            if self.training and self.gradient_checkpointing:

                def custom_forward(img_states, txt_states, cond, position_img, position_txt, mask):
                    return block(img_states, txt_states, cond, position_img, position_txt, mask)

                patch_states, text_states = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    patch_states,
                    text_states,
                    conditioning,
                    pos,
                    pos_txt,
                    attn_mask,
                    use_reentrant=False,
                )
            else:
                patch_states, text_states = block(
                    patch_states, text_states, conditioning, pos, pos_txt, attn_mask
                )

        patch_states = F.silu(t_emb + patch_states)
        if patch_states.shape[1] != num_img_tokens:
            if patch_states.shape[1] > num_img_tokens:
                patch_states = patch_states[:, :num_img_tokens, :]
            else:
                pad_len = num_img_tokens - patch_states.shape[1]
                patch_states = torch.cat(
                    [patch_states, patch_states.new_zeros(batch_size, pad_len, patch_states.shape[2])], dim=1
                )

        conditioning_states = patch_states.reshape(batch_size * num_img_tokens, self.hidden_size)
        pixel_states = self.pixel_embedder(sample, img_height=height, img_width=width, patch_size=self.patch_size)
        for block in self.pixel_blocks:
            if self.training and self.gradient_checkpointing:

                def custom_forward(hidden_states, cond):
                    return block(hidden_states, cond, height, width, self.patch_size)

                pixel_states = torch.utils.checkpoint.checkpoint(
                    custom_forward, pixel_states, conditioning_states, use_reentrant=False
                )
            else:
                pixel_states = block(pixel_states, conditioning_states, height, width, self.patch_size)

        pixel_states = self.final_layer(pixel_states)
        patch_area = self.patch_size * self.patch_size
        pixel_states = pixel_states.view(batch_size, num_img_tokens, patch_area, self.out_channels).permute(0, 3, 2, 1)
        pixel_states = pixel_states.contiguous().view(batch_size, self.out_channels * patch_area, num_img_tokens)
        output = F.fold(pixel_states, (height, width), kernel_size=self.patch_size, stride=self.patch_size)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pixeldit_t2i_checkpoint(
        cls,
        checkpoint_path: str,
        model_type: Literal["pixeldit-t2i-1300m"] = "pixeldit-t2i-1300m",
        map_location: str = "cpu",
        strict: bool = True,
    ) -> Tuple["PixelDiTT2ITransformer2DModel", Dict[str, object]]:
        if model_type not in PIXELDIT_T2I_PRESET_CONFIGS:
            raise ValueError(f"Unknown PixelDiT T2I preset '{model_type}'.")

        if checkpoint_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as error:
                raise ImportError("Install safetensors to load .safetensors checkpoints.") from error
            state_dict = load_file(checkpoint_path, device=map_location)
        else:
            loaded = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            if isinstance(loaded, Mapping):
                state_dict = loaded
                for key in ("state_dict", "model", "module", "denoiser"):
                    if key in state_dict and isinstance(state_dict[key], dict):
                        state_dict = state_dict[key]
                        break
            else:
                raise ValueError("Unsupported checkpoint format.")

        config = dict(PIXELDIT_T2I_PRESET_CONFIGS[model_type])
        config["model_type"] = model_type
        model = cls(**config)
        model.load_state_dict(remap_t2i_legacy_state_dict(state_dict), strict=strict)
        metadata = {"checkpoint_path": checkpoint_path, "model_type": model_type}
        return model, metadata


PixelDiTT2IDiffusersModel = PixelDiTT2ITransformer2DModel
