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

import math
from collections.abc import Mapping
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm


PIXELDIT_PRESET_CONFIGS: Dict[str, Dict[str, object]] = {
    "pixeldit-xl": {
        "sample_size": 256,
        "num_groups": 16,
        "hidden_size": 1152,
        "pixel_hidden_size": 16,
        "patch_depth": 26,
        "pixel_depth": 4,
        "patch_size": 16,
    },
}


def remap_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map wrapper/backbone keys from legacy checkpoints to native PixelDiTTransformer2DModel keys."""
    remapped: Dict[str, torch.Tensor] = {}
    prefixes = ("transformer.", "model.", "module.", "denoiser.", "net.")
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        remapped[new_key] = value
    return remapped


def config_from_legacy(config: Dict[str, object]) -> Dict[str, object]:
    """Build native config kwargs from a legacy config.json dict."""
    model_type = config.get("model_type") or config.get("model_name") or config.get("model_size")
    if model_type not in PIXELDIT_PRESET_CONFIGS:
        raise ValueError(f"Unknown PixelDiT preset '{model_type}'. Known: {list(PIXELDIT_PRESET_CONFIGS)}")

    preset = dict(PIXELDIT_PRESET_CONFIGS[model_type])
    preset["num_classes"] = int(config.get("num_classes") or config.get("num_class_embeds") or 1000)
    preset["in_channels"] = int(config.get("in_channels", 3))
    preset["use_pixel_abs_pos"] = bool(config.get("use_pixel_abs_pos", True))
    preset["model_type"] = model_type

    for key in ("sample_size", "num_groups", "hidden_size", "pixel_hidden_size", "patch_depth", "pixel_depth", "patch_size"):
        if config.get(key) is not None:
            preset[key] = config[key]

    return preset


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("Embedding dimension must be even for 2D sin/cos positional embeddings.")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("Embedding dimension must be even for 1D sin/cos positional embeddings.")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def apply_adaln(hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return hidden_states * (1 + scale) + shift


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale: float = 16.0):
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    return freqs_cis.reshape(height * width, -1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timestep_freq = self.timestep_embedding(timesteps, self.frequency_embedding_size)
        mlp_dtype = next(self.mlp.parameters()).dtype
        if timestep_freq.dtype != mlp_dtype:
            timestep_freq = timestep_freq.to(mlp_dtype)
        return self.mlp(timestep_freq)


class ClassEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class RotaryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, hidden_states: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        batch_size, length, channels = hidden_states.shape
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, length, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 1, 3, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = apply_rotary_emb(query, key, freqs_cis=pos)
        query = query.view(batch_size, -1, self.num_heads, channels // self.num_heads).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, channels // self.num_heads).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, self.num_heads, channels // self.num_heads).transpose(1, 2).contiguous()
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, length, channels)
        hidden_states = self.proj(hidden_states)
        return self.proj_drop(hidden_states)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return self.drop(hidden_states)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        return self.linear(hidden_states)


class PatchTokenEmbedder(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, norm_layer=None, bias: bool = True):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return self.norm(hidden_states)


class PixelTokenEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_output: int, use_pixel_abs_pos: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size_output = int(hidden_size_output)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.proj = nn.Linear(self.in_channels, self.hidden_size_output, bias=True)
        self._pos_cache: Dict[tuple[str, int, int], torch.Tensor] = {}

    def _fetch_pixel_pos_image(self, height: int, width: int, device: torch.device, dtype: torch.dtype):
        key = ("image", height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device=device, dtype=dtype)
        if height == width:
            pos_np = get_2d_sincos_pos_embed(self.hidden_size_output, height)
        else:
            grid_h = np.arange(height, dtype=np.float32)
            grid_w = np.arange(width, dtype=np.float32)
            grid = np.meshgrid(grid_w, grid_h)
            grid = np.stack(grid, axis=0).reshape(2, 1, height, width)
            pos_np = get_2d_sincos_pos_embed_from_grid(self.hidden_size_output, grid)
        pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)
        self._pos_cache[key] = pos
        return pos

    def forward(self, inputs: torch.Tensor, img_height: int, img_width: int, patch_size: int):
        if inputs.dim() != 4:
            raise ValueError("PixelTokenEmbedder expects inputs of shape [B,C,H,W]")
        batch_size, channels, height, width = inputs.shape
        if height != img_height or width != img_width:
            raise ValueError("Input resolution does not match img_height/img_width.")
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("Image height and width must be divisible by patch_size.")
        h_tokens, w_tokens = height // patch_size, width // patch_size
        patch_area = patch_size * patch_size
        hidden_states = inputs.permute(0, 2, 3, 1).contiguous()
        hidden_states = self.proj(hidden_states)
        if self.use_pixel_abs_pos:
            pos_full = self._fetch_pixel_pos_image(height, width, inputs.device, inputs.dtype)
            hidden_states = hidden_states + pos_full.view(height, width, self.hidden_size_output).unsqueeze(0)
        hidden_states = hidden_states.view(batch_size, h_tokens, patch_size, w_tokens, patch_size, self.hidden_size_output)
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous()
        return hidden_states.view(batch_size * h_tokens * w_tokens, patch_area, self.hidden_size_output)


class AugmentedDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, groups: int, mlp_ratio: float = 4.0, adaLN_modulation=None, eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        self.attn = RotaryAttention(hidden_size, num_heads=groups, qkv_bias=False, eps=eps)
        self.norm2 = RMSNorm(hidden_size, eps=eps)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = adaLN_modulation if adaLN_modulation is not None else nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor, pos: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(
            6, dim=-1
        )
        hidden_states = hidden_states + gate_msa * self.attn(
            apply_adaln(self.norm1(hidden_states), shift_msa, scale_msa), pos
        )
        hidden_states = hidden_states + gate_mlp * self.mlp(
            apply_adaln(self.norm2(hidden_states), shift_mlp, scale_mlp)
        )
        return hidden_states


class PiTBlock(nn.Module):
    def __init__(
        self,
        pixel_hidden_size: int,
        patch_hidden_size: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_hidden_size: Optional[int] = None,
        attn_num_heads: Optional[int] = None,
        rope_fn=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.context_dim = int(patch_hidden_size)
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else self.context_dim
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        if self.attn_dim % self.num_heads != 0:
            raise ValueError("pixel attention hidden size must be divisible by pixel num_heads")
        patch_area = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(patch_area * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, patch_area * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=eps)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False, eps=eps)
        self.norm2 = RMSNorm(self.pixel_dim, eps=eps)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)
        self.adaLN_modulation = nn.Sequential(nn.Linear(self.context_dim, 6 * self.pixel_dim * patch_area, bias=True))
        self._pos_cache: Dict[tuple[int, int], torch.Tensor] = {}
        self._rope_fn = rope_fn if rope_fn is not None else precompute_freqs_cis_2d

    def _fetch_pos(self, height: int, width: int, device: torch.device):
        key = (height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        pos = self._rope_fn(self.attn_dim // self.num_heads, height, width).to(device)
        self._pos_cache[key] = pos
        return pos

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        image_height: int,
        image_width: int,
        patch_size: int,
    ) -> torch.Tensor:
        batch_tokens, patch_area, channels = hidden_states.shape
        if channels != self.pixel_dim:
            raise ValueError(f"PiTBlock expected pixel_dim={self.pixel_dim}, got {channels}")
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError("Image height and width must be divisible by patch_size.")
        h_tokens, w_tokens = image_height // patch_size, image_width // patch_size
        length = h_tokens * w_tokens
        batch_size = batch_tokens // length
        cond_params = self.adaLN_modulation(conditioning).view(batch_tokens, patch_area, 6 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)
        hidden_norm = apply_adaln(self.norm1(hidden_states), shift_msa, scale_msa)
        hidden_flat = hidden_norm.view(batch_tokens, patch_area * self.pixel_dim)
        hidden_comp = self.compress_to_attn(hidden_flat).view(batch_size, length, self.attn_dim)
        pos_comp = self._fetch_pos(h_tokens, w_tokens, hidden_states.device)
        attn_out = self.attn(hidden_comp, pos_comp)
        attn_flat = self.expand_from_attn(attn_out.view(batch_size * length, self.attn_dim))
        attn_exp = attn_flat.view(batch_tokens, patch_area, self.pixel_dim)
        hidden_states = hidden_states + gate_msa * attn_exp
        mlp_out = self.mlp(apply_adaln(self.norm2(hidden_states), shift_mlp, scale_mlp))
        hidden_states = hidden_states + gate_mlp * mlp_out
        return hidden_states


class PixelDiTTransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["pos", "_pos_cache"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 256,
        in_channels: int = 3,
        num_groups: int = 16,
        hidden_size: int = 1152,
        pixel_hidden_size: int = 16,
        patch_depth: int = 26,
        pixel_depth: int = 4,
        patch_size: int = 16,
        num_classes: int = 1000,
        use_pixel_abs_pos: bool = True,
        norm_eps: float = 1e-6,
        model_type: str | None = None,
        num_class_embeds: int | None = None,
    ):
        super().__init__()
        if num_class_embeds is not None:
            num_classes = int(num_class_embeds)
        if model_type in PIXELDIT_PRESET_CONFIGS:
            preset = PIXELDIT_PRESET_CONFIGS[model_type]
            sample_size = int(preset["sample_size"])
            num_groups = int(preset["num_groups"])
            hidden_size = int(preset["hidden_size"])
            pixel_hidden_size = int(preset["pixel_hidden_size"])
            patch_depth = int(preset["patch_depth"])
            pixel_depth = int(preset["pixel_depth"])
            patch_size = int(preset["patch_size"])

        self.sample_size = int(sample_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.num_classes = int(num_classes)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.norm_eps = float(norm_eps)
        self.gradient_checkpointing = False

        if self.pixel_depth <= 0:
            raise ValueError("PixelDiT expects pixel_depth > 0 to preserve the dual-level pipeline")

        self.pixel_embedder = PixelTokenEmbedder(
            self.in_channels, self.pixel_hidden_size, use_pixel_abs_pos=self.use_pixel_abs_pos
        )
        self.s_embedder = PatchTokenEmbedder(self.in_channels * self.patch_size**2, self.hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.y_embedder = ClassEmbedder(self.num_classes + 1, self.hidden_size)

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels, eps=self.norm_eps)
        self.patch_blocks = nn.ModuleList(
            [AugmentedDiTBlock(self.hidden_size, self.num_groups, eps=self.norm_eps) for _ in range(self.patch_depth)]
        )
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    patch_size=self.patch_size,
                    num_heads=self.num_groups,
                    mlp_ratio=4.0,
                    eps=self.norm_eps,
                )
                for _ in range(self.pixel_depth)
            ]
        )
        self._precompute_pos: Dict[tuple[int, int], torch.Tensor] = {}
        self._initialize_weights()

    def _fetch_pos(self, height: int, width: int, device: torch.device):
        key = (height, width)
        if key in self._precompute_pos:
            return self._precompute_pos[key].to(device)
        pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
        self._precompute_pos[key] = pos
        return pos

    def _initialize_weights(self) -> None:
        weight = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)
        for block in self.pixel_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float],
        class_labels: Union[torch.Tensor, int],
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        if sample.dim() != 4:
            raise ValueError("PixelDiTTransformer2DModel expects sample of shape [B,C,H,W]")
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

        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=sample.device, dtype=torch.long)
        class_labels = class_labels.to(device=sample.device, dtype=torch.long).reshape(-1)
        if class_labels.shape[0] == 1 and batch_size > 1:
            class_labels = class_labels.repeat(batch_size)

        pos = self._fetch_pos(height // self.patch_size, width // self.patch_size, sample.device)
        x_patches = F.unfold(sample, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t_emb = self.t_embedder(timestep.view(-1)).view(batch_size, -1, self.hidden_size)
        y_emb = self.y_embedder(class_labels).view(batch_size, 1, self.hidden_size)
        conditioning = F.silu(t_emb + y_emb)

        patch_states = self.s_embedder(x_patches)
        for block in self.patch_blocks:
            if self.training and self.gradient_checkpointing:

                def custom_forward(hidden_states, cond, position):
                    return block(hidden_states, cond, position)

                patch_states = torch.utils.checkpoint.checkpoint(
                    custom_forward, patch_states, conditioning, pos, use_reentrant=False
                )
            else:
                patch_states = block(patch_states, conditioning, pos)
        patch_states = F.silu(t_emb + patch_states)

        length = patch_states.shape[1]
        conditioning_states = patch_states.view(batch_size * length, self.hidden_size)
        pixel_states = self.pixel_embedder(
            sample, img_height=height, img_width=width, patch_size=self.patch_size
        )
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
        pixel_states = pixel_states.view(batch_size, length, patch_area, self.out_channels).permute(0, 3, 2, 1)
        pixel_states = pixel_states.contiguous().view(batch_size, self.out_channels * patch_area, length)
        output = F.fold(pixel_states, (height, width), kernel_size=self.patch_size, stride=self.patch_size)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pixeldit_checkpoint(
        cls,
        checkpoint_path: str,
        model_type: Literal["pixeldit-xl"] = "pixeldit-xl",
        map_location: str = "cpu",
        strict: bool = True,
    ) -> Tuple["PixelDiTTransformer2DModel", Dict[str, object]]:
        if model_type not in PIXELDIT_PRESET_CONFIGS:
            raise ValueError(f"Unknown PixelDiT preset '{model_type}'.")

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

        config = dict(PIXELDIT_PRESET_CONFIGS[model_type])
        config["model_type"] = model_type
        model = cls(**config)
        model.load_state_dict(remap_legacy_state_dict(state_dict), strict=strict)

        metadata = {
            "checkpoint_path": checkpoint_path,
            "model_type": model_type,
        }
        return model, metadata

    def to_pixeldit_checkpoint(self, prefix: str = "") -> Dict[str, torch.Tensor]:
        checkpoint: Dict[str, torch.Tensor] = {}
        for key, value in self.state_dict().items():
            checkpoint[f"{prefix}{key}"] = value.detach().cpu()
        return checkpoint


PixelDiTDiffusersModel = PixelDiTTransformer2DModel
