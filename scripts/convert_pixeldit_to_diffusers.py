#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None

from diffusers.models.transformers import PixelDiTTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "pixeldit-xl": {
        "num_groups": 16,
        "hidden_size": 1152,
        "pixel_hidden_size": 16,
        "patch_depth": 26,
        "pixel_depth": 4,
        "patch_size": 16,
    },
}


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to convert .safetensors checkpoints.")
        state_dict = safe_load_file(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "module", "denoiser"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
    return _clean_state_dict(state_dict)


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("model.", "module.", "denoiser.", "net.")
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _save_config(output_dir: Path, config: Dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def _save_weights(output_dir: Path, state_dict: Dict[str, torch.Tensor], safe_serialization: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(state_dict, str(output_dir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def _write_model_index(output_dir: Path, id2label: Optional[Dict[int, str]]):
    model_index = {
        "_class_name": "PixelDiTPipeline",
        "_diffusers_version": "0.30.1",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "transformer": ["diffusers", "PixelDiTTransformer2DModel"],
    }
    if id2label:
        model_index["id2label"] = {str(key): value for key, value in id2label.items()}
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, sort_keys=True)
        f.write("\n")


def _load_id2label(path: Optional[str]) -> Optional[Dict[int, str]]:
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {int(item[0]): str(item[1]) for item in raw}
    raise ValueError("id2label must be a dict or list of pairs")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PixelDiT checkpoints to a Diffusers pipeline directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to a PixelDiT .safetensors/.bin/.pt checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="pixeldit-xl")
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--use-pixel-abs-pos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true", help="Instantiate the converted transformer and load weights.")
    parser.add_argument("--id2label", default=None, help="Optional JSON file with ImageNet id2label mapping.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"

    state_dict = _load_state_dict(args.checkpoint)
    config = {
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        "use_pixel_abs_pos": args.use_pixel_abs_pos,
        **MODEL_PRESETS[args.model_size],
    }

    if args.check_load:
        model = PixelDiTTransformer2DModel(**config)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            raise SystemExit(1)

    _save_config(transformer_dir, {"_class_name": "PixelDiTTransformer2DModel", **config})
    _save_weights(transformer_dir, state_dict, args.safe_serialization)

    _save_config(
        scheduler_dir,
        {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "stochastic_sampling": False,
        },
    )

    id2label = _load_id2label(args.id2label)
    _write_model_index(output_dir, id2label)
    print(f"Saved Diffusers-style PixelDiT pipeline to {output_dir}")


if __name__ == "__main__":
    main()
