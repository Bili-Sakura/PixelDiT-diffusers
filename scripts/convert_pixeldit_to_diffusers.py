#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None

from src.diffusers.models.transformers.transformer_pixeldit import (
    PIXELDIT_PRESET_CONFIGS,
    PixelDiTTransformer2DModel,
    remap_legacy_state_dict,
)


def _load_state_dict(checkpoint_path: str, use_ema: bool = False) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to convert .safetensors checkpoints.")
        state_dict = safe_load_file(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "module", "denoiser"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break

    remapped = remap_legacy_state_dict(state_dict)
    has_ema = any(key.startswith("ema_") for key in remapped)
    if has_ema:
        source_prefix = "ema_denoiser." if use_ema else "denoiser."
        if any(key.startswith(source_prefix) for key in remapped):
            remapped = {
                key[len(source_prefix) :]: value
                for key, value in remapped.items()
                if key.startswith(source_prefix)
            }
        else:
            remapped = {key: value for key, value in remapped.items() if not key.startswith("ema_")}
    remapped = {
        key: value
        for key, value in remapped.items()
        if not key.startswith("diffusion_trainer.")
    }
    return remapped


def _save_config(output_dir: Path, config: Dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")


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
        "_class_name": ["pipeline", "PixelDiTPipeline"],
        "_diffusers_version": "0.35.1",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "transformer": ["transformer_pixeldit", "PixelDiTTransformer2DModel"],
    }
    if id2label:
        model_index["id2label"] = {str(key): value for key, value in id2label.items()}
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as file:
        json.dump(model_index, file, indent=2, sort_keys=True)
        file.write("\n")


def _write_conversion_metadata(output_dir: Path, model_type: str, checkpoint_path: str, config: Dict[str, Any], use_ema: bool):
    metadata = {
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "use_ema": use_ema,
        "pixeldit_args": {
            "model_type": model_type,
            "sample_size": config["sample_size"],
            "num_classes": config["num_classes"],
            "in_channels": config["in_channels"],
            "use_pixel_abs_pos": config["use_pixel_abs_pos"],
        },
    }
    with open(output_dir / "conversion_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")


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
    parser.add_argument("--model-size", choices=sorted(PIXELDIT_PRESET_CONFIGS), default="pixeldit-xl")
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--use-pixel-abs-pos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true", help="Instantiate the converted transformer and load weights.")
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ema_denoiser.* weights when present in Lightning checkpoints (recommended for inference).",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Override preset sample_size (e.g. 512 for ImageNet-512 checkpoints).")
    parser.add_argument("--scheduler-shift", type=float, default=None, help="Override FlowMatchEulerDiscreteScheduler shift (e.g. 2.0 for 512).")
    parser.add_argument("--id2label", default=None, help="Optional JSON file with ImageNet id2label mapping.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"

    state_dict = _load_state_dict(args.checkpoint, use_ema=args.use_ema)
    config = {
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        "use_pixel_abs_pos": args.use_pixel_abs_pos,
        "model_type": args.model_size,
        **PIXELDIT_PRESET_CONFIGS[args.model_size],
    }
    if args.sample_size is not None:
        config["sample_size"] = args.sample_size

    if args.check_load:
        model = PixelDiTTransformer2DModel(**config)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            raise SystemExit(1)

    _save_config(transformer_dir, {"_class_name": "PixelDiTTransformer2DModel", **config})
    _save_weights(transformer_dir, state_dict, args.safe_serialization)

    scheduler_shift = args.scheduler_shift
    if scheduler_shift is None:
        scheduler_shift = 3.0 if int(config["sample_size"]) >= 512 else 1.0
    _save_config(
        scheduler_dir,
        {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "_diffusers_version": "0.36.0",
            "num_train_timesteps": 1000,
            "shift": scheduler_shift,
            "stochastic_sampling": False,
        },
    )
    (scheduler_dir / "scheduler_config.json").write_text((scheduler_dir / "config.json").read_text(encoding="utf-8"), encoding="utf-8")

    id2label = _load_id2label(args.id2label)
    _write_model_index(output_dir, id2label)
    _write_conversion_metadata(output_dir, args.model_size, args.checkpoint, config, args.use_ema)
    print(f"Saved Diffusers-style PixelDiT pipeline to {output_dir}")


if __name__ == "__main__":
    main()
