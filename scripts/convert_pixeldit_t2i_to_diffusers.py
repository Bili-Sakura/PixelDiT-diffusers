#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

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

from src.diffusers.models.transformers.transformer_pixeldit_t2i import (
    PIXELDIT_T2I_PRESET_CONFIGS,
    PixelDiTT2ITransformer2DModel,
    config_from_legacy_t2i,
    remap_t2i_legacy_state_dict,
)


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
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
    return remap_t2i_legacy_state_dict(state_dict)


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


def _write_model_index(output_dir: Path):
    model_index = {
        "_class_name": ["pipeline", "PixelDiTT2IPipeline"],
        "_diffusers_version": "0.35.1",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "transformer": ["transformer_pixeldit_t2i", "PixelDiTT2ITransformer2DModel"],
    }
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as file:
        json.dump(model_index, file, indent=2, sort_keys=True)
        file.write("\n")


def _write_conversion_metadata(
    output_dir: Path,
    model_type: str,
    checkpoint_path: str,
    config: Dict[str, Any],
    scheduler_shift: float,
):
    metadata = {
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "scheduler_shift": scheduler_shift,
        "pixeldit_t2i_args": {
            "model_type": model_type,
            "sample_size": config["sample_size"],
            "txt_embed_dim": config["txt_embed_dim"],
            "txt_max_length": config["txt_max_length"],
            "in_channels": config["in_channels"],
            "use_pixel_abs_pos": config["use_pixel_abs_pos"],
        },
    }
    with open(output_dir / "conversion_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")


def _copy_hub_modules(output_dir: Path):
    transformer_src = REPO_ROOT / "src/diffusers/models/transformers/transformer_pixeldit.py"
    transformer_t2i_src = REPO_ROOT / "src/diffusers/models/transformers/transformer_pixeldit_t2i.py"
    transformer_dst = output_dir / "transformer"
    transformer_dst.mkdir(parents=True, exist_ok=True)

    shutil.copy2(transformer_src, transformer_dst / "transformer_pixeldit.py")
    shutil.copy2(transformer_t2i_src, transformer_dst / "transformer_pixeldit_t2i.py")

    hub_t2i_path = transformer_dst / "transformer_pixeldit_t2i.py"
    hub_t2i_text = hub_t2i_path.read_text(encoding="utf-8")
    hub_t2i_text = hub_t2i_text.replace(
        "from .transformer_pixeldit import (",
        "import sys\n"
        "from pathlib import Path as _Path\n"
        "_MODULE_DIR = _Path(__file__).resolve().parent\n"
        "if str(_MODULE_DIR) not in sys.path:\n"
        "    sys.path.insert(0, str(_MODULE_DIR))\n"
        "from transformer_pixeldit import (",
    )
    hub_t2i_path.write_text(hub_t2i_text, encoding="utf-8")

    _write_hub_pipeline(output_dir)


def _write_hub_pipeline(output_dir: Path):
    pipeline_src = REPO_ROOT / "src/diffusers/pipelines/pixeldit/pipeline_pixeldit_t2i.py"
    hub_pipeline_text = pipeline_src.read_text(encoding="utf-8")
    hub_pipeline_text = hub_pipeline_text.replace(
        "from __future__ import annotations\n\nimport os\n",
        "from __future__ import annotations\n\nimport os\nimport sys\n",
    )
    hub_pipeline_text = hub_pipeline_text.replace(
        "from pathlib import Path\n",
        "from pathlib import Path\n\n"
        "_COLLECTION_ROOT = Path(__file__).resolve().parents[4]\n"
        "_DIFFUSERS_SRC = _COLLECTION_ROOT / \"libs/diffusers/src\"\n"
        "if _DIFFUSERS_SRC.is_dir() and str(_DIFFUSERS_SRC) not in sys.path:\n"
        "    sys.path.insert(0, str(_DIFFUSERS_SRC))\n",
        1,
    )
    hub_pipeline_text = hub_pipeline_text.replace(
        "try:\n    from diffusers.schedulers import FlowDPMSolverMultistepScheduler\n    from diffusers.schedulers.flow_dpm import create_flow_dpm_solver\nexcept ImportError:\n    from diffusers.schedulers.scheduling_flow_dpm_multistep import FlowDPMSolverMultistepScheduler\n    from diffusers.schedulers.flow_dpm import create_flow_dpm_solver\n\n",
        "from diffusers.schedulers import FlowDPMSolverMultistepScheduler\nfrom diffusers.schedulers.flow_dpm import create_flow_dpm_solver\n\n",
    )
    hub_pipeline_text = hub_pipeline_text.replace(
        "from ...models.transformers.transformer_pixeldit_t2i import PixelDiTT2ITransformer2DModel\n",
        "import importlib.util\n",
    )
    hub_pipeline_text = hub_pipeline_text.replace(
        "transformer: PixelDiTT2ITransformer2DModel,",
        "transformer,",
    )
    hub_pipeline_text = hub_pipeline_text.replace(
        "            transformer = PixelDiTT2ITransformer2DModel.from_pretrained(transformer_path, **model_kwargs)\n",
        "            transformer_module_path = Path(transformer_path) / 'transformer_pixeldit_t2i.py'\n"
        "            spec = importlib.util.spec_from_file_location('transformer_pixeldit_t2i', transformer_module_path)\n"
        "            module = importlib.util.module_from_spec(spec)\n"
        "            spec.loader.exec_module(module)\n"
        "            transformer = module.PixelDiTT2ITransformer2DModel.from_pretrained(transformer_path, **model_kwargs)\n",
    )
    (output_dir / "pipeline.py").write_text(hub_pipeline_text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PixelDiT T2I checkpoints to a Diffusers pipeline directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to a PixelDiT T2I .pth checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--config", default=None, help="Optional legacy config.json path.")
    parser.add_argument("--model-size", choices=sorted(PIXELDIT_T2I_PRESET_CONFIGS), default="pixeldit-t2i-1300m")
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--scheduler-shift", type=float, default=4.0)
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true", help="Instantiate the converted transformer and load weights.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"

    state_dict = _load_state_dict(args.checkpoint)
    if args.config:
        legacy_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        config = config_from_legacy_t2i(legacy_config)
    else:
        config = dict(PIXELDIT_T2I_PRESET_CONFIGS[args.model_size])
        config["model_type"] = args.model_size
        config["in_channels"] = 3
        config["use_pixel_abs_pos"] = True

    config["sample_size"] = args.sample_size

    if args.check_load:
        model = PixelDiTT2ITransformer2DModel(**config)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            raise SystemExit(1)

    _save_config(transformer_dir, {"_class_name": "PixelDiTT2ITransformer2DModel", **config})
    _save_weights(transformer_dir, state_dict, args.safe_serialization)

    scheduler_config = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "_diffusers_version": "0.36.0",
        "num_train_timesteps": 1000,
        "shift": args.scheduler_shift,
        "stochastic_sampling": False,
    }
    _save_config(scheduler_dir, scheduler_config)
    with open(scheduler_dir / "scheduler_config.json", "w", encoding="utf-8") as file:
        json.dump(scheduler_config, file, indent=2, sort_keys=True)
        file.write("\n")

    _write_model_index(output_dir)
    _write_conversion_metadata(output_dir, args.model_size, args.checkpoint, config, args.scheduler_shift)
    _copy_hub_modules(output_dir)
    print(f"Saved Diffusers-style PixelDiT T2I pipeline to {output_dir}")


if __name__ == "__main__":
    main()
