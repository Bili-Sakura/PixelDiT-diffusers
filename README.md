# PixelDiT Diffusers Refactor

This repository is fully organized around a Diffusers-style package layout, following the same migration pattern used in `JiT-diffusers` and `NiT-diffusers`.

Legacy standalone training/evaluation codepaths have been removed so the tree is focused on reusable Diffusers components and checkpoint conversion.

## Package layout

- `src/diffusers/models/transformers/transformer_pixeldit.py`: `PixelDiTTransformer2DModel` (`ModelMixin`/`ConfigMixin`) class-conditional transformer, presets, and legacy state-dict remapping.
- `scheduler/scheduler_config.json`: use built-in `FlowMatchEulerDiscreteScheduler` with deterministic flow matching.
- `src/diffusers/pipelines/pixeldit/pipeline_pixeldit.py`: `PixelDiTPipeline` with classifier-free guidance and dynamic `height`/`width` inference.
- `scripts/convert_pixeldit_to_diffusers.py`: converts legacy PixelDiT checkpoints to Diffusers model directories.
- `scripts/convert_diffusers_to_pixeldit.py`: converts Diffusers PixelDiT models back to legacy checkpoint format.
- `scripts/sample_pixeldit.py`: batch image sampling script for converted models.

## Convert a checkpoint

```bash
python scripts/convert_pixeldit_to_diffusers.py \
  --checkpoint checkpoints/pixeldit_xl_1600k.safetensors \
  --output pixeldit-xl-diffusers \
  --model-size pixeldit-xl \
  --check-load
```

The generated `conversion_metadata.json` includes Diffusers-style fields for round-trip conversion.

## Convert back to legacy checkpoint

```bash
python scripts/convert_diffusers_to_pixeldit.py \
  --model_path pixeldit-xl-diffusers \
  --output_path checkpoint-converted.safetensors
```

## Sample

```bash
python scripts/sample_pixeldit.py \
  --model pixeldit-xl-diffusers \
  --class-label 207 \
  --height 256 \
  --width 256 \
  --num-inference-steps 100 \
  --guidance-scale 3.25 \
  --guidance-interval-min 0.1 \
  --guidance-interval-max 1.0
```

You can run inference at any resolution divisible by the model patch size by passing `height` and `width` to pipeline `__call__`.

## Load from a Hub bundle

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "UserID/PixelDiT-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
image = pipe(class_labels=207, guidance_scale=3.25).images[0]
```

## Notes

- This repository is intended for Diffusers integration and checkpoint conversion workflows.
- For direct upstreaming, copy files under `src/diffusers` into matching paths in `huggingface/diffusers` and register lazy imports there.

## Citation

```bibtex
@inproceedings{yu2025pixeldit,
      title={PixelDiT: Pixel Diffusion Transformers for Image Generation},
      author={Yongsheng Yu and Wei Xiong and Weili Nie and Yichen Sheng and Shiqiu Liu and Jiebo Luo},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2026},
}
```
