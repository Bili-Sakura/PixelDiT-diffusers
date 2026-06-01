<p align="center">
  <img src="assets/pixeldit-logo.png" height="120" />
</p>

<h2 align="center">PixelDiT Diffusers Integration</h2>

This repository contains a Diffusers-style implementation of PixelDiT. The legacy
standalone training, preprocessing, and YAML config codepaths have been removed
so the remaining tree mirrors the package boundaries used by `huggingface/diffusers`.

## Package layout

- `src/diffusers/models/transformers/transformer_pixeldit.py`:
  `PixelDiTTransformer2DModel`, a `ModelMixin`/`ConfigMixin` class-conditional
  PixelDiT transformer.
- `src/diffusers/pipelines/pixeldit/pipeline_pixeldit.py`:
  `PixelDiTPipeline`, a Diffusers pipeline with classifier-free guidance for
  native-resolution pixel-space sampling.
- `scripts/convert_pixeldit_to_diffusers.py`:
  converts original PixelDiT checkpoints to a Diffusers pipeline directory.
- `scripts/sample_pixeldit.py`:
  samples from a converted pipeline.

## Convert a checkpoint

```bash
python scripts/convert_pixeldit_to_diffusers.py \
  --checkpoint checkpoints/pixeldit_xl_1600k.safetensors \
  --output pixeldit-xl-diffusers \
  --model-size pixeldit-xl
```

The converted directory contains `model_index.json`, transformer weights and
config, and scheduler config. Use `--id2label` to embed ImageNet class names in
`model_index.json` if desired.

## Sample

```bash
python scripts/sample_pixeldit.py \
  --model pixeldit-xl-diffusers \
  --class-label 207 \
  --height 256 \
  --width 256 \
  --num-inference-steps 100 \
  --guidance-scale 3.25 \
  --guidance-low 0.1 \
  --guidance-high 1.0
```

## Upstreaming to Diffusers

Copy the files under `src/diffusers` into the matching locations in the
`huggingface/diffusers` repository and add the classes to Diffusers' lazy import
registries. The module names and save/load artifacts are already aligned with
Diffusers package conventions.

## Citation

```bibtex
@inproceedings{yu2025pixeldit,
      title={PixelDiT: Pixel Diffusion Transformers for Image Generation},
      author={Yongsheng Yu and Wei Xiong and Weili Nie and Yichen Sheng and Shiqiu Liu and Jiebo Luo},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2026},
}
```
