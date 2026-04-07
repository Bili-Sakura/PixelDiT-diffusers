# Class-to-Image Generation (ImageNet)

PixelDiT-XL class-conditioned image generation on ImageNet.

## Pre-trained Models

| Model | Epochs | Resolution | gFID | Checkpoint |
|:---:|:---:|:---:|:---:|:---:|
| PixelDiT-XL | 80  | 256×256 | 2.36 | [🤗 HuggingFace](https://huggingface.co/nvidia/PixelDiT-ImageNet/resolve/main/imagenet256_pixeldit_xl_epoch80.ckpt) |
| PixelDiT-XL | 160 | 256×256 | 1.97  | [🤗 HuggingFace](https://huggingface.co/nvidia/PixelDiT-ImageNet/resolve/main/imagenet256_pixeldit_xl_epoch160.ckpt) |
| PixelDiT-XL | 320 | 256×256 | 1.61 | [🤗 HuggingFace](https://huggingface.co/nvidia/PixelDiT-ImageNet/resolve/main/imagenet256_pixeldit_xl_epoch320.ckpt) |
| PixelDiT-XL | 850 | 512×512 | 1.81  | [🤗 HuggingFace](https://huggingface.co/nvidia/PixelDiT-ImageNet/resolve/main/imagenet512_pixeldit_xl.ckpt) |


## Data Preparation

For ImageNet 256×256 dataset preparation, please follow the instructions in [REPA-E](https://github.com/End2End-Diffusion/REPA-E). Specifically, download and extract the ImageNet-1K training split, then run the preprocessing script provided in the REPA-E repository.

For ImageNet 512×512, we use the [dataset_tool.py](https://github.com/NVlabs/edm2/blob/main/dataset_tool.py) from [EDM2](https://github.com/NVlabs/edm2) to prepare the data.

After preprocessing, update the `data_dir` field in the config YAML (e.g., `configs/pix256_xl.yaml` or `configs/pix512_xl.yaml`) to point to your processed data directory.

## Training

### ImageNet 256×256

```bash
cd c2i/
bash train_c2i.sh --num-gpus 8 --config configs/pix256_xl.yaml
```

### ImageNet 512×512

```bash
cd c2i/
bash train_c2i.sh --num-gpus 8 --config configs/pix512_xl.yaml
```

### Resume from Checkpoint

```bash
cd c2i/
bash train_c2i.sh --num-gpus 8 --config configs/pix256_xl.yaml \
  --ckpt-path /path/to/checkpoint.ckpt
```

### `train_c2i.sh` Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | configs/pix256_xl.yaml | Config YAML path |
| `--ckpt-path` | (empty) | Checkpoint to resume from |

Auto-resume is enabled by default in the config (`auto_resume: true`). If a previous checkpoint exists in the output directory, training resumes automatically.

Checkpoints are auto-downloaded from HuggingFace if the file does not exist locally. Just pass the filename to `--ckpt_path`.

## Evaluation

Evaluation generates 50K images via `main.py predict`, then computes FID using the [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations).

### Step 1: Generate Samples

All commands below generate images under `c2i/train_logs/`. Override sampler params on the CLI as needed.

**Epoch 80 (ImageNet 256×256):**

```bash
cd c2i/
torchrun --nproc_per_node=8 main.py predict \
  -c configs/pix256_xl.yaml \
  --ckpt_path=imagenet256_pixeldit_xl_epoch80.ckpt \
  --model.diffusion_sampler.class_path=src.diffusion.FlowDPMSolverSampler \
  --model.diffusion_sampler.init_args.num_steps=100 \
  --model.diffusion_sampler.init_args.guidance=3.25 \
  --model.diffusion_sampler.init_args.timeshift=1.0 \
  --model.diffusion_sampler.init_args.guidance_interval_min=0.1 \
  --model.diffusion_sampler.init_args.guidance_interval_max=1.0 \
  --per_run_seed=false --seed_everything=5000
```

**Epoch 160 (ImageNet 256×256):**

```bash
cd c2i/
torchrun --nproc_per_node=8 main.py predict \
  -c configs/pix256_xl.yaml \
  --ckpt_path=imagenet256_pixeldit_xl_epoch160.ckpt \
  --model.diffusion_sampler.class_path=src.diffusion.FlowDPMSolverSampler \
  --model.diffusion_sampler.init_args.num_steps=100 \
  --model.diffusion_sampler.init_args.guidance=3.25 \
  --model.diffusion_sampler.init_args.timeshift=1.0 \
  --model.diffusion_sampler.init_args.guidance_interval_min=0.1 \
  --model.diffusion_sampler.init_args.guidance_interval_max=1.0 \
  --per_run_seed=false --seed_everything=5000
```

**Epoch 320 (ImageNet 256×256):**

```bash
cd c2i/
torchrun --nproc_per_node=8 main.py predict \
  -c configs/pix256_xl.yaml \
  --ckpt_path=imagenet256_pixeldit_xl_epoch320.ckpt \
  --model.diffusion_sampler.class_path=src.diffusion.FlowDPMSolverSampler \
  --model.diffusion_sampler.init_args.num_steps=100 \
  --model.diffusion_sampler.init_args.guidance=2.75 \
  --model.diffusion_sampler.init_args.timeshift=1.0 \
  --model.diffusion_sampler.init_args.guidance_interval_min=0.1 \
  --model.diffusion_sampler.init_args.guidance_interval_max=0.9 \
  --per_run_seed=false --seed_everything=1600
```

**ImageNet 512×512:**

```bash
cd c2i/
torchrun --nproc_per_node=8 main.py predict \
  -c configs/pix512_xl.yaml \
  --ckpt_path=imagenet512_pixeldit_xl.ckpt \
  --model.diffusion_sampler.class_path=src.diffusion.FlowDPMSolverSampler \
  --model.diffusion_sampler.init_args.num_steps=100 \
  --model.diffusion_sampler.init_args.guidance=3.5 \
  --model.diffusion_sampler.init_args.timeshift=2.0 \
  --model.diffusion_sampler.init_args.guidance_interval_min=0.1 \
  --model.diffusion_sampler.init_args.guidance_interval_max=1.0 \
  --per_run_seed=false --seed_everything=10000
```

### Sampler Settings Summary

| Setting | 80 ep (256) | 160 ep (256) | 320 ep (256) | 512×512 |
|---------|:-----------:|:------------:|:------------:|:-------:|
| CFG Scale | 3.25 | 3.25 | 2.75 | 3.5 |
| Steps | 100 | 100 | 100 | 100 |
| Time Shift | 1.0 | 1.0 | 1.0 | 2.0 |
| CFG Interval | [0.1, 1.0] | [0.1, 1.0] | [0.1, 0.9] | [0.1, 1.0] |
| Sampler | FlowDPMSolver | FlowDPMSolver | FlowDPMSolver | FlowDPMSolver |

### Step 2: Compute FID

After generating samples, compute FID with the [ADM evaluation toolkit](https://github.com/openai/guided-diffusion/tree/main/evaluations).

The generated `output.npz` is saved alongside the images in the predict output directory.
