# ResShift Inference & Training Instructions

This guide explains how to run ResShift and ResShift+EF with pretrained weights. ResShift+EF requires a sunglint mask.

## Setup

### 1. Clone ResShift Repository

```bash
git clone https://github.com/zsyOAOA/ResShift.git
cd ResShift
```

### 2. Add Modified Files

Place/Replace the following files from the `ResShift-for-SU` folder into the cloned `ResShift` project:
- `ResShift-for-SU/inference_resshift.py` → `ResShift/inference_resshift.py`
- `ResShift-for-SU/sampler.py` → `ResShift/sampler.py`
- `ResShift-for-SU/datapipe/datasets.py` → `ResShift/datapipe/datasets.py`
- `ResShift-for-SU/utils/util_image.py` → `ResShift/utils/util_image.py`

For training only:
- `ResShift-for-SU/trainer.py` → `ResShift/trainer.py`

### 3. Download Pretrained Weights

Place downloaded weights in `ResShift/weights/`:
- Autoencoder (required for both): `autoencoder_vq_f4.pth`
- ResShift model: e.g., `su_resshift_200000.pth`
- ResShift+EF model: e.g., `su_resshift+ef_160000.pth`

Download links:
- VQGAN Autoencoder: `https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth`
- ResShift weights: `https://drive.google.com/file/d/1GFuzjS7o1plvUDDA5UqBY7OLXLLeTgfm/view?usp=sharing`
- ResShift+EF weights: `https://drive.google.com/file/d/1vs-sTJqbAUHjuQ6SBSMzPRT416RFHsTG/view?usp=sharing`

### 4. Prepare Configuration

Place configuration files in `ResShift/configs/`:
- `ResShift-for-SU/configs/sea_undistort_resshift.yaml` → `ResShift/configs/sea_undistort_resshift.yaml`
- `ResShift-for-SU/configs/sea_undistort_resshift_ef.yaml` → `ResShift/configs/sea_undistort_resshift_ef.yaml`

Important: Verify all paths in the YAML configuration(s) (data roots, weight paths).

## Running Inference

### Basic Command Structure

```bash
python inference_resshift.py -i <input_path> -o <output_path> --task <su-resshift|su-resshift+ef> --scale 1 [--mask_path <mask_dir>]
```

- `--mask_path` is required when `--task su-resshift+ef`.

### Example Commands

ResShift+EF (with mask):
```bash
python inference_resshift.py \
  -i <path_to_input_images> \
  -o <path_to_output_folder> \
  --mask_path <path_to_mask_dir> \
  --task su-resshift+ef \
  --scale 1
```

ResShift (no mask):
```bash
python inference_resshift.py \
  -i <path_to_input_images> \
  -o <path_to_output_folder> \
  --task su-resshift \
  --scale 1
```

### Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `-i` | Directory containing input images | Yes |
| `-o` | Directory to save processed images | Yes |
| `--task` | Choose model: `su-resshift` or `su-resshift+ef` | Yes |
| `--scale` | Image scale factor (e.g., 1) | Yes |
| `--mask_path` | Directory with mask images (ResShift+EF only) | Required for `su-resshift+ef` |

## Running Training

### Basic Command Structure

```bash
torchrun --nproc_per_node=<num_gpus> main.py --cfg_path <config_file> --save_dir <logging_dir> [--resume <checkpoint_path>]
```

### Example Command

```bash
torchrun --nproc_per_node=2 main.py \
  --cfg_path configs/sea_undistort_resshift_ef.yaml \
  --save_dir ./logging
```

## File Structure

```
ResShift/
├── inference_resshift.py
├── sampler.py
├── trainer.py
├── configs/
│   ├── sea_undistort_resshift.yaml
│   └── sea_undistort_resshift_ef.yaml
├── weights/
│   ├── autoencoder_vq_f4.pth
│   ├── su_resshift_200000.pth
│   └── su_resshift+ef_160000.pth
├── datapipe/
│   ├── datasets.py
│   └── [Other Files]
├── utils/
│   ├── util_image.py
│   └── [Other Files]
└── [ResShift Files]
```

### Credits and Attribution

- This repository includes only the files we modified to suit our research needs.
- The full original ResShift project is publicly available: `https://github.com/zsyOAOA/ResShift`.
- We thank the original authors for their valuable contributions.
