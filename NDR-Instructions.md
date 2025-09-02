# NDR-Restore Inference Instructions

This guide explains how to run NDR-Restore for image restoration using pretrained weights.

## Setup

### 1. Clone NDR-Restore Repository

```bash
git clone https://github.com/mdyao/NDR-Restore.git
cd NDR-Restore
```

### 2. Add Inference & Train Files

Place the following files from the `NDR-Restore-for-SU` folder into the `NDR-Restore` project root:
- `NDR-Restore-for-SU/inference.py` → `NDR-Restore/inference.py`
- `NDR-Restore-for-SU/inference_overlap.py` → `NDR-Restore/inference_overlap.py`

For training only:
- `NDR-Restore-for-SU/train.py` → `NDR-Restore/train.py`

### 3. Download Pretrained Weights

Download the pretrained model weights:
- **Download Link**: [NDR-Restore.zip](https://drive.google.com/file/d/1mOFFbo6BiEMIFm8cQ0ysQcYOE8yz-Wqe/view?usp=sharing)
- **Extract** the downloaded file
- **Place** the weights file (e.g., `su_ndr_latest.pth`) in the `pretrained/` folder

### 4. Prepare Configuration

Place the configuration file into the options directory:
- `NDR-Restore-for-SU/options/inference_sea_undistort_ndr.yml` → `NDR-Restore/options/inference_sea_undistort_ndr.yml`

For training only:
- `NDR-Restore-for-SU/options/train_sea_undistort_ndr.yml` → `NDR-Restore/options/train_sea_undistort_ndr.yml`

**Important**: Verify the weights path in the inference YAML file:
```yaml
path:
  pretrain_model_G: pretrained/su_ndr_latest.pth
```

**Important for training only:**: Verify the paths in the YAML file:
`train_sea_undistort_ndr.yml`

## Running Inference

### Basic Command Structure

```bash
python inference_overlap.py -opt <config_file> -input_folder <input_path> -output_folder <output_path>
```

### Example Command

```bash
python inference_overlap.py \
    -opt options/inference_sea_undistort_ndr.yml \
    -input_folder <path_to_input_images> \
    -output_folder <path_to_output_folder>
```

### Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `-opt` | Path to YAML configuration file | Yes |
| `-input_folder` | Directory containing input images | Yes |
| `-output_folder` | Directory to save processed images | Yes |


## Running Training

### Basic Command Structure

```bash
python train.py -opt <config_file>
```

### Example Command

```bash
python train.py -opt options/train_sea_undistort_ndr.yml
```

### Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `-opt` | Path to YAML configuration file | Yes |

## Configuration Options

### Patch Processing Parameters

You can modify these parameters in the `inference_overlap.py` file:

- **`patch_size`**: Maximum patch size for processing
  - Default: 1024

- **`overlap`**: Overlap between patches (inference_overlap.py only)
  - Default: 128
  - Prevents artifacts at patch boundaries

## Output

- Processed images are saved in the specified output folder
- Original filenames are preserved
- Images are automatically normalized and converted to RGB format

## File Structure

```
NDR-Restore/
├── inference.py              # Basic inference script
├── inference_overlap.py      # Advanced inference with patch overlap
├── train.py                  # Basic training script
├── options/
│   ├── inference_sea_undistort_ndr.yml
│   └── train_sea_undistort_ndr.yml
├── pretrained/
│   └── su_ndr_latest.pth
└── [NDR-Restore Files]
```

### Credits and Attribution

- This repository includes only the files we modified to suit our research needs.
- The full original NDR-Restore project is publicly available: [NDR-Restore repository](https://github.com/mdyao/NDR-Restore).
- We thank the original authors for their valuable contributions.
