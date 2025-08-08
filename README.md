


# Sea-Undistort: A Synthetic Dataset for Restoring Through-Water Images in Airborne Bathymetric Mapping

[Sea-Undistort](https://doi.org/10.5281/zenodo.15639838) a synthetic dataset created using the open-source 3D graphics platform Blender. The dataset comprises 1000 image pairs, each consisting of 512×512 pixel RGB renderings of shallow underwater scenes. Every pair includes a “non-distorted” image, representing minimal surface and column distortions, and a corresponding “distorted” version that incorporates realistic optical phenomena such as sun glint, wave-induced deformations, turbidity, and light scattering. These effects are procedurally generated to replicate the diverse challenges encountered in through-water imaging for bathymetry. The scenes are designed with randomized combinations of typical shallow-water seabed types, including rocky outcrops, sandy flats, gravel beds, and seagrass patches, capturing a wide range of textures, reflectance patterns, and radiometric conditions. Refraction is accurately modeled in both the distorted and non-distorted images to maintain geometric consistency with real underwater imaging physics.

In addition, camera settings are uniformly sampled within specific ranges to ensure diverse imaging conditions. Sensor characteristics include a physical width of 36 mm and effective pixel widths of 4000 or 5472 pixels. Focal lengths of 20 mm and 24 mm are simulated with only the central 512x512 pixels rendered. Camera altitude ranges from 30 m to 200 m, resulting in a ground sampling distance (GSD) between 0.014 m and 0.063 m. Average depths range from –0.5 m to –8 m, with a maximum tilt angle of 5°. Sun elevation angles between 25° and 70°, along with varying atmospheric parameters (e.g., air, dust), are used to simulate different illumination conditions. Generated images are accompanied by a .json file containing this metadata per image. 

Sea-Undistort is designed to support supervised training of deep learning models for through-water image enhancement and correction, enabling generalization to real-world conditions where undistorted ground truth is otherwise unobtainable.
<br />
<br />
[![MagicBathy](https://img.shields.io/badge/MagicBathy-Project-red.svg)](https://www.magicbathy.eu) <br />
DOI of Dataset Repository [![DOI](https://zenodo.org/badge/748123214.svg)](https://doi.org/10.5281/zenodo.15639838)
<br />

# Package for benchmarking Sea-Undistort dataset.

This repository contains the code of the paper "M. Kromer, P. Agrafiotis, and B. Demir, "Sea-Undistort: A Dataset for Through-Water Image Restoration in High Resolution Airborne Bathymetric Mapping" submitted to IEEE GRSL"<br />

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2405.15477) [![IEEE](https://img.shields.io/badge/IEEE-Paper-blue.svg)](https://ieeexplore.ieee.org/document/10641355)

## Citation

If you find this repository useful, please consider giving a star ⭐.<br />
If you use the code in this repository or the dataset please cite:

>....
```

```
<br />

# Getting started

## Downloading the dataset

For downloading the dataset and a detailed explanation of it, please visit the MagicBathy Project website at [https://www.magicbathy.eu/magicbathynet.html](https://www.magicbathy.eu/magicbathynet.html)


## Dataset structure
The folder structure should be as follows:
```
┗ 📂 Sea-Undistort/
  ┣ 📜 render_0000_ground.png
  ┣ 📜 render_0000_no_sunglint.png
  ┣ 📜 render_0000_no_waves.png
  ┣ 📜 render_0000.png
  ┣ 📜 render_0001_ground.png
  ┣ 📜 render_0001_no_sunglint.png
  ┣ 📜 render_0001_no_waves.png
  ┣ 📜 render_0001.png
  ┣ 📜 ...
  ┣ 📜 render_1199_ground.png
  ┣ 📜 render_1199_no_sunglint.png
  ┣ 📜 render_1199_no_waves.png
  ┣ 📜 render_1199.png
  ┣ 📜 scene_settings.json
  ┗ 📜 LICENSE_and_info.txt
```

## Clone the repo

`git clone https://github.com/pagraf/Sea-Undistort.git`

## Installation Guide
The requirements are easily installed via Anaconda (recommended):

`conda env create -f environment.yml`

After the installation is completed, activate the environment:

`conda activate magicbathynet`

Open Jupyter Notebook:

`jupyter notebook`

## Train and Test the models
To train and test the models use 

## Pre-trained Deep Learning Models
We provide code and model weights for the following deep learning models that have been pre-trained on Sea-Undistort for through water image restoration:

| Model Names | Pre-Trained PyTorch Models                                                                                                                | 
| ----------- |----------| ---- |----------------------------------------------------------------------------------------------------------------------------------------------|
| NDR-Restore | [unet_aerial_an.zip](https://drive.google.com/file/d/1vrYwOGEPbiuyvAmtE8-SfbDfzVWU8oMD/view?usp=sharing) |
| ResShift | [segformer_aerial_an.zip](https://drive.google.com/file/d/1rUr_KvAgOKwBmykLoprUy4Aw4fCiYGIm/view?usp=sharing)            |
| ResShift+EF | [unet_aerial_pl.zip](https://drive.google.com/file/d/1PVIRvFpiw4xf6xgLCF4Bzhpb_2wD3Q3G/view?usp=sharing) |

To achieve the results presented in the paper, use the parameters and the specific train-evaluation splits provided in the dataset. 

## Example testing results
Example patch of the Agia Napa area (left), pixel classification results obtained by U-Net (middle) and predicted bathymetry obtained by MagicBathy-U-Net (right). For more information on the results and accuracy achieved read our [paper](https://www.magicbathy.eu/). 


![img_410_aerial](https://github.com/pagraf/MagicBathyNet/assets/35768562/132b4166-b012-476b-9653-b511ede2c6f3)
![aerial_410_unet](https://github.com/pagraf/MagicBathyNet/assets/35768562/8a293815-87b4-4f45-b5de-c99f7c827bb5)
![depth_410_aerial](https://github.com/pagraf/MagicBathyNet/assets/35768562/7995efd7-f85e-4411-8037-4a68c9780bfb)



## Authors
Maximilian Kromer and Panagiotis Agrafiotis [https://www.user.tu-berlin.de/pagraf/](https://www.user.tu-berlin.de/pagraf/)

## Feedback
Feel free to give feedback, by sending an email to: agrafiotis@tu-berlin.de
<br />
<br />

# Funding
This work is part of **MagicBathy project funded by the European Union’s HORIZON Europe research and innovation programme under the Marie Skłodowska-Curie GA 101063294**. Work has been carried out at the [Remote Sensing Image Analysis group](https://rsim.berlin/). For more information about the project visit [https://www.magicbathy.eu/](https://www.magicbathy.eu/).
