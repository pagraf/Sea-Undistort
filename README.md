
<img width="944" height="135" alt="sea-undistort_logo" src="https://github.com/user-attachments/assets/da6cc0b3-fc0c-4433-9333-48e2fd3f89db" />


# Sea-Undistort: A Synthetic Dataset for Restoring Through-Water Images in Airborne Bathymetric Mapping

[Sea-Undistort](https://doi.org/10.5281/zenodo.15639838) a synthetic dataset created using the open-source 3D graphics platform Blender. The dataset comprises 1200 image pairs, each consisting of 512×512 pixel RGB renderings of shallow underwater scenes. Every pair includes a “non-distorted” image, representing minimal surface and column distortions, and a corresponding “distorted” version that incorporates realistic optical phenomena such as sun glint, wave-induced deformations, turbidity, and light scattering. These effects are procedurally generated to replicate the diverse challenges encountered in through-water imaging for bathymetry. The scenes are designed with randomized combinations of typical shallow-water seabed types, including rocky outcrops, sandy flats, gravel beds, and seagrass patches, capturing a wide range of textures, reflectance patterns, and radiometric conditions. Refraction is accurately modeled in both the distorted and non-distorted images to maintain geometric consistency with real underwater imaging physics.

In addition, camera settings are uniformly sampled within specific ranges to ensure diverse imaging conditions. Sensor characteristics include a physical width of 36 mm and effective pixel widths of 4000 or 5472 pixels. Focal lengths of 20 mm and 24 mm are simulated with only the central 512x512 pixels rendered. Camera altitude ranges from 30 m to 200 m, resulting in a ground sampling distance (GSD) between 0.014 m and 0.063 m. Average depths range from –0.5 m to –8 m, with a maximum tilt angle of 5°. Sun elevation angles between 25° and 70°, along with varying atmospheric parameters (e.g., air, dust), are used to simulate different illumination conditions. Generated images are accompanied by a .json file containing this metadata per image. 

# Package for benchmarking Sea-Undistort

Sea-Undistort is designed to support supervised training of deep learning models for through-water image enhancement and correction, enabling generalization to real-world conditions where undistorted ground truth is otherwise unobtainable.
<br />
<br />
[![MagicBathy](https://img.shields.io/badge/MagicBathy-Project-red.svg)](https://www.magicbathy.eu) <br />
DOI of Dataset Repository [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15639838.svg)](https://doi.org/10.5281/zenodo.15639838)
<br />

# Package for benchmarking Sea-Undistort dataset.

This repository contains the code of the paper "M. Kromer, P. Agrafiotis, and B. Demir, "Sea-Undistort: A Dataset for Through-Water Image Restoration in High Resolution Airborne Bathymetric Mapping" submitted to IEEE GRSL"<br />

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://doi.org/10.48550/arXiv.2508.07760)

## Citation

If you find this repository useful, please consider giving a star ⭐.<br />
If you use the code in this repository or the dataset please cite:


>Kromer, M., Agrafiotis, P., & Demir, B. (2025). Sea-Undistort: A dataset for through-water image restoration in high resolution airborne bathymetric mapping. arXiv. https://arxiv.org/abs/2508.07760
>
```
@misc{kromer2025seaundistortdatasetthroughwaterimage,
      title={Sea-Undistort: A Dataset for Through-Water Image Restoration in High Resolution Airborne Bathymetric Mapping}, 
      author={Maximilian Kromer and Panagiotis Agrafiotis and Begüm Demir},
      year={2025},
      eprint={2508.07760},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2508.07760}, 
}
```
<br />

# Getting started

## Downloading the dataset

For downloading the dataset and a detailed explanation of it, please visit the MagicBathy Project website at [https://www.magicbathy.eu/Sea-Undistort.html](https://www.magicbathy.eu/Sea-Undistort.html)


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

## Pre-trained Deep Learning Models
We provide code and model weights for the following deep learning models that have been pre-trained on Sea-Undistort for through water image restoration:

| Model Names | Pre-Trained PyTorch Models                                                                                | 
| ----------- |-----------------------------------------------------------------------------------------------------------|
| NDR-Restore | [NDR-Restore.zip](https://drive.google.com/file/d/1mOFFbo6BiEMIFm8cQ0ysQcYOE8yz-Wqe/view?usp=sharing)  |
| ResShift | [ResShift.zip](https://drive.google.com/file/d/1GFuzjS7o1plvUDDA5UqBY7OLXLLeTgfm/view?usp=sharing)|
| ResShift+EF | [ResShift+EF.zip](https://drive.google.com/file/d/1vs-sTJqbAUHjuQ6SBSMzPRT416RFHsTG/view?usp=sharing)  |

To achieve the results presented in the paper, use the parameters and the specific train-evaluation splits provided in the dataset. 

## Example testing results
Example imagery from the Agia Napa area: : (ul) original patches; restorations using (ur) NDR-Restore, (bl) ResShift, and (br) ResShift+EF.
<img width="200" height="200" alt="IMG_4085_org" src="https://github.com/user-attachments/assets/191ab0f3-0c87-4a3b-9248-9f2aaa88bb6c" />
<img width="200" height="200" alt="IMG_4085_ndr" src="https://github.com/user-attachments/assets/251779e3-e2b1-4997-8bd8-0d98a08c8de5" />
<img width="200" height="200" alt="IMG_4085_resshift" src="https://github.com/user-attachments/assets/ae84cc41-68a3-4785-b377-880cf83bce1a" />
<img width="200" height="200" alt="IMG_4085_resshift+ef" src="https://github.com/user-attachments/assets/eace004a-e2c4-4db2-8f71-5b1b23e2da25" />
<br />

## Authors
Maximilian Kromer and Panagiotis Agrafiotis [https://www.user.tu-berlin.de/pagraf/](https://www.user.tu-berlin.de/pagraf/)


## Feedback
Feel free to give feedback, by sending an email to: agrafiotis@tu-berlin.de
<br />



# Funding
This work is part of **MagicBathy project funded by the European Union’s HORIZON Europe research and innovation programme under the Marie Skłodowska-Curie GA 101063294**. Work has been carried out at the [Remote Sensing Image Analysis group](https://rsim.berlin/). For more information about the project visit [https://www.magicbathy.eu/](https://www.magicbathy.eu/).
