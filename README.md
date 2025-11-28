# Patch-level Anatomy- and Style-Guided 3D Diffusion for Multi-site T1 MRI Harmonization

This is the code for the paper "Patch-level Anatomy- and Style-Guided 3D Diffusion for Multi-site T1 MRI Harmonization" submitted to MIDL 2026.

[figure_overview.pdf](https://github.com/user-attachments/files/23819860/figure_overview.pdf)

## 1. Datasets
The list of the MRI volumes used to train and test the model during training are available in the 'Train Dataset' and 'Test Dataset' files.
All the experiments have been running on the SRPBS sataset.

## 2. Training script
The script use for training is a notebook. Before using this script to train on the training volumes, you should preprocessd the volumes with the following steps:
(i) skull stripping
(ii) bias-field correction
(iii) affine registration to 1 mm MNI space
(iv) cropping to 160 × 192 × 160 voxels
(v) median-based intensity normalization

Change the path for the logs and to save your checkpoint directly in the notebook

### 3. Inference script
To run the inference script on your preprocessed images, you can use the following model weights : path_to_drive.
You need to specify the following parameter:
--lamdba: Guidance lambda value (default: 0.8)
--brain_folder: Folder with the preprocessed volumes to harmonize
--save_dir: Folder to save harmonized volumes
