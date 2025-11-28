# Patch-level Anatomy- and Style-Guided 3D Diffusion for Multi-site T1 MRI Harmonization

This repository contains the official code for the paper **“Patch-level Anatomy- and Style-Guided 3D Diffusion for Multi-site T1 MRI Harmonization”**, submitted to **MIDL 2026**.

![figure_overview_page-0001](https://github.com/user-attachments/assets/c76ce139-3744-49ac-b84f-a2bd0d45ce06)

---

## 1. Datasets
The lists of MRI volumes used for training and testing are provided in the **Train Dataset** and **Test Dataset** files.

All experiments were conducted on the **SRPBS dataset**.

---

## 2. Training Script
Training is performed using a Jupyter notebook.  
Before launching the training, you must preprocess all volumes using the following steps:

1. Skull stripping  
2. Bias-field correction  
3. Affine registration to MNI space (1 mm)  
4. Cropping to **160 × 192 × 160** voxels  
5. Median-based intensity normalization  

Update the paths for logs and checkpoint saving directly in the notebook before running the training.

---

## 3. Inference Script
To run the inference script on your preprocessed images, you can use the following model weights: **path_to_drive**.

You must specify the following parameters:

- `--lamdba` Guidance lambda value (default: 0.8)  
- `--brain_folder` Folder containing the preprocessed volumes to harmonize  
- `--save_dir` Folder where the harmonized volumes will be saved  
