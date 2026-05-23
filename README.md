# polyp-Seg
This repository provides the official code and the dataset.

# Title
Real-Time Colorectal Polyp Segmentation on High-DeûnitionNarrow-Band Imaging Data via a Lightweight Multi-ScaleAttention Network

## Description
This repository contains the official PyTorch implementation and dataset for the paper detailing TNA-Net, a real-time and lightweight segmentation framework tailored for Narrow-Band Imaging (NBI) colonoscopy. The proposed network achieves an optimal trade-off between segmentation accuracy and computational efficiency (95 FPS).

## Overview of the Architecture
This study proposes an efficient colorectal polyp segmentation network built upon MobileNetV4, designed to enhance computational efficiency while maintaining high segmentation accuracy. The overall architecture is illustrated in Fig. Within a U-Net–like framework, MobileNetV4 is adopted as a lightweight backbone, and both the ACSSA module and the RCAConv Module are integrated to strengthen multi-scale feature representation.
<img width="2154" height="2154" alt="image" src="https://github.com/user-attachments/assets/eda37397-1946-4842-a798-664c92787c3e" />



## Dataset Details
The self-built dataset comprises 1470 high-quality NBI colonoscopy images with corresponding pixel-level segmentation annotations. It consists of two components: 1,000 NBI images publicly released by the 7th International Symposium on Image Computing and Digital Medicine (ISICDM 2024) (conference website:https://www.imagecomputing.org/isicdm2024), and 470 NBI images collected from clinical colonoscopy procedures at Xiangyang Central Hospital. The self-collected images were retrospectively screened by a junior endoscopist from colonoscopy cases conducted between January 1, 2023, and March 10, 2024. The inclusion criteria require patients to have polyps observed under NBI mode along with corresponding pathological diagnostic reports, where serves as the gold standard for determining the presence and type of polyps. A total of 470 images from 72 patients were collected, and each image was pixel-level annotated by experienced endoscopists to produce high-quality segmentation masks. The data collected from Xiangyang Central Hospital strictly adhere to the ethical principles outlined in the Declaration of Helsinki, and approval is obtained from the Medical Ethics Committee of Xiangyang Central Hospital (No. 2024–145). Written informed consent is waived due to the retrospective nature of the study, as permitted by the committees. The dataset from Xiangyang Central Hospital can be downloaded using the following links:Google Drive:[link](https://drive.google.com/file/d/1_i2oLwRR0ETTpBQKXTC5sF99iAcGoNlB/view?usp=drive_link)
Examples of raw NBI images and their corresponding masks from self-collected dataset.
<img width="2137" height="783" alt="image" src="https://github.com/user-attachments/assets/858711b6-da77-4c11-9ad4-7433a6857277" />


The image below presents several representative complex cases from our constructed dataset. The polyps have been annotated for reference, but these cases also present a variety of visual challenges. (a) Blurred edges and reflections fuse the polyp with the mucosa. (b) Low contrast with mucosal folds concealing polyp boundaries. (c) Mottled surface absorbed by vascular texture. (d) Multiple small polyps with low contrast are obscured by colonic secretion. (e) Inapparent lesion boundary due to extreme contrast loss.
<img width="2204" height="464" alt="image" src="https://github.com/user-attachments/assets/34ac5c5a-0ba9-4692-9ce8-a735bf26de6f" />


## Code Information
This repository contains seven core Python scripts that make up the TNA-Net framework:
**`model.py`**: The core implementation of the TNA-Net architecture. This file includes the definitions for the Anti-Aliasing Feature Mixer (AAFM), the Texture-Preserving Gated Bridge (TPG-Bridge), and the Context-Aware Restoration Stream (CARS) decoder.
**`mobilenetv4.py`**: Implements the lightweight MobileNetV4 backbone used by TNA-Net for efficient feature extraction.
**`train.py`**: The main script used to train the model. It handles the training loop, loss calculation, and saving the best model weights.
**`test.py`**: The script for evaluating the trained model's segmentation performance on the test datasets.
**`time.py`**: A dedicated script to measure the model's inference speed (FPS), computational cost (FLOPs), and parameter count (Params) to validate its real-time capabilities.
**`metrics.py`**: Contains the implementations for all evaluation metrics used in the study, including Dice coefficient, mean Intersection over Union (mIoU), and Recall.
**`utils.py`**: Includes auxiliary utility functions required for the pipeline.

## Requirements
The code is built and tested on Python 3.8+ and PyTorch. Below are the main dependencies required to run the project:
* Python >= 3.8
* PyTorch >= 1.10.0
* torchvision >= 0.11.0
* numpy
* opencv-python
* tqdm (optional, for training progress bars)
*Tip: You can easily install the necessary environment using standard pip commands (e.g., `pip install torch torchvision numpy opencv-python`).*

## Usage Instructions
1. **Data Preparation:** Place the datasets in the `./data` directory.
2. **Training:** Run the training script (e.g., `python train.py --batch_size 8 --epochs 300`).
3. **Evaluation:** Run the testing script to output the FPS and Dice metrics ( `python test.py --weights ./checkpoints/best.pth`).

## Main Result
Qualitative comparison of segmentation results between proposed model and five state-of-the-art methods. (Green areas indicate true positives (correct segmentation), red areas show false positives (over-segmentation), and yellow areas represent false negatives (under-segmentation).)
<img width="2206" height="976" alt="image" src="https://github.com/user-attachments/assets/645b361c-0eb1-4a04-b3f9-3d23dfcab022" />


## Acknowledgments
We sincerely thank the ISICDM2024 organizing committee for providing the NBI images dataset from Challenge Project 1, which has greatly supported this research.
