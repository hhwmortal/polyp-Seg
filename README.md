# polyp-Seg
This repository provides the official code and the dataset.
## Overview of the Architecture
This study proposes an efficient colorectal polyp segmentation network built upon MobileNetV4, designed to enhance computational efficiency while maintaining high segmentation accuracy. The overall architecture is illustrated in Fig. Within a U-Net–like framework, MobileNetV4 is adopted as a lightweight backbone, and both the ACSSA module and the RCAConv Module are integrated to strengthen multi-scale feature representation.
<img width="692" height="396" alt="d92a6a11-aa20-4a00-af5d-edcf04125707" src="https://github.com/user-attachments/assets/0830abaa-540b-4999-8756-5f044cf4d9d3" />

## Dataset Details
The self-built dataset comprises 1470 high-quality NBI colonoscopy images with corresponding pixel-level segmentation annotations. It consists of two components: 1,000 NBI images publicly released by the 7th International Symposium on Image Computing and Digital Medicine (ISICDM 2024) (conference website:https://www.imagecomputing.org/isicdm2024), and 470 NBI images collected from clinical colonoscopy procedures at Xiangyang Central Hospital. The self-collected images were retrospectively screened by a junior endoscopist from colonoscopy cases conducted between January 1, 2023, and March 10, 2024. The inclusion criteria require patients to have polyps observed under NBI mode along with corresponding pathological diagnostic reports, where serves as the gold standard for determining the presence and type of polyps. A total of 470 images from 72 patients were collected, and each image was pixel-level annotated by experienced endoscopists to produce high-quality segmentation masks. The data collected from Xiangyang Central Hospital strictly adhere to the ethical principles outlined in the Declaration of Helsinki, and approval is obtained from the Medical Ethics Committee of Xiangyang Central Hospital (No. 2024–145). Written informed consent is waived due to the retrospective nature of the study, as permitted by the committees. The dataset from Xiangyang Central Hospital can be downloaded using the following links:Google Drive:[link](https://drive.google.com/file/d/1_i2oLwRR0ETTpBQKXTC5sF99iAcGoNlB/view?usp=drive_link)
Examples of raw NBI images and their corresponding masks from self-collected dataset.
<img width="865" height="325" alt="image" src="https://github.com/user-attachments/assets/d6681029-6f15-49e2-ac87-1f079449aae2" />

The image below presents several representative complex cases from our constructed dataset. The polyps have been annotated for reference, but these cases also present a variety of visual challenges. (a) Blurred edges and reflections fuse the polyp with the mucosa. (b) Low contrast with mucosal folds concealing polyp boundaries. (c) Mottled surface absorbed by vascular texture. (d) Multiple small polyps with low contrast are obscured by colonic secretion. (e) Inapparent lesion boundary due to extreme contrast loss.
<img width="865" height="179" alt="image" src="https://github.com/user-attachments/assets/b50b3324-9a76-44ad-b472-ba11c101616b" />

## Main Result
Qualitative comparison of segmentation results between proposed model and five state-of-the-art methods. (Green areas indicate true positives (correct segmentation), red areas show false positives (over-segmentation), and yellow areas represent false negatives (under-segmentation).)
<img width="1102" height="488" alt="image" src="https://github.com/user-attachments/assets/18ac0bcb-095f-41f7-8585-a9fa472f31ea" />

## Acknowledgments
We sincerely thank the ISICDM2024 organizing committee for providing the NBI images dataset from Challenge Project 1, which has greatly supported this research.
