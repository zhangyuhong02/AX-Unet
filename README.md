# Pancreas Image Segmentation Using AX-Unet

## Introduction

According to the Report on Cancer from National Cancer Institute in 2021, pancreatic cancer is the third leading cause of cancer-related death in the United States ([1](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.894970/full#B1)). The identification and analysis of pancreatic region play an important role in the diagnosis of pancreatic tumors. As an important and challenging problem in medical image analysis, pancreas is one of the most challenging organs for automated segmentation, which aim to assign semantic class labels to different tomography image regions in a data-driven learning fashion. Usually, such a learning problem encounters numerous difficulties such as severe class imbalance, background clutter with confusing distractions, and variable location and geometric features. According to statistical analysis, pancreas occupies less than 0.5% fraction of entire CT volume ([2](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.894970/full#B2)), which has a visually blurry inter-class boundary with respect to other tissues.

In this article, we combine the advantages of deepLabV series, Unet, and Xception networks to present a novel deep learning framework AX-Unet for pancreas CT image segmentation to assist physicians for the screening of pancreatic tumors. The whole AX-Unet still preserves the encoder-decoder structure of Unet. In our framework, we incorporate a modified atrous spatial pyramid pooling (ASPP) module to learn the location information. The modified ASPP can also extract multi-level contextual information to reduce information loss during downsampling. We also introduce a special group convolution operation on the feature map at each level to decouple the information between channels, achieving more complete information extraction. Finally, we employ an explicit boundary-aware loss function to tackle the blurry boundary problem. The experimental results on two public datasets validated the superiority of the proposed AX-Unet model to the states-of-the-art methods.

In summary, we propose a novel deep learning framework AX-Unet for pancreas CT image segmentation. Our framework has several advantages as follows.

1. In our framework, we introduce a special group convolution, depth-wise separable convolution, to decouple the two types of information based on the assumption that inter-channel and intra-channel information are not correlated. This design can achieve better performance with even less computation than the normal convolution.

2. We restructure the ASPP module, and the extraction and fusion of multi-level global contextual features is achieved by multi-scale dilate convolution, which enables a better handling of the large scale variance of the objects without introducing additional operations. The efficacy of the restructured ASPP is validated in our ablation studies on foreground target localization.

3. We propose a loss function that can explicitly perceive the boundary of the target and combine the focal loss and generalized dice loss (GDL) to solve the problem of category imbalance. The weighted sum of the above parts is used as our final loss function, which can explicitly perceive the boundary of the target.

4. We segment a large number of external unlabeled pancreas images using our trained model. The analysis of the imagomics features of the pancreatic region shows a significant difference between patients with pancreatic tumors and normal people (*p* ≤ 0.05), which may provide a promising and reliable way to assist physicians for the screening of pancreatic tumors.

## Datasets

Following previous work of pancreas segmentation, two different abdominal CT datasets are used:

​		● As one of the largest and most authoritative Open Source Dataset in pancreas segmentation, the NIH pancreas segmentation dataset sourced from TCIA (The Cancer Imaging Archive) provides an easy and fair way for method comparisons ([51](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.894970/full#B51)). The dataset contains 82 contrast-enhanced abdominal CT volumes. The resolution of each CT scan is 512 × 512 × L, where L have a range of 181 to 466 which is the number of sampling slices along the long axis of the body. The dataset contains a total of 19,327 slices from the 82 subjects, and the slice thickness varies from 0.5 to 1.0 mm. Only the CT slices containing the pancreas are used as input to the system. We followed the standard four-fold cross-validation, where the dataset is split to four folds, each fold contains images of 20 subjects, and the proposed model was trained on 3 folds and tested on the remaining fold.

​		● The Medical Segmentation Decathlon ([52](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.894970/full#B52)) is a challenge to test the generalizability of machine learning algorithms when applied to 10 different semantic segmentation tasks. In addition, we use the pancreas part in modality of portal venous phase CT from Memorial Sloan Kettering Cancer Center. We used the official training-test splits where 281 subjects are in training set and 139 subjects are in test set.

## Train in Datasets🚀

- First you need to clone the project locally, or you can run it using google colab

  ```
  https://github.com/zhangyuhong02/AX-Unet.git
  ```
### Medical Image Preprocessing

  #### Dependencies
  The following libraries and dependencies are required:

  - numpy
  - nibabel
  - util (custom library)
  - tqdm
  - PIL (Python Imaging Library)
  - paddle.vision.transforms.functional (part of PaddlePaddle library)

  #### Data Paths
  The script assumes the presence of the following data paths:

  - `volumes_path`: Path to the directory containing CT scan data
  - `labels_path`: Path to the directory containing corresponding labels
  - `data_preprocess_path`: Path to save preprocessed data
  - `label_preprocess_path`: Path to save preprocessed labels

  #### Preprocessing Steps

  1. Loading the libraries and dependencies:
     - `numpy`, `nibabel`, `util`, `tqdm`: Required libraries for data processing and progress monitoring
     - `PIL.Image`, `paddle.vision.transforms.functional`: Libraries for image processing

  2. Setting up the data paths:
     - `volumes_path`: Directory path for CT scan data
     - `labels_path`: Directory path for corresponding labels
     - `data_preprocess_path`: Directory path to save preprocessed data
     - `label_preprocess_path`: Directory path to save preprocessed labels

  3. Retrieving the list of CT scan data and labels:
     - `volumes`: List of files in the `volumes_path` directory
     - `labels`: List of files in the `labels_path` directory
     - Asserts that the number of volumes and labels are equal

  4. Preprocessing loop:
     - Iterates over each CT scan and corresponding label
     - Preprocessing steps are applied to each data pair

  5. Loading CT scan and label data:
     - Uses `nibabel.load` to load CT scan data from the volumes directory
     - Uses `nibabel.load` to load corresponding label data from the labels directory

  6. Preprocessing operations:
     - Noise removal: Clips CT scan data to the range of -1024 to 1024
     - Contrast enhancement: (Commented out) Adjusts the contrast of CT scan data
     - Label clipping: Clips label data to specific values (e.g., class 1)
     - Normalization: Normalizes CT scan data between 0 and 1

  7. Slice-based processing and saving:
     - Loops through each slice in the CT scan volume
     - Processes the CT scan and label data for each slice
     - Reshapes and transposes the data to the required format
     - Saves the preprocessed CT scan and label data as `.npy` files

###   Train model

Just run `pipline.ipynb`

### Model evaluation

Just run `boundary aware loss building.ipynb`
