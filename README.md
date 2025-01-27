# Med_GCD
PRDL:Prediction refinement and discriminative learning for generalized category discovery in medical image classification
# MONICA
MONICA: Benchmarking on Long-tailed Medical Image Classification

## Introduction
We build a unified codebase called Medical OpeN-source Long-taIled ClassifiCAtion (MONICA), which implements over 30 methods developed in long-tailed Learning and evaluated on
12 long-tailed medical datasets covering 6 medical domains.
![alt text](fig.png)

## Installation
First, clone the repo and cd into the directory:
```shell
git clone this repo.
cd MONICA
```
Then create a conda env and install the dependencies:
```shell
conda create -n MONICA
conda activate MONICA
conda env create -f MONICA.yml
```

## 1. Prepare Datasets

### Data Download
| Domain           | Dataset         | Link                                                                                   | License        |
|------------------|-----------------|----------------------------------------------------------------------------------------|----------------|
| Dermatology      | ISIC2019        | https://challenge.isic-archive.com/data/#2019                                          | CC-BY-NC       |
| Dermatology      | DermaMNIST      | https://medmnist.com/                                                                  | CC BY-NC 4.0   |
| Ophthalmology    | ODIR            | https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k            | not specified  |
| Ophthalmology    | RFMiD           | https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid | CC BY-NC 4.0   |
| Radiology        | OragnA/C/SMNIST | https://medmnist.com/                                                                  | CC BY 4.0   |
| Radiology        | CheXpert        | https://stanfordmlgroup.github.io/competitions/chexpert/                               | [Stanford University Dataset Research Use Agreement](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2) |
| Pathology        | PathMNIST       | https://medmnist.com/                                                                  | CC BY 4.0   |
| Pathology        | WILDS-Camelyon17 (In Progress)      | https://wilds.stanford.edu/datasets/                                                                  | CC0 1.0   |
| Hematology       | BloodMNIST      | https://medmnist.com/                                                                  | CC BY 4.0   |
| Histology        | TissueMNIST     | https://medmnist.com/                                                                  | CC BY 4.0   |
| Gastroenterology | KVASIR          | https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset                             | ODbL 1.0       |

Please follow the same license as the original datasets.
### Image Preprocessing for Non-Image Datasets
If you download from the correct links, most of the evaluated datasets are conducted in image format.
We need to process the MedMNIST data for the unified training. Please find ./utils/process_medmnist.ipynb for reference.
```python
for idx in range(train_images.shape[0]):
  img = train_images[idx]
  label = train_labels[idx][0]
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  save_name = 'train_%s_%s.jpg' %(idx, label)
  cv2.imwrite('%s/%s' %(target_dir+dataset, save_name), img_rgb)
```
To match the pre-split set in `./numpy/medmnist`, all the images in MedMNIST will be stored in the format **<em>{split}\_{image_idx}\_{label}.jpg</em>**.


## 2. The Structure of Pre-Split Numpy Files

For the unified benchmark, we have pre-split the `train/val/test` sets, which are stored in `numpy files`.

There are four numpy files for each dataset/setting.

Take the ISIC datasets as an example, we have the three split files which start with the split `'train/val/test'` followed by the imbalance ratio `100/200/500`, e.g, `train_100.npy`.

We also have one dictionary file, e.g, `dic.npy`.

In `./dataset/dataloader.py`, you can find how we access the image_name and its label:

```python
self.np_dict = np.load(dict_path,allow_pickle=True).item()
self.np_path = np.load(np_path,allow_pickle=True)
self.img_name = []
self.img_label = []
for _ in self.np_path:
  self.img_name.append(_)
  self.img_label.append(self.np_dict[_])
```

### Customized Datasets
You can conduct your datasets in the following format:
```python
import numpy as np
import random
data_name = ['image_%s'%i for i in range(1000) ]
label = np.random.randint(0,10,1000)
dic = {data_name[i]:label[i] for i in range(1000)}
random.shuffle(data_name)
train = data_name[:700]
val = data_name[700:800]
test = data_name[800:1000]
np.save('train',train)
np.save('val',val)
np.save('test',test)
np.save('dic',dic)
```
And replace the `img_path` and `np_path` with your numpy files in the config files.

## 3. Start Training

### Config Files

Take the isic_GCL_2nd as the example (only keep some important hyperparameters).
```yaml
general: # Define some general parameters.
  img_size: 224
  seed: 1
  num_classes: 8
  dataset_name: 'isic'
  method: 'GCL_2nd'

model:
  if_resume: True # If this is set as True, the model will load for the resume_path.
  resume_path: './outputs/isic/100_GCL_224_resnet50_True_256_1_50/best.pt' # If if_resume is set as False, this will not work.
  if_freeze_encoder: True # If if_resume is set as False, this will not work.
  model_name: resnet50
  pretrained: True # If load the default pretrained weights provided by timm (from huggingface).


datasets:
  sampler: GCL # Sampler strategy.
  img_path: '/mnt/sda/datasets/isic2019/train/' # The image will be loaded as `img_path + name stored in np_path'. So if you can leave this blank if the full path is stored in 'np_path'.
  train:
    np_path: './numpy/isic/train_100.npy'
    dict_path: './numpy/isic/dic.npy' # Make sure this is consistency to the keys stored in the dic files.
  val:
    np_path: './numpy/isic/val_100.npy'
    dict_path: './numpy/isic/dic.npy'
  test:
    np_path: './numpy/isic/test_100.npy'
    dict_path: './numpy/isic/dic.npy'
  transforms:
    train: 'strong'
    val_test: 'crop'

```
### Support Methods
| Methods                | Paper                                                                                      | Link (TBD) | Offical Codes (TBD)  |
|------------------------|--------------------------------------------------------------------------------------------|------------|----------------------|
| ERM (Crossentropy)     | NA                                                                                         |            |                      |
| Re-sampling            | NA                                                                                         |            |                      |
| Re-weighting           | NA                                                                                         |            |                      |
| MixUp                  | mixup: Beyond empirical risk minimization                                                  |            |                      |
| Focal Loss             | Focal loss for dense object detection                                                      |            |                      |
| Classifier Re-training | Decoupling representation and classifier for long-tailed recognition                       |            |                      |
| T-Norm                 | Decoupling representation and classifier for long-tailed recognition                       |            |                      |
| LWS                    | Decoupling representation and classifier for long-tailed recognition                       |            |                      |
| KNN                    | Decoupling representation and classifier for long-tailed recognition                       |            |                      |
| CBLoss                 | Class-balanced loss based on effective number of samples                                   |            |                      |
| CBLoss_Focal           | Class-balanced loss based on effective number of samples                                   |            |                      |
| LADELoss               | Disentangling label distribution for long-tailed visual recognition                        |            |                      |
| LDAM                   | Learning imbalanced datasets with label-distribution-aware margin loss                     |            |                      |
| Logits Adjust Loss     | Long-tail learning via logit adjustment                                                    |            |                      |
| Logits Adjust Posthoc  | Long-tail learning via logit adjustment                                                    |            |                      |
| PriorCELoss            | Disentangling label distribution for long-tailed visual recognition                        |            |                      |
| RangeLoss              | Range Loss for Deep Face Recognition with Long-Tailed Training Data                        |            |                      |
| SEQLLoss               | Equalization loss for long-tailed object recognition                                       |            |                      |
| VSLoss                 | Label-imbalanced and group-sensitive classification under overparameterization             |            |                      |
| WeightedSoftmax        | Deep Long-Tailed Learning: A Survey                                                        |            |                      |
| BalancedSoftmax        | Balanced meta-softmax for long-tailed visual recognition                                   |            |                      |
| De-Confound            | Long-tailed classification by keeping the good and removing the bad momentum causal effect |            |                      |
| DisAlign               | Distribution alignment: A unified framework for long-tail visual recognition               |            |                      |
| GCL first stage        | Long-tailed visual recognition via gaussian clouded logit adjustment                       |            |                      |
| GCL second stage       | Long-tailed visual recognition via gaussian clouded logit adjustment                       |            |                      |
| MiSLAS                 | Improving calibration for long-tailed recognition                                          |            |                      |
| RSG                    | Rsg: A simple but effective module for learning imbalanced datasets                        |            |                      |
| SADE                   | Long-tailed recognition by routing diverse distribution-aware experts                      |            |                      |
| SAM                    | Sharpness-aware minimization for efficiently improving generalization                      |            |                      |
| BBN                    | Bbn: Bilateral-branch network with cumulative learning for long-tailed visual recognition  |            |                      |

| Methods                | Paper                                                                                      | Link (TBD) | Offical Codes (TBD)  |
|------------------------|--------------------------------------------------------------------------------------------|------------|----------------------|
| BYOL                   | Bootstrap your own latent-a new approach to self-supervised learning                       |            |                      |
| MOCOv2                 | Improved baselines with momentum contrastive learning                                      |            |                      |
| MAE (RetFound)         | RETFound: a foundation model for generalizable disease detection from retinal image        |            |                      |
| CAEv2 (PanDerm)        | A General-Purpose Multimodal Foundation Model for Dermatology                              |            |                      |
| DINOv2 (TBD)           | Dinov2: Learning robust visual features without supervision                                |            |                      |

### Support Backbones
| Backbones        | Paper                                                                                |
|------------------|--------------------------------------------------------------------------------------|
| ResNet           | Deep residual learning for image recognition                                         |
| ViT              | An image is worth 16x16 words: Transformers for image recognition at scale           |
| Swin Transformer | Swin transformer: Hierarchical vision transformer using shifted windows              |
| ConvNext         | A convnet for the 2020s                                                              |
| RetFound         | RETFound: a foundation model for generalizable disease detection from retinal images |
| PanDerm          | A General-Purpose Multimodal Foundation Model for Dermatology                        |

Noted: If you apply foundation models, please modify the model path and use the backbone of the foundation model as the model name, e.g., ViT for RetFound.
### Script Training
For non-MedMNIST Training, please use command like:
```bash
python main.py --config ./configs/isic/100/isic_ERM.yml
```

For MedMNIST Training, please find the `./train.sh` script for reference.

## Disclaimer

This repository is provided for research purposes only. The datasets used in this project are either publicly available under their respective licenses or referenced from external sources. Redistribution of data files included in this repository is not permitted unless explicitly allowed by the original dataset licenses.

### Data Usage
Please ensure that you comply with the licensing terms of the datasets before using them. The authors are not responsible for any misuse of the data. If you are using any dataset provided or linked in this repository, it is your responsibility to adhere to the license terms provided by the dataset creators.

For questions or concerns, please contact the repository maintainers.
# Parametric Classification for Generalized Category Discovery: A Baseline Study


<p align="center">
    <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Wen_Parametric_Classification_for_Generalized_Category_Discovery_A_Baseline_Study_ICCV_2023_paper.html"><img src="https://img.shields.io/badge/-ICCV%202023-68488b"></a>
    <a href="https://arxiv.org/abs/2211.11727"><img src="https://img.shields.io/badge/arXiv-2211.11727-b31b1b"></a>
    <a href="https://wen-xin.info/simgcd"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://github.com/CVMI-Lab/SlotCon/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</p>
<p align="center">
	Parametric Classification for Generalized Category Discovery: A Baseline Study (ICCV 2023)<br>
  By
  <a href="https://wen-xin.info">Xin Wen</a>*, 
  <a href="https://bzhao.me/">Bingchen Zhao</a>*, and 
  <a href="https://xjqi.github.io/">Xiaojuan Qi</a>.
</p>

![teaser](assets/teaser.jpg)

Generalized Category Discovery (GCD) aims to discover novel categories in unlabelled datasets using knowledge learned from labelled samples.
Previous studies argued that parametric classifiers are prone to overfitting to seen categories, and endorsed using a non-parametric classifier formed with semi-supervised $k$-means.

However, in this study, we investigate the failure of parametric classifiers, verify the effectiveness of previous design choices when high-quality supervision is available, and identify unreliable pseudo-labels as a key problem. We demonstrate that two prediction biases exist: the classifier tends to predict seen classes more often, and produces an imbalanced distribution across seen and novel categories. 
Based on these findings, we propose a simple yet effective parametric classification method that benefits from entropy regularisation, achieves state-of-the-art performance on multiple GCD benchmarks and shows strong robustness to unknown class numbers.
We hope the investigation and proposed simple framework can serve as a strong baseline to facilitate future studies in this field.

## Running

### Dependencies

```
pip install -r requirements.txt
```

### Config

Set paths to datasets and desired log directories in ```config.py```


### Datasets

We use fine-grained benchmarks in this paper, including:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

We also use generic object recognition datasets, including:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet-100/1K](https://image-net.org/download.php)


### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```

We found picking the model according to 'Old' class performance could lead to possible over-fitting, and since 'New' class labels on the held-out validation set should be assumed unavailable, we suggest not to perform model selection, and simply use the last-epoch model.

## Results
Our results:

<table><thead><tr><th>Source</th><th colspan="3">Paper (3 runs) </th><th colspan="3">Current Github (5 runs) </th></tr></thead><tbody><tr><td>Dataset</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td>CIFAR10</td><td>97.1±0.0</td><td>95.1±0.1</td><td>98.1±0.1</td><td>97.0±0.1</td><td>93.9±0.1</td><td>98.5±0.1</td></tr><tr><td>CIFAR100</td><td>80.1±0.9</td><td>81.2±0.4</td><td>77.8±2.0</td><td>79.8±0.6</td><td>81.1±0.5</td><td>77.4±2.5</td></tr><tr><td>ImageNet-100</td><td>83.0±1.2</td><td>93.1±0.2</td><td>77.9±1.9</td><td>83.6±1.4</td><td>92.4±0.1</td><td>79.1±2.2</td></tr><tr><td>ImageNet-1K</td><td>57.1±0.1</td><td>77.3±0.1</td><td>46.9±0.2</td><td>57.0±0.4</td><td>77.1±0.1</td><td>46.9±0.5</td></tr><tr><td>CUB</td><td>60.3±0.1</td><td>65.6±0.9</td><td>57.7±0.4</td><td>61.5±0.5</td><td>65.7±0.5</td><td>59.4±0.8</td></tr><tr><td>Stanford Cars</td><td>53.8±2.2</td><td>71.9±1.7</td><td>45.0±2.4</td><td>53.4±1.6</td><td>71.5±1.6</td><td>44.6±1.7</td></tr><tr><td>FGVC-Aircraft</td><td>54.2±1.9</td><td>59.1±1.2</td><td>51.8±2.3</td><td>54.3±0.7</td><td>59.4±0.4</td><td>51.7±1.2</td></tr><tr><td>Herbarium 19</td><td>44.0±0.4</td><td>58.0±0.4</td><td>36.4±0.8</td><td>44.2±0.2</td><td>57.6±0.6</td><td>37.0±0.4</td></tr></tbody></table>

## Citing this work

If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{wen2023simgcd,
    author    = {Wen, Xin and Zhao, Bingchen and Qi, Xiaojuan},
    title     = {Parametric Classification for Generalized Category Discovery: A Baseline Study},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
    pages     = {16590-16600}
}
```

## Acknowledgements

The codebase is largely built on this repo: https://github.com/sgvaze/generalized-category-discovery.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

