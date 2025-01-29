# PRDL
PRDL:Prediction refinement and discriminative learning for generalized category discovery in medical image classification

## Introduction
Generalized category discovery (GCD) aims to use the knowledge learned from known category data to identify new categories that were not defined during the model training phase. Many methods have been proposed and have significantly improved the performance of GCD in natural images. However, no work has been done in discovering new classes based on medical images and disease categories, which is crucial for understanding and diagnosing specific diseases. In addition, existing methods tend to produce predictions that are biased toward known categories and have difficulty distinguishing between categories of medical images that have visually similar appearances. To this end, we propose a novel method called PRDL, which is based on Prediction Refinement and Discriminative Learning strategies for generalized category discovery in medical image classification. Specifically, we first propose a cross-sample knowledge-guided clustering module that utilizes the knowledge of other samples in the same batch to mitigate the prediction bias of the model, thus refining the predictions for unbiased clustering. Second, we also use the refined predictions to guide the representation learning process to learn better feature embeddings. Finally, in order to learn more discriminative features, we propose a prototype-based discriminative learning strategy that utilizes a set of learnable prototypes to enhance intra-class compactness and inter-class dispersion.
Extensive experiments on four medical image classification datasets validate the effectiveness of the proposed algorithm.

## Running

### Dependencies

```
pip install -r requirements.txt
```

### Datasets

#### Data Download
| Domain           | Dataset         | Link                                                                                   | License        |
|------------------|-----------------|----------------------------------------------------------------------------------------|----------------|
| Pathology        | NCT-CRC-HE-100K    | https://zenodo.org/records/1214456                                                                  | CC BY 4.0   |
| Radiology        | OrganAMNIST        | https://medmnist.com/                    | CC0 1.0   |
| Radiology       | OrganCMNIST     | https://medmnist.com/                                                                  | CC BY 4.0   |
| Gastroenterology | KVASIR          | https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset                             | ODbL 1.0       |

### Scripts

**Train the model**:

```
CUDA_VISIBLE_DEVICES=0 python scripts/train_GCD_MIA.py --dataset_name ${DATASET_NAME}
```


## Acknowledgements

The codebase is largely built on this repo: https://github.com/sgvaze/generalized-category-discovery.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Disclaimer

This repository is provided for research purposes only. The datasets used in this project are either publicly available under their respective licenses or referenced from external sources. Redistribution of data files included in this repository is not permitted unless explicitly allowed by the original dataset licenses.

### Data Usage
Please ensure that you comply with the licensing terms of the datasets before using them. The authors are not responsible for any misuse of the data. If you are using any dataset provided or linked in this repository, it is your responsibility to adhere to the license terms provided by the dataset creators.


