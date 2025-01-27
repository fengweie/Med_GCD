# +
from load_data.data_utils import MergedDataset
from load_data.info import INFO, HOMEPAGE, DEFAULT_ROOT
from load_data.prepare_Medminist import get_medminist_datasets

from copy import deepcopy
import pickle
import os


get_dataset_funcs = {
    'pathmnist': get_medminist_datasets,
    "organcmnist": get_medminist_datasets,
    "organamnist": get_medminist_datasets,
    'tissuemnist': get_medminist_datasets,
    "bloodmnist": get_medminist_datasets,
    "dermamnist": get_medminist_datasets,
    "organsmnist": get_medminist_datasets,
}

def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError
    root_path = '/mnt/sdc/fengwei/GCD_medical/MedMNISTv2/'
    # Get datasets
    args = get_class_splits(args)
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(data_flag=dataset_name, train_transform=train_transform, test_transform=test_transform,
                           as_rgb=True, root_path=root_path, train_classes=args.train_classes,
                           prop_train_labels=args.prop_train_labels, split_train_val=False, seed=0)


    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
#     unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
#     unlabelled_train_examples_test = deepcopy(datasets['val'])
#     unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, args, datasets

# def get_datasets_v2(dataset_name, train_transform, test_transform, args):

#     """
#     :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
#              test_dataset,
#              unlabelled_train_examples_test,
#              datasets
#     """

#     #
#     if dataset_name not in get_dataset_funcs.keys():
#         raise ValueError
#     # Get datasets
#     args = get_class_splits(args)
#     get_dataset_f = get_dataset_funcs[dataset_name]
#     datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform)
#     # Set target transforms:
#     target_transform_dict = {}
#     for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
      
#         target_transform_dict[cls] = i
#     target_transform = lambda x: target_transform_dict[x]

#     for dataset_name, dataset in datasets.items():
#         if dataset is not None:
#             dataset.target_transform = target_transform

#     # Train split (labelled and unlabelled classes) for training
#     train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
#                                   unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

#     test_dataset = datasets['test']
# #     unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
#     unlabelled_train_examples_test = deepcopy(datasets['val'])
#     unlabelled_train_examples_test.transform = test_transform

#     return train_dataset, test_dataset, unlabelled_train_examples_test, datasets
def get_class_splits(args):

    if args.dataset_name == 'pathmnist' or args.dataset_name == "organcmnist" or args.dataset_name == "organamnist" or args.dataset_name == "organsmnist":
        info = INFO[args.dataset_name]
        n_channels = info['n_channels']
        args.total_classes = len(info['label'])
        args.image_size = 28
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, args.total_classes)
    elif args.dataset_name == 'tissuemnist' or args.dataset_name == "bloodmnist" or args.dataset_name == "dermamnist":
        info = INFO[args.dataset_name]
        n_channels = info['n_channels']
        args.total_classes = len(info['label'])
        
        args.image_size = 28
        args.train_classes = range(4)
        args.unlabeled_classes = range(4, args.total_classes)
    elif args.dataset_name == 'endo':

        args.image_size = 224
        args.total_classes = 8

        args.train_classes = range(5)
        args.unlabeled_classes = range(5, args.total_classes)      
    else:
        raise NotImplementedError

    return args
