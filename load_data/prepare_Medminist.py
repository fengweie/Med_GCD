# +
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from load_data.data_utils import subsample_instances
from load_data.info import INFO, HOMEPAGE, DEFAULT_ROOT

class MedMNIST(Dataset):
    flag = ...
    def __init__(self,
                 split,
                 transform=None,
                 target_transform=None,
                 as_rgb=False,
                 root=DEFAULT_ROOT):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation

        '''
        self.info = INFO[self.flag]

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError("Failed to setup the default `root` directory. " +
                               "Please specify and create the `root` directory manually.")

        if not os.path.exists(
                        os.path.join(self.root, "{}.npz".format(self.flag))):
                    raise RuntimeError('Dataset not found. ' +
                                       ' You can set `download=True` to download it')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split == 'train':
            self.data = np.concatenate((npz_file['train_images'],npz_file['val_images']),axis=0)
            self.target = np.concatenate((npz_file['train_labels'],npz_file['val_labels']),axis=0)

        elif self.split == 'test':
            self.data = npz_file['test_images']
            self.target = npz_file['test_labels']
        else:
            raise ValueError
        self.target = self.target[:,0]
        values, counts = np.unique(self.target, return_counts=True)
#         print(self.target.shape)
        print(self.data.shape,values, counts)
        self.target = np.array(self.target).tolist()
  
        self.target = [int(x) for x in self.target]
        self.uq_idxs = np.array(range(len(self.data)))

    def __len__(self):
        return self.data.shape[0]
    def __repr__(self):
        '''Adapted from torchvision.ss'''
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} ({self.flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)
class MedMNIST2D(MedMNIST):
    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        '''

        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
#         print(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        idx = self.uq_idxs[index]

        return img, target, idx

    def save(self, folder, postfix="png", write_csv=True):

        from load_data.utils import save2d

        save2d(imgs=self.data,
               labels=self.target,
               img_folder=os.path.join(folder, self.flag),
               split=self.split,
               postfix=postfix,
               csv_path=os.path.join(folder, f"{self.flag}.csv") if write_csv else None)

    def montage(self, length=20, replace=False, save_folder=None):
        from load_data.utils import montage2d

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(imgs=self.data,
                                n_channels=self.info['n_channels'],
                                sel=sel)

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(os.path.join(save_folder,
                                          f"{self.flag}_{self.split}_montage.jpg"))

        return montage_img
class PathMNIST(MedMNIST2D):
    flag = "pathmnist"


class OCTMNIST(MedMNIST2D):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST2D):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST2D):
    flag = "chestmnist"


class DermaMNIST(MedMNIST2D):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST2D):
    flag = "retinamnist"


class BreastMNIST(MedMNIST2D):
    flag = "breastmnist"


class BloodMNIST(MedMNIST2D):
    flag = "bloodmnist"


class TissueMNIST(MedMNIST2D):
    flag = "tissuemnist"


class OrganAMNIST(MedMNIST2D):
    flag = "organamnist"


class OrganCMNIST(MedMNIST2D):
    flag = "organcmnist"


class OrganSMNIST(MedMNIST2D):
    flag = "organsmnist"
    
    
def subsample_dataset(dataset, idxs):

    dataset.data = dataset.data[idxs]
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cars = np.array(include_classes)     #
#     print(np.unique(dataset.target))
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_medminist_datasets(data_flag, train_transform, test_transform, as_rgb, root_path, total_class = 100, train_classes=range(50),
                           prop_train_labels=0.5, split_train_val=False, seed=0):

    info = INFO[data_flag]
    total_class = len(info['label'])

    np.random.seed(seed)
    # Init entire training set
    if info['python_class']=="PathMNIST":
        whole_training_set = PathMNIST(split='train', transform=train_transform, as_rgb=as_rgb, root=root_path)
        whole_training_set_test = PathMNIST(split='train', transform=test_transform, as_rgb=as_rgb, root=root_path)
        # Get test set for all classes
        test_dataset = PathMNIST(split='test', transform=test_transform, as_rgb=as_rgb, root=root_path)
    elif info['python_class']=="DermaMNIST":
        whole_training_set = DermaMNIST(split='train', transform=train_transform, as_rgb=as_rgb, root=root_path)
        
        whole_training_set_test = DermaMNIST(split='train', transform=test_transform, as_rgb=as_rgb, root=root_path)
        # Get test set for all classes
        test_dataset = DermaMNIST(split='test', transform=test_transform, as_rgb=as_rgb, root=root_path)
    elif info['python_class']=="BloodMNIST":
        whole_training_set = BloodMNIST(split='train', transform=train_transform, as_rgb=as_rgb, root=root_path)
        
        whole_training_set_test = BloodMNIST(split='train', transform=test_transform, as_rgb=as_rgb, root=root_path)
        test_dataset = BloodMNIST(split='test', transform=test_transform, as_rgb=as_rgb, root=root_path)
    elif info['python_class']=="TissueMNIST":
        whole_training_set = TissueMNIST(split='train', transform=train_transform, as_rgb=as_rgb, root=root_path)
        
        # Get test set for all classes
        whole_training_set_test = TissueMNIST(split='train', transform=test_transform, as_rgb=as_rgb, root=root_path)
        test_dataset = TissueMNIST(split='test', transform=test_transform, as_rgb=as_rgb, root=root_path)
    elif info['python_class']=="OrganAMNIST":
        whole_training_set = OrganAMNIST(split='train', transform=train_transform, as_rgb=as_rgb, root=root_path)
       
        whole_training_set_test = OrganAMNIST(split='train', transform=test_transform, as_rgb=as_rgb, root=root_path)
        test_dataset = OrganAMNIST(split='test', transform=test_transform, as_rgb=as_rgb, root=root_path) 
    elif info['python_class']=="OrganCMNIST":
        whole_training_set = OrganCMNIST(split='train', transform=train_transform, as_rgb=as_rgb, root=root_path)
        
        whole_training_set_test = OrganCMNIST(split='train', transform=test_transform, as_rgb=as_rgb, root=root_path)
        test_dataset = OrganCMNIST(split='test', transform=test_transform, as_rgb=as_rgb, root=root_path)     
    elif info['python_class']=="OrganSMNIST":
        whole_training_set = OrganSMNIST(split='train', transform=train_transform, as_rgb=as_rgb, root=root_path)
       
        whole_training_set_test = OrganSMNIST(split='train', transform=test_transform, as_rgb=as_rgb, root=root_path)
        test_dataset = OrganSMNIST(split='test', transform=test_transform, as_rgb=as_rgb, root=root_path) 
    
    
    # Get labelled training set which has subsampled classes, then subsample some indices from that
    known_classes = train_classes
    unknown_classes = range(len(train_classes), total_class)
    whole_known_classes_set = subsample_classes(deepcopy(whole_training_set), include_classes=known_classes)

    labeled_known_idxs = []
    unlabeled_known_idxs = []
    known_classes = np.unique(whole_known_classes_set.target)
    for cls in known_classes:
        cls_idxs = np.where(whole_known_classes_set.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False,
                              size=((int((1 - prop_train_labels) * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        labeled_known_idxs.extend(t_)
        unlabeled_known_idxs.extend(v_)
    lt_labeled_known_dataset = subsample_dataset(deepcopy(whole_known_classes_set), labeled_known_idxs)
    lt_unlabeled_known_dataset = subsample_dataset(deepcopy(whole_known_classes_set), unlabeled_known_idxs)

    lt_unlabeled_unknown_dataset = subsample_classes(deepcopy(whole_training_set),
                                                     include_classes=range(len(train_classes), total_class))

    # Either split train into train and val or use test set as val
    train_dataset_labelled = lt_labeled_known_dataset
    train_dataset_unlabelled = torch.utils.data.ConcatDataset(
        [lt_unlabeled_known_dataset, lt_unlabeled_unknown_dataset])
    val_dataset_labelled = None

    whole_known_classes_set_test = subsample_classes(deepcopy(whole_training_set_test), include_classes=known_classes)
    
    lt_unlabeled_known_dataset_test = subsample_dataset(deepcopy(whole_known_classes_set_test), unlabeled_known_idxs)

    lt_unlabeled_unknown_dataset_test = subsample_classes(deepcopy(whole_training_set_test),
                                                     include_classes=range(len(train_classes), total_class))
    train_dataset_unlabelled_test = torch.utils.data.ConcatDataset(
    [lt_unlabeled_known_dataset_test, lt_unlabeled_unknown_dataset_test])
    
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
#         'val': train_dataset_unlabelled_test,
        'test': train_dataset_unlabelled_test,
    }
    return all_datasets

if __name__ == '__main__':
    data_flag = "organamnist"
    root_path = '/mnt/workdir/fengwei/NCD/MedMNISTv2'
    x = get_medminist_datasets(data_flag, None, None, as_rgb=True, root_path=root_path, train_classes=range(4),
                           prop_train_labels=0.5, split_train_val=False, seed=0)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

#     print('Printing labelled and unlabelled overlap...')
#     print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
#     print('Printing total instances in train...')
#     print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))
    values, counts = np.unique(x["train_labelled"].target, return_counts=True)
    print('Num Labelled Classes:',values, counts)
#     values, counts = np.unique(x["train_unlabelled"].target, return_counts=True)
#     print('Num Unabelled Classes:',values, counts)
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

