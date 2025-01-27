from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import Sampler
from glob import glob
import numpy as np
import os
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
from collections import Counter
def load_CIFAR_batch(task,use_seed):
    output_dir = "/mnt/workdir/fengwei/NCD/ISIC_split"
    np.random.seed(use_seed)
    print("seed")
    if task == 0:
        AK_path_npy = os.path.join(output_dir, "AK_data" + ".npy")
        AK = np.load(AK_path_npy)
        AK_num = len(AK)
        AK_label =  np.zeros([AK_num])

        MEL_path_npy = os.path.join(output_dir, "MEL_data" + ".npy")
        MEL = np.load(MEL_path_npy)
        # MEL = MEL[:500]
        subsample_indices = np.random.choice(range(len(MEL)), replace=False,
                                             size=(500),)
        MEL = MEL[subsample_indices]
        MEL_num = len(MEL)
        MEL_label = 1 * np.ones([MEL_num])

        NV_path_npy = os.path.join(output_dir, "NV_data" + ".npy")
        NV = np.load(NV_path_npy)
        # NV = NV[:500]
        subsample_indices = np.random.choice(range(len(NV)), replace=False,
                                             size=(500),)
        NV = NV[subsample_indices]
        NV_num = len(NV)
        NV_label = 2 * np.ones([NV_num])

        BCC_path_npy = os.path.join(output_dir, "BCC_data" + ".npy")
        BCC = np.load(BCC_path_npy)
        # BCC = BCC[:500]
        subsample_indices = np.random.choice(range(len(BCC)), replace=False,
                                             size=(500),)
        BCC = BCC[subsample_indices]

        BCC_num = len(BCC)
        BCC_label = 3 * np.ones([BCC_num])

        BKL_path_npy = os.path.join(output_dir, "BKL_data" + ".npy")
        BKL = np.load(BKL_path_npy)
        # BKL = BKL[:500]
        subsample_indices = np.random.choice(range(len(BKL)), replace=False,
                                             size=(500),)
        BKL = BKL[subsample_indices]
        BKL_num = len(BKL)
        BKL_label = 4 * np.ones([BKL_num])

        SCC_path_npy = os.path.join(output_dir, "SCC_data" + ".npy")
        SCC = np.load(SCC_path_npy)
        SCC_num = len(SCC)
        SCC_label = 5 * np.ones([SCC_num])

        DF_path_npy = os.path.join(output_dir, "DF_data" + ".npy")
        DF = np.load(DF_path_npy)
        DF_num = len(DF)
        DF_label = 6 * np.ones([DF_num])

        VASC_path_npy = os.path.join(output_dir, "VASC_data" + ".npy")
        VASC = np.load( VASC_path_npy)
        VASC_num = len(VASC)
        VASC_label = 7 * np.ones([VASC_num])

        class_path = np.concatenate((BCC, DF, NV, VASC, BKL, SCC, MEL, AK,
                                          ), axis=0)
        class_label = np.concatenate(
            (BCC_label, DF_label, NV_label, VASC_label, BKL_label, SCC_label, MEL_label, AK_label
             ), axis=0)
            # X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
            # X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    elif task ==1:
        AK_path_npy = os.path.join(output_dir, "AK_data" + ".npy")
        AK = np.load(AK_path_npy)
        AK_num = len(AK)
        AK_label = 4 * np.ones([AK_num])

        MEL_path_npy = os.path.join(output_dir, "MEL_data" + ".npy")
        MEL = np.load(MEL_path_npy)
        # MEL = MEL[:500]
        subsample_indices = np.random.choice(range(len(MEL)), replace=False,
                                             size=(500),)
        MEL = MEL[subsample_indices]
        MEL_num = len(MEL)
        MEL_label = 5 * np.ones([MEL_num])

        NV_path_npy = os.path.join(output_dir, "NV_data" + ".npy")
        NV = np.load(NV_path_npy)
        # NV = NV[:500]
        subsample_indices = np.random.choice(range(len(NV)), replace=False,
                                             size=(500),)
        NV = NV[subsample_indices]
        NV_num = len(NV)
        NV_label = 6 * np.ones([NV_num])

        BCC_path_npy = os.path.join(output_dir, "BCC_data" + ".npy")
        BCC = np.load(BCC_path_npy)
        # BCC = BCC[:500]
        subsample_indices = np.random.choice(range(len(BCC)), replace=False,
                                             size=(500),)
        BCC = BCC[subsample_indices]
        BCC_num = len(BCC)
        BCC_label = 7 * np.ones([BCC_num])

        BKL_path_npy = os.path.join(output_dir, "BKL_data" + ".npy")
        BKL = np.load(BKL_path_npy)
        # BKL = BKL[:500]
        subsample_indices = np.random.choice(range(len(BKL)), replace=False,
                                             size=(500),)
        BKL = BKL[subsample_indices]
        BKL_num = len(BKL)
        BKL_label = np.zeros([BKL_num])

        SCC_path_npy = os.path.join(output_dir, "SCC_data" + ".npy")
        SCC = np.load(SCC_path_npy)
        SCC_num = len(SCC)
        SCC_label = 1 * np.ones([SCC_num])

        DF_path_npy = os.path.join(output_dir, "DF_data" + ".npy")
        DF = np.load(DF_path_npy)
        DF_num = len(DF)
        DF_label = 2 * np.ones([DF_num])

        VASC_path_npy = os.path.join(output_dir, "VASC_data" + ".npy")
        VASC = np.load(VASC_path_npy)
        VASC_num = len(VASC)
        VASC_label = 3 * np.ones([VASC_num])
        class_path = np.concatenate((BCC, DF, NV, VASC, BKL, SCC, MEL, AK,
                                     ), axis=0)
        class_label = np.concatenate(
            (BCC_label, DF_label, NV_label, VASC_label, BKL_label, SCC_label, MEL_label, AK_label
             ), axis=0)

    return class_path, class_label

# print(len(BCC), len(DF), len(NV), len(VASC), len(BKL), len(SCC), len(MEL), len(AK))
# 3323 239 12875 253 2624 628 4522 867

def load_ISIC(task,seen_label_list,unseen_label_list,labeled_ratio,use_seed):
    train_set, train_labels = load_CIFAR_batch(task=task,use_seed=use_seed)

    seen_ind = [i for i in range(len(train_labels)) if train_labels[i] in seen_label_list]
    unseen_ind = [i for i in range(len(train_labels)) if train_labels[i] in unseen_label_list]

    seen_data = train_set[seen_ind]
    train_labels = np.array(train_labels)
    seen_targets = train_labels[seen_ind].tolist()

    unseen_data = train_set[unseen_ind]
    unseen_targets = train_labels[unseen_ind].tolist()

    seen_class_lebeled_path,  meah_path, seen_class_lebeled_label,meah_label = \
        train_test_split(seen_data, seen_targets, test_size=labeled_ratio, random_state=0, stratify=seen_targets)
    seen_class_unlebeled_path,  test_seen_class_path, seen_class_unlebeled_label,test_seen_class_label = \
        train_test_split(meah_path, meah_label, test_size=0.4, random_state=0, stratify=meah_label)
    unseen_class_unlebeled_path,  test_unseen_class_path, unseen_class_unlebeled_label, test_unseen_class_label = \
        train_test_split(unseen_data, unseen_targets, test_size=0.2, random_state=0, stratify=unseen_targets)
    return seen_class_lebeled_path, seen_class_lebeled_label,\
           seen_class_unlebeled_path, seen_class_unlebeled_label, \
           test_seen_class_path, test_seen_class_label,\
           unseen_class_unlebeled_path, unseen_class_unlebeled_label, \
           test_unseen_class_path, test_unseen_class_label

class ISIC_loader(data.Dataset):

    def __init__(self, data_path, train_labels,
                 transform=None, target_transform=None):
        self.data = data_path
        self.targets  = train_labels
        self.transform = transform
        self.target_transform = target_transform
        # print(Counter(self.targets))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        return len(self.data)

def CIFAR10Loader(root, task, batch_size, resample=False, num_workers=2,  aug=None, shuffle=True, target_list=range(4)):
    if aug==None:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    elif aug=='once':
        transform = transforms.Compose([
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomCrop((224, 224)),
            # RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]))
    dataset = CIFAR10(root=root,task = task, transform=transform, target_list=target_list)

    if resample:
        class_sample_count = np.array(
            [len(np.where(dataset.targets == t)[0]) for t in np.unique(dataset.targets)])
        weight = 1. / class_sample_count
        print(class_sample_count, weight)
        samples_weight = np.array([weight[int(t)] for t in dataset.targets])
        samples_weight = torch.from_numpy(samples_weight).cuda()
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        loader = data.DataLoader(dataset, batch_size=batch_size,
                                 # shuffle=shuffle,
                                 sampler=sampler,
                                 num_workers=num_workers)
    else:
        loader = data.DataLoader(dataset, batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers)
    return loader
# def get_ISIC(labeled_ratio):
#     # BCC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/BCC/*.jpg"))
#     # BCC_num = len(BCC)
#     # BCC_label = np.zeros([BCC_num])
#     # BKL = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/BKL/*.jpg"))
#     # BKL_num = len(BKL)
#     # BKL_label = 1*np.ones([BKL_num])
#     # NV = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/NV/*.jpg"))
#     # NV_num = len(NV)
#     # NV_label = 2*np.ones([NV_num])
#     # MEL = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/MEL/*.jpg"))
#     # MEL_num = len(MEL)
#     # MEL_label = 3*np.ones([MEL_num])
#     #
#     # DF = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/DF/*.jpg"))
#     # DF_num = len(DF)
#     # DF_label = 4*np.ones([DF_num])
#     # SCC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/SCC/*.jpg"))
#     # SCC_num = len(SCC)
#     # SCC_label = 5*np.ones([SCC_num])
#     # VASC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/VASC/*.jpg"))
#     # VASC_num = len(VASC)
#     # VASC_label = 6*np.ones([VASC_num])
#     # AK = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/AK/*.jpg"))
#     # AK_num = len(AK)
#     # AK_label = 7*np.ones([AK_num])
#     #
#     # seen_class_path = np.concatenate((BCC, BKL, NV, MEL,
#     #                            ), axis=0)
#     # seen_class_label = np.concatenate((BCC_label, BKL_label, NV_label, MEL_label,
#     #                             ), axis=0)
#     # unseen_class_path = np.concatenate((DF, SCC, VASC, AK,
#     #                            ), axis=0)
#     # unseen_class_label = np.concatenate((DF_label, SCC_label, VASC_label, AK_label,
#     #                             ), axis=0)
#
#     output_dir = "/mnt/workdir/fengwei/NCD/ISIC_split"
#     seen_data_npy = os.path.join(output_dir, "seen_class_data" + ".npy")
#     seen_class_path = np.load(seen_data_npy)
#     seen_label_npy = os.path.join(output_dir, "seen_class_label" + ".npy")
#     seen_class_label = np.load(seen_label_npy)
#
#
#     unseen_data_npy = os.path.join(output_dir, "unseen_class_data" + ".npy")
#     unseen_class_path = np.load(unseen_data_npy)
#     unseen_label_npy = os.path.join(output_dir, "unseen_class_label" + ".npy")
#     unseen_class_label = np.load(unseen_label_npy)
#
#     seen_class_lebeled_path,  meah_path, seen_class_lebeled_label,meah_label = \
#         train_test_split(seen_class_path, seen_class_label, test_size=labeled_ratio, random_state=0, stratify=seen_class_label)
#     seen_class_unlebeled_path,  test_seen_class_path, seen_class_unlebeled_label,test_seen_class_label = \
#         train_test_split(meah_path, meah_label, test_size=0.4, random_state=0, stratify=meah_label)
#     unseen_class_unlebeled_path,  test_unseen_class_path, unseen_class_unlebeled_label, test_unseen_class_label = \
#         train_test_split(unseen_class_path, unseen_class_label, test_size=0.2, random_state=0, stratify=unseen_class_label)
#     return seen_class_lebeled_path, seen_class_lebeled_label,\
#            seen_class_unlebeled_path, seen_class_unlebeled_label, \
#            test_seen_class_path, test_seen_class_label,\
#            unseen_class_unlebeled_path, unseen_class_unlebeled_label, \
#            test_unseen_class_path, test_unseen_class_label
#     # seen_class_lebeled_path,  seen_class_unlebeled_path, seen_class_lebeled_label,seen_class_unlebeled_label = \
#     #     train_test_split(seen_class_path, seen_class_label, test_size=labeled_ratio, random_state=0, stratify=seen_class_label)
#
#     # return seen_class_lebeled_path, seen_class_lebeled_label,\
#     #        seen_class_unlebeled_path, seen_class_unlebeled_label,\
#     #        unseen_class_path,unseen_class_label
#            # test_seen_class_path, test_seen_class_label,\
#            # unseen_class_unlebeled_path, unseen_class_unlebeled_label, \
#            # test_unseen_class_path, test_unseen_class_label
def get_Kvasir(labeled_ratio):
    BCC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/BCC/*.jpg"))
    BCC_num = len(BCC)
    BCC_label = np.zeros([BCC_num])
    BKL = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/BKL/*.jpg"))
    BKL_num = len(BKL)
    BKL_label = 1*np.ones([BKL_num])
    NV = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/NV/*.jpg"))
    NV_num = len(NV)
    NV_label = 2*np.ones([NV_num])
    MEL = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/MEL/*.jpg"))
    MEL_num = len(MEL)
    MEL_label = 3*np.ones([MEL_num])

    DF = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/DF/*.jpg"))
    DF_num = len(DF)
    DF_label = 4*np.ones([DF_num])
    SCC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/SCC/*.jpg"))
    SCC_num = len(SCC)
    SCC_label = 5*np.ones([SCC_num])
    VASC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/VASC/*.jpg"))
    VASC_num = len(VASC)
    VASC_label = 6*np.ones([VASC_num])
    AK = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/AK/*.jpg"))
    AK_num = len(AK)
    AK_label = 7*np.ones([AK_num])

    seen_class_path = np.concatenate((BCC, BKL, NV, MEL,
                               ), axis=0)
    seen_class_label = np.concatenate((BCC_label, BKL_label, NV_label, MEL_label,
                                ), axis=0)
    unseen_class_path = np.concatenate((DF, SCC, VASC, AK,
                               ), axis=0)
    unseen_class_label = np.concatenate((DF_label, SCC_label, VASC_label, AK_label,
                                ), axis=0)
    seen_class_lebeled_path, seen_class_lebeled_label, meah_path, meah_label = \
        train_test_split(seen_class_path, seen_class_label, test_size=labeled_ratio, random_state=0, stratify=seen_class_label)
    seen_class_unlebeled_path, seen_class_unlebeled_label, test_seen_class_path, test_seen_class_label = \
        train_test_split(meah_path, meah_label, test_size=0.6, random_state=0, stratify=seen_class_label)
    unseen_class_unlebeled_path, unseen_class_unlebeled_label, test_unseen_class_path, test_unseen_class_label = \
        train_test_split(unseen_class_path, unseen_class_label, test_size=0.8, random_state=0, stratify=unseen_class_label)
    return seen_class_lebeled_path, seen_class_lebeled_label,\
           seen_class_unlebeled_path, seen_class_unlebeled_label, \
           test_seen_class_path, test_seen_class_label,\
           unseen_class_unlebeled_path, unseen_class_unlebeled_label, \
           test_unseen_class_path, test_unseen_class_label