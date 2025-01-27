# from __future__ import print_function
# from PIL import Image
# import os
# import os.path
# import sys
#
# if sys.version_info[0] == 2:
#     import cPickle as pickle
# else:
#     import pickle
# import torchvision
# import torch
# import torch.utils.data as data
# from torchvision import transforms
# import itertools
# import numpy as np
# from torch.utils.data import Dataset
#
# class load_dataset(Dataset):
#
#     def __init__(self, data_path, label,
#                  transform=None):
#         super(load_dataset, self).__init__()
#
#         self.images = data_path
#         # print(label)
#         self.labels = label.astype(int)[:,np.newaxis]
#         # print(label.shape)
#
#         self.transform = transform
#
#         print('Total # images:{}, labels:{}'.format(len(self.images),len(self.labels)))
#     def __getitem__(self, index):
#         """
#         Args:
#             index: the index of item
#         Returns:
#             image and its labels
#         """
#         # items = self.images[index].split('.')
#         # #study = items[2] + '/' + items[3]
#         # image_name = self.images[index]
#         # image = Image.open(image_name).convert('RGB')
#         image = self.images[index]
#         label = self.labels[index]
#         image = Image.fromarray(image.astype('uint8')).convert('RGB')
#         # print(image.size)
#         # print(label.shape)
#         if self.transform is not None:
#             image = self.transform(image)
#         print(image.shape,label)
#         return index, image, torch.FloatTensor(label)
#
#     def __len__(self):
#         return len(self.images)
#
#
# # Dictionary of transforms
# normalize = transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
# dict_transform = {
#     'ISIC_train': transforms.Compose([
#         transforms.Resize((224, 224)),
#                                     # transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
#                                     transforms.RandomHorizontalFlip(),
#                                     # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#                                     # transforms.RandomRotation(10),
#                                     # transforms.RandomResizedCrop(224),
#                                     transforms.ToTensor(),
#                                     # normalize
#     ]),
#     'ISIC_test': transforms.Compose([
#         transforms.Resize((224,224)),
#                                                 transforms.ToTensor(),
#                                                 # normalize
#     ])
# }

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
from collections import Counter
from PIL import Image
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
import torch.utils.data as data
from torchvision import transforms
import itertools
import torch
from torch.utils.data.sampler import Sampler
from glob import glob
# 加载cifar10的数据
def load_CIFAR_batch(filename):
    output_dir = "/mnt/workdir/fengwei/NCD/ISIC_split"

    AK_path_npy = os.path.join(output_dir, "AK_data" + ".npy")
    AK = np.load(AK_path_npy)
    AK_num = len(AK)
    AK_label =  np.zeros([AK_num])

    DF_path_npy = os.path.join(output_dir, "DF_data" + ".npy")
    DF = np.load(DF_path_npy)
    DF_num = len(DF)
    DF_label = 1 * np.ones([DF_num])

    NV_path_npy = os.path.join(output_dir, "NV_data" + ".npy")
    NV = np.load(NV_path_npy)
    NV_num = len(NV)
    NV_label = 2 * np.ones([NV_num])

    VASC_path_npy = os.path.join(output_dir, "VASC_data" + ".npy")
    VASC = np.load( VASC_path_npy)
    VASC_num = len(VASC)
    VASC_label = 3 * np.ones([VASC_num])

    BKL_path_npy = os.path.join(output_dir, "BKL_data" + ".npy")
    BKL = np.load(BKL_path_npy)
    BKL_num = len(BKL)
    BKL_label = 4 * np.ones([BKL_num])

    SCC_path_npy = os.path.join(output_dir, "SCC_data" + ".npy")
    SCC = np.load(SCC_path_npy)
    SCC_num = len(SCC)
    SCC_label = 5 * np.ones([SCC_num])

    MEL_path_npy = os.path.join(output_dir, "MEL_data" + ".npy")
    MEL = np.load(MEL_path_npy)
    MEL_num = len(MEL)
    MEL_label = 6 * np.ones([MEL_num])

    BCC_path_npy = os.path.join(output_dir, "BCC_data" + ".npy")
    BCC = np.load(BCC_path_npy)
    BCC_num = len(BCC)
    BCC_label =7 * np.ones([BCC_num])
    class_path = np.concatenate((BCC, DF, NV, VASC, BKL, SCC, MEL, AK,
                                      ), axis=0)
    class_label = np.concatenate(
        (BCC_label, DF_label, NV_label, VASC_label, BKL_label, SCC_label, MEL_label, AK_label
         ), axis=0)
        # X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
        # X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

    return class_path, class_label

# print(len(BCC), len(DF), len(NV), len(VASC), len(BKL), len(SCC), len(MEL), len(AK))
# 3323 239 12875 253 2624 628 4522 867

def load_ISIC(filename):
    X, Y = load_CIFAR_batch(filename)
    # print(X.shape,Y.shape)
    # xs = np.vstack(xs).reshape(-1, 3, 32, 32)
    # xs = xs.transpose((0, 2, 3, 1))  # convert to HWC
    # # Xtrain = np.concatenate(xs)  # 使变成行向量
    # # Ytrain = np.concatenate(ys)
    return X, Y


class OPENWORLDCIFAR10(data.Dataset):
    def __init__(self, root, labeled=True, labeled_num=4, labeled_ratio=0.5, rand_number=0, transform=None,
                  unlabeled_idxs=None):
        self.root = root

        self.transform = transform
        train_set, train_labels = load_ISIC(self.root)

        self.data = train_set
        self.targets = train_labels
        # print(Counter(self.targets))
        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)

        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
            print(Counter(self.targets))
        else:
            self.shrink_data(unlabeled_idxs)
            print(Counter(self.targets))

        self.targets = np.asarray(self.targets)[:, np.newaxis].astype(int)

    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]

    def __len__(self):

        return len(self.targets)

    def __getitem__(self, index):
        # print(np.asarray(self.targets).shape)

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape,target.shape)
        return index, img, target

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
dict_transform = {
    'ISIC_train': transforms.Compose([
        transforms.Resize((224, 224)),
                                    # transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                    transforms.RandomHorizontalFlip(),
                                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                    # transforms.RandomRotation(10),
                                    # transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    normalize
    ]),
    'ISIC_test': transforms.Compose([
        transforms.Resize((224,224)),
                                                transforms.ToTensor(),
                                                normalize
    ])
}