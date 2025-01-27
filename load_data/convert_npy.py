import argparse
import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

from collections import Counter
from sklearn.model_selection import train_test_split
import math

AK = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/AK/*.jpg"))

# AK = AK[:500]
# AK_num = len(AK)
# AK_label =  np.zeros([AK_num])

DF = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/DF/*.jpg"))
# DF_num = len(DF)
# DF_label = 1 * np.ones([DF_num])


NV = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/NV/*.jpg"))
# NV = NV[:500]
# NV_num = len(NV)
# NV_label = 2 * np.ones([NV_num])


VASC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/VASC/*.jpg"))
# VASC_num = len(VASC)
# VASC_label = 3 * np.ones([VASC_num])

BKL = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/BKL/*.jpg"))
# BKL = BKL[:500]
# BKL_num = len(BKL)
# BKL_label = 4 * np.ones([BKL_num])
SCC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/SCC/*.jpg"))
# SCC_num = len(SCC)
# SCC_label = 5 * np.ones([SCC_num])

MEL = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/MEL/*.jpg"))
# MEL = MEL[:500]
# MEL_num = len(MEL)
# MEL_label = 6 * np.ones([MEL_num])


BCC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/BCC/*.jpg"))
# BCC = BCC[:500]
# BCC_num = len(BCC)
# BCC_label =7 * np.ones([BCC_num])
# print(len(BCC), len(DF), len(NV), len(VASC), len(BKL), len(SCC), len(MEL), len(AK))
# 3323 239 12875 253 2624 628 4522 867

seen_class_path = np.concatenate((BCC, DF, NV, VASC,
                                  ), axis=0)
seen_class_label = np.concatenate((np.zeros([len(BCC)]), 1 * np.ones([len(DF)]), 2 * np.ones([len(NV)]), 3 * np.ones([len(VASC)]),
                                   ), axis=0)
unseen_class_path = np.concatenate((BKL, SCC, MEL, AK,
                                    ), axis=0)
unseen_class_label = np.concatenate((4 * np.ones([len(BKL)]), 5 * np.ones([len(SCC)]), 6 * np.ones([len(MEL)]), 7 * np.ones([len(AK)]),
                                     ), axis=0)

print('Counter(seen_class_label)\n',Counter(seen_class_label))

print('Counter(unseen_class_label)\n',Counter(unseen_class_label))
output_dir = '/mnt/workdir/fengwei/NCD/ISIC_split'
def _ham_save_data(output_dir, fname, data):
    f = os.path.join(output_dir, fname + ".npy")
    np.save(f, data)
# w = 256
# h = 192
path_datas = []
img_datas = []
label_datas = []
for idx in range(len(seen_class_path)):
    path_datas.append(seen_class_path[idx])
    label_datas.append(seen_class_label[idx])
    img_datas.append(np.asarray(Image.open(seen_class_path[idx]).convert('RGB')))
# print(idatas)
# xdatas = _ham_images_load(idatas, w, h)
xdatas = np.stack(img_datas, 0)
ydatas = np.stack(label_datas, 0)
# xdatas = xdatas.astype("float32")
sample = {'image':xdatas,'label':ydatas}
ilen = len(ydatas)
_ham_save_data(output_dir, "seen_class_data" , xdatas)

_ham_save_data(output_dir, "seen_class_label" , ydatas)

# AK = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/AK/*.jpg"))
# DF = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/DF/*.jpg"))
# NV = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/NV/*.jpg"))
# VASC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/VASC/*.jpg"))
# BKL = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/BKL/*.jpg"))
# SCC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/SCC/*.jpg"))
# MEL = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/MEL/*.jpg"))
# BCC = sorted(glob("/mnt/workdir/fengwei/NCD/ISIC_split/BCC/*.jpg"))
#
# img_datas = []
# for idx in range(len(BCC)):
#     img = Image.open(BCC[idx]).convert('RGB')
#     img = img.resize((256, 256), Image.BICUBIC)
#     img_datas.append(np.asarray(img))
#
# xdatas = np.stack(img_datas, 0)
# print(xdatas.shape)
# output_dir = "/mnt/workdir/fengwei/NCD/ISIC_split"
# f = os.path.join(output_dir, "BCC_data" + ".npy")
# np.save(f, xdatas)
