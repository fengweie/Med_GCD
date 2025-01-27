

# +
import os
import os.path as osp
from PIL import Image
from torchvision.transforms import Resize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import cv2
import glob
from pathlib import Path
import multiprocessing
from multiprocessing.dummy import Pool
import concurrent.futures
import glob

import numpy
from PIL import Image, ImageDraw
import logging
import time
import torch
import torchvision
# import pytorch_lightning as pl


from torch.utils.data import Dataset
import cv2
import pydicom
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

# from transformers import AutoTokenizer, AutoModel
# def findAllFile(base):
#     for root, ds, fs in os.walk(base):
#         for f in fs:
#             if f.endswith('.jpg'):
#                 fullname = os.path.join(root, f)
#                 yield fullname
# base = '/mnt/workdir/fengwei/NCD/MGCD/endo/kvasir-dataset-v2/'
# width = 256
# height = 256
# for idx in findAllFile(base):
#     print(idx)
#     image_name = idx 
#     src = cv2.imread(idx, cv2.IMREAD_COLOR)
#     dst = cv2.resize(src, dsize=(width, height))
#     cv2.imwrite(image_name,dst)


class endo_dataset(Dataset):
    def __init__(self, img_lst, label_list, transform):
        self.data = img_lst
        self.target = label_list
        self.preprocess = transform

        self.labels_index = ['normal-z-line',  'normal-pylorus','normal-cecum','dyed-lifted-polyps', "dyed-resection-margins",
                        'esophagitis','polyps','ulcerative-colitis']
            
#         for index, row in tqdm(df.iterrows()):
#             img_path = row['img_path']
#             x = cv2.imread(str(img_path), 0)
#             img = Image.fromarray(x).convert('RGB')
#             if self.preprocess is not None:
#                 img = self.preprocess(img)
#             self.images.append(img)

#             # labeled classes
#             if row['label'] == 'normal-z-line':
#                 label = torch.tensor(labels_index.index('normal-z-line'))

#             elif row['label'] == 'normal-pylorus':
#                 label = torch.tensor(labels_index.index('normal-pylorus'))
#             elif row['label'] == 'normal-cecum':
#                 label = torch.tensor(labels_index.index('normal-cecum'))
#             elif row['label'] == 'dyed-lifted-polyps':     #unlabeled class 1
#                 label = torch.tensor(labels_index.index('dyed-lifted-polyps'))
#             elif row['label'] == 'dyed-resection-margins':
#                 label = torch.tensor(labels_index.index('dyed-resection-margins'))
#             elif row['label'] == 'esophagitis':
#                 label = torch.tensor(labels_index.index('esophagitis'))
#             elif row['label'] == 'polyps':
#                 label = torch.tensor(labels_index.index('polyps'))
#             elif row['label'] == 'ulcerative-colitis':      #unlabeled class 2
#                 label = torch.tensor(labels_index.index('ulcerative-colitis'))
           
#             self.labels.append(label)


    def __getitem__(self, item):
        img_path = self.data[item]
        x = cv2.imread(str(img_path), 0)
        img = Image.fromarray(x).convert('RGB')
        if self.preprocess is not None:
            img = self.preprocess(img)   
            
        # labeled classes
        label = self.target[item]
#         label = torch.tensor(label_id)
            
        return img, label, item

    def __len__(self):
        return len(self.data)


def get_endo_datasets(train_transform, test_transform):
    img_path_all = glob.glob('/mnt/workdir/fengwei/NCD/MGCD/endo/kvasir-dataset-v2/*/*.jpg')

    data_merge=pd.DataFrame({"img_path":img_path_all})

    def get_label_id(x):
        return x["img_path"].split("/")[-2]
    def get_label(x):
        labels_index = ['normal-z-line',  'normal-pylorus','normal-cecum','dyed-lifted-polyps', "dyed-resection-margins",
                        'esophagitis','polyps','ulcerative-colitis']
        label_id = x["img_path"].split("/")[-2]
        if label_id == 'normal-z-line':
            label = labels_index.index('normal-z-line')

        elif label_id == 'normal-pylorus':
            label = labels_index.index('normal-pylorus')
        elif label_id == 'normal-cecum':
            label = labels_index.index('normal-cecum')
        elif label_id == 'dyed-lifted-polyps':     #unlabeled class 1
            label = labels_index.index('dyed-lifted-polyps')
        elif label_id == 'dyed-resection-margins':
            label = labels_index.index('dyed-resection-margins')
        elif label_id == 'esophagitis':
            label = labels_index.index('esophagitis')
        elif label_id == 'polyps':
            label = labels_index.index('polyps')
        elif label_id == 'ulcerative-colitis':      #unlabeled class 2
            label = labels_index.index('ulcerative-colitis') 
        return label
    data_merge['label_id'] = data_merge.apply(get_label_id, axis=1)
    data_merge['label'] = data_merge.apply(get_label, axis=1)   

#     print(data_merge["label"].value_counts())
#     print(data_merge["label_id"].value_counts())

    # normal-pylorus            1000
    # polyps                    1000
    # esophagitis               1000
    # dyed-resection-margins    1000
    # dyed-lifted-polyps        1000
    # normal-z-line             1000
    # ulcerative-colitis        1000
    # normal-cecum              1000
    labeled_class = ['normal-z-line',  'normal-pylorus','normal-cecum','dyed-lifted-polyps', "dyed-resection-margins"]

    unlabeled_class_set_1 =  ['esophagitis','polyps','ulcerative-colitis']

    train_labeled_old_df = pd.DataFrame(columns=data_merge.columns)
    train_unlabeled_old_df = pd.DataFrame(columns=data_merge.columns)
    train_unlabeled_new_df_set_1 = pd.DataFrame(columns=data_merge.columns)

    num_train_labeled_old = [500,500,500,500,500]
    num_train_unlabeled_old = [500,500,500,500,500]
    num_train_unlabeled_new_set_1 = [1000,1000,1000]

    for i in range(len(labeled_class)):
        label_one = labeled_class[i]
        num_1,num_2 = num_train_labeled_old[i],num_train_unlabeled_old[i]
        data_one_label = data_merge[data_merge['label_id']==label_one]
        n = len(data_one_label)
        print(label_one, n)    

        train_labeled_old_df = pd.concat([train_labeled_old_df, data_one_label[: num_1]], ignore_index=True)   
        train_unlabeled_old_df = pd.concat([train_unlabeled_old_df, data_one_label[num_1:(num_1+num_2)]], ignore_index=True)

    for i in range(len(unlabeled_class_set_1)):
        label_one = unlabeled_class_set_1[i]
        num_1 = num_train_unlabeled_new_set_1[i]
        data_one_label = data_merge[data_merge['label_id']==label_one]
        n = len(data_one_label)
        print(label_one, n)    

        train_unlabeled_new_df_set_1 = pd.concat([train_unlabeled_new_df_set_1, data_one_label[: num_1]], ignore_index=True)  


    print(len(train_labeled_old_df),len(train_unlabeled_old_df),len(train_unlabeled_new_df_set_1))
    ######################## dataloader
    lt_labeled_known_dataset = endo_dataset(img_lst= train_labeled_old_df['img_path'].values.tolist(), 
                                            label_list=train_labeled_old_df['label'].values.tolist(), transform=train_transform)
    lt_unlabeled_known_dataset = endo_dataset(img_lst= train_unlabeled_old_df['img_path'].values.tolist(), 
                                            label_list= train_unlabeled_old_df['label'].values.tolist(), transform=train_transform)
  

    lt_unlabeled_unknown_dataset = endo_dataset(img_lst= train_unlabeled_new_df_set_1['img_path'].values.tolist(), 
                                            label_list= train_unlabeled_new_df_set_1['label'].values.tolist(), transform=train_transform)


    # Either split train into train and val or use test set as val
    train_dataset_labelled = lt_labeled_known_dataset
    train_dataset_unlabelled = torch.utils.data.ConcatDataset(
        [lt_unlabeled_known_dataset, lt_unlabeled_unknown_dataset])
    val_dataset_labelled = None
    
    lt_unlabeled_known_dataset_test = endo_dataset(img_lst= train_unlabeled_old_df['img_path'].values.tolist(), 
                                            label_list= train_unlabeled_old_df['label'].values.tolist(), transform=test_transform)


    lt_unlabeled_unknown_dataset_test = endo_dataset(img_lst= train_unlabeled_new_df_set_1['img_path'].values.tolist(), 
                                            label_list= train_unlabeled_new_df_set_1['label'].values.tolist(), transform=test_transform)
  
    train_dataset_unlabelled_test = torch.utils.data.ConcatDataset(
    [lt_unlabeled_known_dataset_test, lt_unlabeled_unknown_dataset_test])
    
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': train_dataset_unlabelled_test,
        'test': train_dataset_unlabelled_test,
    }
    return all_datasets
if __name__ == '__main__':
    x = get_endo_datasets(None, None)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')
    values, counts = np.unique(x["train_labelled"].target, return_counts=True)
    print('Num Labelled Classes:',values, counts)
    # values, counts = np.unique(x["train_unlabelled"].labels, return_counts=True)
    # print('Num Unabelled Classes:',values, counts)
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')





# +
# df_all = pd.read_csv('/mnt/workdir/fengwei/NCD/labeled-images/image-labels.csv')
# findings = df_all['Finding'].values
# print(df_all['Finding'].value_counts())
# findings_list = list(np.unique(findings))
# findings_to_class = dict(zip(findings_list, np.arange(len(findings_list))))
# class_to_findings = dict(zip(np.arange(len(findings_list)),findings_list))
#
# images = df_all['Video file'].values
# df_all['category'] = df_all.Finding.replace(findings_to_class)
# df_all.drop(['Organ', 'Classification'], axis=1, inplace=True)
# df_all.columns = ['image_id', 'finding_name', 'finding']
#
# num_ims = len(df_all)
# meh, df_val1 = train_test_split(df_all, test_size=num_ims//5, random_state=0, stratify=df_all.finding)
# meh, df_val2 = train_test_split(meh,    test_size=num_ims//5, random_state=0, stratify=meh.finding)
# meh, df_val3 = train_test_split(meh,    test_size=num_ims//5, random_state=0, stratify=meh.finding)
# df_val5, df_val4 = train_test_split(meh,test_size=num_ims//5, random_state=0, stratify=meh.finding)
#
# df_train1 = pd.concat([df_val2,df_val3,df_val4,df_val5], axis=0)
# df_train2 = pd.concat([df_val1,df_val3,df_val4,df_val5], axis=0)
# df_train3 = pd.concat([df_val1,df_val2,df_val4,df_val5], axis=0)
# df_train4 = pd.concat([df_val1,df_val2,df_val3,df_val5], axis=0)
# df_train5 = pd.concat([df_val1,df_val2,df_val3,df_val4], axis=0)
#
# df_train1.to_csv('data/train_endo1.csv', index=None)
# df_val1.to_csv('data/val_endo1.csv', index=None)
#
# df_train2.to_csv('data/train_endo2.csv', index=None)
# df_val2.to_csv('data/val_endo2.csv', index=None)
#
# df_train3.to_csv('data/train_endo3.csv', index=None)
# df_val3.to_csv('data/val_endo3.csv', index=None)
#
# df_train4.to_csv('data/train_endo4.csv', index=None)
# df_val4.to_csv('data/val_endo4.csv', index=None)
#
# df_train5.to_csv('data/train_endo5.csv', index=None)
# df_val5.to_csv('data/val_endo5.csv', index=None)
