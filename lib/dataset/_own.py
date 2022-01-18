from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
import pandas as pd

class _OWN(data.Dataset):
    def __init__(self, config, is_train=True):

        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']

        # convert name:indices to name:string
        with open(txt_file, 'r', encoding='utf-8') as file:
            # self.labels = [{c.split(' ')[0]: c.split(' ')[-1][:-1]} for c in file.readlines()]
            self.labels = [{c.split(' ')[0]: c.split(' ')[1:-1]} for c in file.readlines()]

        # for i in self.labels:
        #     print(i)

            # for c in file.readlines():
            #     print("///////////////////////")
            #     print(c)

        print("load {} images!".format(self.__len__()))
        # print(self.labels)

    def __len__(self):
        # print("//////////////////////" + str(len(self.labels)))
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        img = cv2.imread(os.path.join(self.root, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img = pd.read_csv(os.path.join(self.root, img_name))
        # img = img.values

        img_h, img_w = img.shape
        # print('&&&&&&&&&&&&&&&')
        # print(img_h, img_w)
        # print('img_h is {} **************'.format(img_h))
        # print('img_w is {} **************'.format(img_w))
        ratio = self.inp_h / self.inp_w  # 64/280
        if img_h / img_w > ratio:    # width is smaller than 280
            # print('width small!')

            img = cv2.copyMakeBorder(
                img, 0, 0, 0, int(img_h/ratio - img_w), cv2.BORDER_CONSTANT, value=0)

        img_h, img_w = img.shape
        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ttttt.jpg'), img)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))
        # img_h, img_w = img.shape
        # print('*************{}'.format(img_h))
        # print(img_w)
        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx








