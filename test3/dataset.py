__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
# third-party library
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import json
import torch

class Detection1DataSet(data.Dataset):
    def __init__(self,root,label_dir,transforms=None,train=True,test=False):
        self.test = test
        imgs = [os.path.join(root,img) for img in os.listdir(root)]

        self.label_dir = label_dir

        imgs_num = len(imgs)

        np.random.seed(100)

        if train == True:
            self.imgs = imgs[0:int(0.7*imgs_num)]
        else:
            self.imgs = imgs[int(0.7*imgs_num):]

        if transforms is None:

            # 数据转换操作，测试验证和训练的数据转换有所区别
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # 测试集和验证集不用数据增强
            if self.test or not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    normalize
                ])
                # 训练集需要数据增强
            else:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        '''
        img_path = self.imgs[index]

        json_name = img_path.split('\\')[-1].replace('.jpg','.json')

        json_path = os.path.join(self.label_dir,json_name)

        with open(json_path) as load_f:
            obj = json.load(load_f)

        label0 = torch.zeros((13*13,4),dtype=torch.float32)
        label1 = torch.zeros((13*13,1),dtype=torch.float32)
        label2 = torch.zeros((13*13),dtype=torch.int64)

        for ob in obj:
            classic = ob['classic']+1
            rect = ob['rect']

            rect_center = ((rect[0]+rect[2])/2,(rect[1]+rect[3])/2)

            center_index_x = int(rect_center[0]/32)
            center_index_y = int(rect_center[1]/32)

            sub_x = rect_center[0] - 32*center_index_x
            sub_y = rect_center[1] - 32*center_index_y
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]

            pos = center_index_y*13+center_index_x
            label0[pos][0] = sub_x/32
            label0[pos][1] = sub_y/32
            label0[pos][2] = width/416
            label0[pos][3] = height/416
            label1[pos][0] = 1.0

            label2[pos] = classic

        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label0,label1,label2

    def __len__(self):
        '''
        返回数据集中所有图片的个数
        '''
        return len(self.imgs)
