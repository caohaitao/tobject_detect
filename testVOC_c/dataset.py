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
from vocglobal import *
import random

class Detection1DataSet(data.Dataset):
    def __init__(self,root,label_dir,nums,transforms=None,train=True,test=False):
        self.test = test
        imgs = [os.path.join(root,img) for img in os.listdir(root)]

        self.label_dir = label_dir

        imgs_num = len(imgs)

        np.random.seed(6)

        if train == True:
            self.imgs = imgs[0:int(0.7*imgs_num)]
        else:
            self.imgs = imgs[int(0.7*imgs_num):]

        if nums < len(self.imgs):
            self.imgs = random.sample(self.imgs,nums)

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

        label1 = torch.zeros((CEL_NUMS * CEL_NUMS,1), dtype=torch.float32)
        for ob in obj:
            classic = ob['classic']
            rect = ob['rect']

            x_begin = int(rect[0]/CEL_LEN)
            x_end = int(rect[2]/CEL_LEN)+1
            x_end = min(x_end,CEL_NUMS)
            y_begin = int(rect[1]/CEL_LEN)
            y_end = int(rect[3]/CEL_LEN)+1
            y_end = min(y_end,CEL_NUMS)
            for index_y in range(y_begin,y_end):
                for index_x in range(x_begin,x_end):
                    temp_rect = [index_x*CEL_LEN,index_y*CEL_LEN,(index_x+1)*CEL_LEN,(index_y+1)*CEL_LEN]
                    cross_rect = get_two_rect_cross_rect(temp_rect,rect)
                    iou = compute_iou(cross_rect,temp_rect)
                    pos = index_y * CEL_NUMS + index_x
                    # print('pos=%d'%pos)
                    # print('rect=',rect)
                    label1[pos] = iou





        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label1

    def __len__(self):
        '''
        返回数据集中所有图片的个数
        '''
        return len(self.imgs)
