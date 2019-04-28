__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
import io
import sys
import os
from PIL import Image,ImageDraw,ImageFont
import random
import math
import json
import shutil
import cv2
import numpy as np
import torch
from vocglobal import *
from xml.dom.minidom import parse
import xml.dom.minidom

class classic_data:
    def __init__(self,index,color,label):
        self.index = index
        self.color = color
        self.label = label

class label_data:
    def __init__(self):
        self.classic = None
        self.rect = None


classics = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
            'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa',
            'train','tvmonitor']




def iou_check(classic,rect,rects):
    for r in rects:
        iou = compute_iou(rect,r.rect)
        if classic ==r.classic:
            if iou>0.0:
                return False
        else:
            if iou>0.15:
                return False
    return True

def create_simple_rect(voc_pic,voc_label_dir,folder_path,label_folder,index):
    res = []
    img = Image.open(voc_pic)
    if img.mode == 'L':
        img = img.convert('RGB')
    w_ori = img.size[0]
    h_ori = img.size[1]
    img = img.resize((IMG_WIDTH,IMG_HEIGHT))
    # img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    w_scale = IMG_WIDTH/w_ori
    h_scale = IMG_HEIGHT/h_ori

    voc_name = voc_pic.split("\\")[-1].replace('.jpg','.xml')
    voc_label_path = os.path.join(voc_label_dir,voc_name)
    DOMTree = xml.dom.minidom.parse(voc_label_path)
    collection = DOMTree.documentElement
    objects = collection.getElementsByTagName("object")

    for obj in objects:
        name = obj.getElementsByTagName("name")[0].childNodes[0].data
        bnd_box = obj.getElementsByTagName("bndbox")[0]
        xmin = int(bnd_box.getElementsByTagName("xmin")[0].childNodes[0].data)
        ymin = int(bnd_box.getElementsByTagName("ymin")[0].childNodes[0].data)
        xmax = int(bnd_box.getElementsByTagName("xmax")[0].childNodes[0].data)
        ymax = int(bnd_box.getElementsByTagName("ymax")[0].childNodes[0].data)

        xmin = int(xmin*w_scale)
        ymin = int(ymin*h_scale)
        xmax = int(xmax*w_scale)
        ymax = int(ymax*h_scale)
        ld = label_data()
        ld.classic = classics.index(name)
        ld.rect = (xmin,ymin,xmax,ymax)
        res.append(ld)

    file_path = format("%s\\img_%d.jpg" % (folder_path, index))
    img.save(file_path,'jpeg')

    label_json = json.dumps(res,default=lambda obj:obj.__dict__,indent=4)
    label_path = format("%s\\img_%d.json"%(label_folder,index))
    f = open(label_path,'w+')
    f.writelines(label_json)
    f.close()

def data_show(img_dir,label_dir):
    datas = []
    for (root, dirs, files) in os.walk(img_dir):
        for item in files:
            d = os.path.join(root, item)
            js_name = item.replace('.jpg','.json')
            label_d = os.path.join(label_dir,js_name)
            datas.append((d,label_d))

    for d in datas:
        img_path = d[0]
        label_path = d[1]
        img = Image.open(img_path)

        with open(label_path) as f:
            json_obj = json.load(f)

        for label_obj in json_obj:
            type = label_obj['classic']
            rect = label_obj['rect']
            xs = [rect[0],rect[2]]
            ys = [rect[1],rect[3]]
            draw = ImageDraw.Draw(img)
            draw.line((xs[0],ys[0],xs[1],ys[0]),fill=(255,0,0),width=3)
            draw.line((xs[1], ys[0], xs[1], ys[1]), fill=(255, 0, 0),width=3)
            draw.line((xs[1], ys[1], xs[0], ys[1]), fill=(255, 0, 0),width=3)
            draw.line((xs[0], ys[1], xs[0], ys[0]), fill=(255, 0, 0),width=3)
            draw.text((xs[0],ys[0]),classics[type])

        img = np.array(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)


        cv2.imshow('test',img)
        cv2.waitKey(0)

    print(datas)

class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x,y,z):
        a = (z==1).float()*(x-y)
        return torch.mean(torch.pow(a, 2))

if __name__=='__main__':
    img_foler = 'imgs\\imgs'
    label_foler = 'imgs\\labels'
    if not os.path.exists(img_foler):
        os.makedirs(img_foler)

    if not os.path.exists(label_foler):
        os.makedirs(label_foler)

    voc_dir = r'E:\tensorflow_datas\voc\voc2007\JPEGImages'
    voc_pics = [os.path.join(voc_dir, img) for img in os.listdir(voc_dir)]

    voc_anno_dir = r'E:\tensorflow_datas\voc\voc2007\Annotations'

    for i in range(200):
        create_simple_rect(voc_pics[i],voc_anno_dir,img_foler,label_foler,i)


    # data_show(img_foler,label_foler)

    # a = torch.tensor([[0,1,2,3,4],[1,2,3,4,5]],dtype=torch.float32)
    # b = torch.tensor([[1,2,3,4,5],[2,3,4,5,6]],dtype=torch.float32)
    # z = torch.tensor([[1,0,1,0,1],[1,0,1,0,1]],dtype=torch.float32)
    #
    # loss_func1 = My_loss()
    #
    # # sum = 0
    # # for i in range(a.size()[0]):
    # #     if a[i]%2==0:
    # #         sum += a[i]
    #
    # # sum = ((a % 2 == 0).long() * a).sum()
    # # sum = (a<4).float()*a
    #
    # loss = loss_func1(a,b,z)
    # print(loss)

