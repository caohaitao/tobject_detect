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

class classic_data:
    def __init__(self,index,color,label):
        self.index = index
        self.color = color
        self.label = label

class label_data:
    def __init__(self):
        self.classic = None
        self.rect = None

classics ={
    0:classic_data(0,(237,100,92),"red"),
    1:classic_data(1,(96,234,134),"green"),
    2:classic_data(2,(100,100,240),"blue"),
    3:classic_data(3,(240,240,120),"yellow"),
    4:classic_data(4,(50,240,240),"Cyan")
}

img_width = 416
img_height = 416


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


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

def create_simple_rect(disturb_paths,folder_path,label_folder,index):
    res = []
    obj_nums = random.randint(1,4)
    use_disturb_index = random.randint(0,len(disturb_paths)-1)
    # img = Image.open(disturb_paths[use_disturb_index])
    # if img.mode == 'L':
    #     img = img.convert('RGB')
    # img = img.resize((img_width,img_height))
    img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    while True:
        pos_x = random.randint(60,360)
        pos_y = random.randint(60,360)
        rect_width = random.randint(32,128)
        rect_height = random.randint(32,128)
        type = random.randint(0,4)

        min_x = pos_x
        min_y = pos_y
        max_x = int(min(pos_x + rect_width,img_width))
        max_y = int(min(pos_y + rect_height,img_width))

        ld = label_data()
        ld.classic = type
        ld.rect = (min_x,min_y,max_x,max_y)

        if iou_check(ld.classic,ld.rect,res) == False:
            continue
        draw.rectangle(ld.rect,fill=classics[type].color)
        res.append(ld)
        if len(res) == obj_nums:
            break

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
            rect = label_obj['rect']
            xs = [rect[0],rect[2]]
            ys = [rect[1],rect[3]]
            draw = ImageDraw.Draw(img)
            draw.line((xs[0],ys[0],xs[1],ys[0]),fill=(0,0,0),width=3)
            draw.line((xs[1], ys[0], xs[1], ys[1]), fill=(0, 0, 0),width=3)
            draw.line((xs[1], ys[1], xs[0], ys[1]), fill=(0, 0, 0),width=3)
            draw.line((xs[0], ys[1], xs[0], ys[0]), fill=(0, 0, 0),width=3)

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

    disturb_dir = r'E:\tensorflow_datas\coco\train2014\train2014'
    disturb_paths = [os.path.join(disturb_dir, img) for img in os.listdir(disturb_dir)]

    for i in range(3000):
        create_simple_rect(disturb_paths,img_foler,label_foler,i)

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

