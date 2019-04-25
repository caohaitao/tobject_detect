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

class classic_data:
    def __init__(self,index,color,label):
        self.index = index
        self.color = color
        self.label = label

class label_data:
    def __init__(self):
        self.classic = None
        self.rect = None

classics = {
    0:classic_data(0,(255,0,0),"red"),
    1:classic_data(1,(0,255,0),"green"),
    2:classic_data(2,(0,0,255),"blue"),
    3:classic_data(3,(255,255,0),"yellow"),
    4:classic_data(4,(0,255,255),"Cyan")
}

img_width = 416
img_height = 416

def create_simple_rect(folder_path,label_folder,index):
    pos_x = random.randint(100,300)
    pos_y = random.randint(100,300)
    rect_width = random.randint(96,256)
    rect_height = random.randint(96,256)
    type = random.randint(0,4)

    file_path = format("%s\\img_%d.jpg"%(folder_path,index))

    img = Image.new('RGB',(img_width,img_height),(255,255,255))
    draw = ImageDraw.Draw(img)

    min_x = pos_x
    min_y = pos_y
    max_x = int(min(pos_x + rect_width,img_width))
    max_y = int(min(pos_y + rect_height,img_width))

    ld = label_data()
    ld.classic = type
    ld.rect = (min_x,min_y,max_x,max_y)

    draw.rectangle(ld.rect,fill=classics[type].color)

    img.save(file_path,'jpeg')

    label_json = json.dumps(ld,default=lambda obj:obj.__dict__,indent=4)
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

        rect = json_obj['rect']
        xs = [rect[0],rect[2]]
        ys = [rect[1],rect[3]]
        draw = ImageDraw.Draw(img)
        draw.line((xs[0],ys[0],xs[1],ys[0]),fill=(255,0,0))
        draw.line((xs[1], ys[0], xs[1], ys[1]), fill=(255, 0, 0))
        draw.line((xs[1], ys[1], xs[0], ys[1]), fill=(255, 0, 0))
        draw.line((xs[0], ys[1], xs[0], ys[0]), fill=(255, 0, 0))

        img = np.array(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)


        cv2.imshow('test',img)
        cv2.waitKey(0)

    print(datas)


if __name__=='__main__':
    img_foler = 'imgs\\imgs'
    label_foler = 'imgs\\labels'
    if not os.path.exists(img_foler):
        os.makedirs(img_foler)

    if not os.path.exists(label_foler):
        os.makedirs(label_foler)
    # for i in range(1000):
    #     create_simple_rect(img_foler,label_foler,i)

    data_show(img_foler,label_foler)

