from train import *
import os
from torchvision import transforms as T
import json
import torch
from PIL import Image
import cv2
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import random
from test_global import *

classics = {
    0:u"red",
    1:u"green",
    2:u"blue",
    3:u"yellow",
    4:u"Cyan"
}

def read_one_pic(img_path):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    transforms = T.Compose([
        T.ToTensor(),
        normalize
    ])

    data = Image.open(img_path)
    data = transforms(data)
    data = data.reshape(1,data.size()[0],data.size()[1],data.size(2))
    return data

def draw_by_one_data(pic_path,rect_datas):
    # i, data1, data2
    img = Image.open(pic_path)
    draw = ImageDraw.Draw(img)
    for rd in rect_datas:
        i = rd[0]
        data1 = rd[1]
        data2 = rd[2]
        index_x = int(i % CEL_NUMS)
        index_y = int(i / CEL_NUMS)
        sub_x = data1[0] * CEL_LEN
        sub_y = data1[1] * CEL_LEN
        center_x = sub_x + CEL_LEN * index_x
        center_y = sub_y + CEL_LEN * index_y
        width = data1[2] * IMG_WIDTH
        height = data1[3] * IMG_HEIGHT

        classic = torch.argmax(data2)
        rect = [center_x-width/2,center_y-height/2,center_x+width/2,center_y+height/2]


        xs = [rect[0], rect[2]]
        ys = [rect[1], rect[3]]
        draw.line((xs[0], ys[0], xs[1], ys[0]), fill=(0, 0, 0),width=2)
        draw.line((xs[1], ys[0], xs[1], ys[1]), fill=(0, 0, 0),width=2)
        draw.line((xs[1], ys[1], xs[0], ys[1]), fill=(0, 0, 0),width=2)
        draw.line((xs[0], ys[1], xs[0], ys[0]), fill=(0, 0, 0),width=2)
        index = int(classic)
        if index == -1:
            draw.text((rect[0], rect[1]), 'unknown', fill=(0, 0, 0))
        else:
            t = classics[index]
            draw.text((rect[0],rect[1]),t,fill=(0,0,0))

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('test', img)
    cv2.waitKey(0)


def cac_one_pic_and_save():
    root = r"imgs\imgs"
    pic_paths = [os.path.join(root,img) for img in os.listdir(root)]
    # pic_paths = pic_paths[int(0.7*len(pic_paths)):]
    pic_paths = pic_paths[:int(0.7 * len(pic_paths))]
    random.shuffle(pic_paths)

    if not os.path.exists(pkl_name):
        print("can not find pkl(%s)"%pkl_name)
        return
    cnn = torch.load(pkl_name).cpu().eval()

    for pic_path in pic_paths:
        img_data = read_one_pic(pic_path)
        result = cnn(img_data)
        result = result.reshape(result.size()[1],result.size()[2])
        results = torch.split(result,[5,5],1)
        rect_datas = []
        for i in range(results[0].size()[0]):
            data1 = results[0][i]
            data2 = results[1][i]
            if (data1[4]<0.5):
                continue
            rect_datas.append([i,data1,data2])
        draw_by_one_data(pic_path,rect_datas)
        pass


if __name__=='__main__':
    cac_one_pic_and_save()