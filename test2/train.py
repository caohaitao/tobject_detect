'''VGG11/13/16/19 in Pytorch.'''
import os
import torch
import torch.nn as nn
import torch.onnx
from torchnet import meter
from torch.autograd import Variable
import time
from dataset import Detection1DataSet
from log import *


pkl_name = "detection2.pkl"
EPOCH = 32
BATCH_SIZE = 8
default_lr = 0.0001


# 输出的通道数目，分别是∆x,∆y,w,h,c+6个分类（0是没有物体，其他分类是分类标签+1）
out_channel = 5+6

cfg = [64, 'M', 128,128, 'M', 256,256,256, 'M', 512,512,512, 'M', 1024,1024, 'M']


class detection1(nn.Module):
    def __init__(self):
        super(detection1, self).__init__()
        self.features = self._make_layers(cfg)

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size()[0],out.size()[1],out.size()[2]*out.size()[3])
        out = out.permute(0,2,1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.Conv2d(in_channels,out_channel,kernel_size=3,padding=1)]
        return nn.Sequential(*layers)


def val(model,dataloader):
    model.eval()

    loss_meter = meter.AverageValueMeter()
    loss_meter1 = meter.AverageValueMeter()
    loss_meter2 = meter.AverageValueMeter()
    loss_func1 = torch.nn.MSELoss()
    loss_func2 = torch.nn.CrossEntropyLoss()
    for ii, (data, label0, label1) in enumerate(dataloader):
        input = Variable(data).cuda()
        target0 = Variable(label0)
        target1 = Variable(label1)

        score = model(input)
        scores = torch.split(score, [5, 6], 2)
        loss1 = loss_func1(scores[0].cpu(), target0)

        loss2 = None
        for i in range(target1.size()[0]):
            if loss2 is None:
                loss2 = 0.5 * loss_func2(scores[1][i].cpu(), target1[i])
            else:
                loss2 += 0.5 * loss_func2(scores[1][i].cpu(), target1[i])
        loss2 /= BATCH_SIZE

        loss1 = 5*loss1
        loss = loss1 + loss2


        loss_meter.add(loss.detach().numpy())
        loss_meter1.add(loss1.detach().numpy())
        loss_meter2.add(loss2.detach().numpy())

    model.train()

    return loss_meter.value()[0],loss_meter1.value()[0],loss_meter2.value()[0]

def save_lr(lr):
    global logger
    f = open("lr.txt","w+")
    s = format("%0.7f"%lr)
    f.writelines(s)
    logger.info("save lr(%0.7f) success"%lr)

def get_lr():
    global logger
    if not os.path.exists("lr.txt"):
        logger.info("not find lr.txt return ,%0.7f"%default_lr)
        return default_lr
    else:
        f = open("lr.txt",'r')
        s = f.read()
        f.close()
        logger.info("get lr(%s) form lr.txt success"%s)
        return float(s)

def train():
    if os.path.exists(pkl_name):
        cnn = torch.load(pkl_name).cuda()
        logger.info("load model(%s) success" % pkl_name)
    else:
        cnn = detection1().cuda()
        logger.info("new model success")

    print(cnn)

    lr = get_lr()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

    loss_func1 = torch.nn.MSELoss().cuda()
    loss_func2 = torch.nn.CrossEntropyLoss().cuda()

    train_data = Detection1DataSet('imgs\\imgs','imgs\\labels',train=True)
    val_data = Detection1DataSet('imgs\\imgs','imgs\\labels',train=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    loss_meter = meter.AverageValueMeter()
    loss_meter1 = meter.AverageValueMeter()
    loss_meter2 = meter.AverageValueMeter()
    previous_loss = 1e100

    for epoch in range(EPOCH):
        loss_meter.reset()
        loss_meter1.reset()
        loss_meter2.reset()

        for ii,(data,label0,label1) in enumerate(train_dataloader):
            input = Variable(data).cuda()
            target0 = Variable(label0).cuda()
            target1 = Variable(label1).cuda()

            optimizer.zero_grad()
            score = cnn(input)
            scores = torch.split(score,[5,6],2)
            loss1 = loss_func1(scores[0],target0)

            loss2 = None
            for i in range(target1.size()[0]):
                if loss2 is None:
                    loss2 = 0.5*loss_func2(scores[1][i],target1[i])
                else:
                    loss2 += 0.5 * loss_func2(scores[1][i], target1[i])
            loss2 /= BATCH_SIZE

            loss1 = 5*loss1
            loss = loss1 + loss2
            if ii%100==0 and ii!=0:
                print("epoch(%d) step(%d) loss1(%0.6f) loss2(%0.6f) loss(%0.6f)"%(
                    epoch,
                    ii,
                    loss_meter1.value()[0],
                    loss_meter2.value()[0],
                    loss_meter.value()[0]
                ))

            loss.backward()
            optimizer.step()

            loss_meter.add(loss.detach().cpu().numpy())
            loss_meter1.add(loss1.detach().cpu().numpy())
            loss_meter2.add(loss2.detach().cpu().numpy())

        torch.save(cnn,pkl_name)
        logger.info("save model(%s) success" % pkl_name)

        val_loss,val_loss1,val_loss2 = val(cnn,val_dataloader)

        logger.info("epoch(%d) train_loss(%0.6f,%0.6f,%0.6f) val_loss(%0.6f,%0.6f,%0.6f)"%
              (epoch,loss_meter1.value()[0],loss_meter2.value()[0],loss_meter.value()[0],
               val_loss1,val_loss2,val_loss))

        if loss_meter.value()[0] > previous_loss:
            logger.info("lr change from %0.7f -> %0.7f"%(lr,lr*0.95))
            lr = lr * 0.95
            save_lr(lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]
        torch.cuda.empty_cache()

if __name__=='__main__':
    logger = Logger(logname='test2.log', loglevel=1, logger="fox").getlog()
    train()