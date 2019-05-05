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


pkl_name = "detection3.pkl"
EPOCH = 16
BATCH_SIZE = 8
default_lr = 0.001


# 输出的通道数目，分别是∆x,∆y,w,h,c+5个分类
out_channel = 5+5

# cfg = [64, 'M', 128,128, 'M', 256,256,256, 'M', 512,512,512, 'M', 1024,1024,1024, 'M',512]
cfg = [64, 'M', 128,128, 'M', 256,256,256, 'M', 512,512,512, 'M',256]


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

class My_mse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x,y,z):
        a = (z==1.0).float()*(x-y)
        return torch.mean(torch.pow(a, 2))

class My_mse2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x,y):
        a = (y==1.0).float()*(x-y)
        m1 = torch.mean(torch.pow(a,2))
        a = (y==0).float()*(x-y)
        m2 = torch.mean(torch.pow(a,2))
        return m1 + 0.5*m2

def val(model,dataloader):
    model.eval()

    loss_meter = meter.AverageValueMeter()
    loss_meter1 = meter.AverageValueMeter()
    loss_meter2 = meter.AverageValueMeter()
    loss_meter3 = meter.AverageValueMeter()

    loss_func1 = My_mse().cuda()
    loss_func2 = My_mse2().cuda()
    loss_func3 = torch.nn.CrossEntropyLoss().cuda()
    for ii, (data, label0, label1,label2) in enumerate(dataloader):
        input = Variable(data).cuda()
        target0 = Variable(label0)
        target1 = Variable(label1)
        target2 = Variable(label2)

        score = model(input)
        scores = torch.split(score, [4,1,5], 2)
        loss1 = loss_func1(scores[0], target0.cuda(), target1.cuda())
        loss2 = loss_func2(scores[1], target1.cuda())

        t1 = target1.reshape(target1.size()[0] * target1.size()[1])

        scores3 = scores[2].reshape(scores[2].size()[0] * scores[2].size()[1], scores[2].size()[2])
        t2 = target2.reshape(target2.size()[0] * target2.size()[1])

        indicates = torch.nonzero(t1 == 1.0).view(-1)

        scores3 = torch.index_select(scores3, 0, indicates.cuda())
        t2 = torch.index_select(t2, 0, indicates)

        loss3 = loss_func3(scores3, t2.cuda())

        loss1 = 5 * loss1
        loss2 = 5 * loss2
        loss = loss1 + loss2 + loss3


        loss_meter.add(loss.detach().cpu().numpy())
        loss_meter1.add(loss1.detach().cpu().numpy())
        loss_meter2.add(loss2.detach().cpu().numpy())
        loss_meter3.add(loss3.detach().cpu().numpy())

    model.train()

    return loss_meter.value()[0],\
           loss_meter1.value()[0],\
           loss_meter2.value()[0],\
           loss_meter3.value()[0]

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

    # loss_func1 = torch.nn.MSELoss().cuda()
    loss_func1 = My_mse().cuda()
    loss_func2 = My_mse2().cuda()
    loss_func3 = torch.nn.CrossEntropyLoss().cuda()
    # loss_func2 = torch.nn.CrossEntropyLoss().cuda()

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
    loss_meter3 = meter.AverageValueMeter()
    previous_loss = 1e100

    for epoch in range(EPOCH):
        loss_meter.reset()
        loss_meter1.reset()
        loss_meter2.reset()
        loss_meter3.reset()

        for ii,(data,label0,label1,label2) in enumerate(train_dataloader):
            input = Variable(data).cuda()
            target0 = Variable(label0)
            target1 = Variable(label1)
            target2 = Variable(label2)

            optimizer.zero_grad()
            score = cnn(input)
            scores = torch.split(score,[4,1,5],2)
            loss1 = loss_func1(scores[0],target0.cuda(),target1.cuda())
            loss2 = loss_func2(scores[1],target1.cuda())

            t1 = target1.reshape(target1.size()[0]*target1.size()[1])

            scores3 = scores[2].reshape(scores[2].size()[0]*scores[2].size()[1],scores[2].size()[2])
            t2 = target2.reshape(target2.size()[0]*target2.size()[1])

            indicates = torch.nonzero(t1==1.0).view(-1)

            scores3 = torch.index_select(scores3, 0, indicates.cuda())
            t2 = torch.index_select(t2, 0, indicates)

            loss3 = loss_func3(scores3,t2.cuda())

            loss1 = 5*loss1
            loss2 = 5*loss2
            loss = loss1 + loss2 + loss3
            # if ii%100==0 and ii!=0:
            #     print("epoch(%d) step(%d) loss1(%0.6f) loss2(%0.6f) loss3(%0.6f) loss(%0.6f)"%(
            #         epoch,
            #         ii,
            #         loss_meter1.value()[0],
            #         loss_meter2.value()[0],
            #         loss_meter3.value()[0],
            #         loss_meter.value()[0]
            #     ))

            loss.backward()
            optimizer.step()

            loss_meter.add(loss.detach().cpu().numpy())
            loss_meter1.add(loss1.detach().cpu().numpy())
            loss_meter2.add(loss2.detach().cpu().numpy())
            loss_meter3.add(loss3.detach().cpu().numpy())

        torch.save(cnn,pkl_name)
        # logger.info("save model(%s) success" % pkl_name)
        torch.cuda.empty_cache()
        val_loss,val_loss1,val_loss2,val_loss3 = val(cnn,val_dataloader)

        logger.info("epoch(%d) train_loss(%0.6f,%0.6f,%0.6f,%0.6f) val_loss(%0.6f,%0.6f,%0.6f,%0.6f)"%
              (epoch,loss_meter1.value()[0],loss_meter2.value()[0],loss_meter3.value()[0],loss_meter.value()[0],
               val_loss1,val_loss2,val_loss3,val_loss))

        if loss_meter.value()[0] > previous_loss:
            logger.info("lr change from %0.7f -> %0.7f"%(lr,lr*0.95))
            lr = lr * 0.95
            save_lr(lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]
        torch.cuda.empty_cache()

if __name__=='__main__':
    logger = Logger(logname='test3.log', loglevel=1, logger="fox").getlog()
    train()