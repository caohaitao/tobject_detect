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
from test_global import *


pkl_name = "detection3.pkl"
EPOCH = 128
BATCH_SIZE = 4
default_lr = 0.0001


# 输出的通道数目，分别是∆x,∆y,w,h,c+5个分类
out_channel = 5+CLASSIC_NUMS

cfg = [64, 'M', 128,128, 'M', 256,256,256,256,256, 'M', 512,512,512,512,512, 'M', 1024,1024,1024,1024, 'M',512,256]
# cfg = [64, 'M', 128,128, 'M', 256,256,256, 'M', 512,512,512, 'M',256]


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
    pass
    # model.eval()
    #
    # loss_meter = meter.AverageValueMeter()
    # loss_meter1 = meter.AverageValueMeter()
    # loss_meter2 = meter.AverageValueMeter()
    # loss_meter3 = meter.AverageValueMeter()
    #
    # loss_func1 = My_mse().cuda()
    # loss_func2 = My_mse2().cuda()
    # loss_func3 = torch.nn.CrossEntropyLoss().cuda()
    # for ii, (data, label0, label1,label2) in enumerate(dataloader):
    #     input = Variable(data).cuda()
    #     target0 = Variable(label0)
    #     target1 = Variable(label1)
    #     target2 = Variable(label2)
    #
    #     score = model(input)
    #     scores = torch.split(score, [4,1,CLASSIC_NUMS], 2)
    #
    #     t1 = target1.reshape(target1.size()[0] * target1.size()[1])
    #     indicates = torch.nonzero(t1 == 1).view(-1)
    #     noindicates = torch.nonzero(t1 == 0).view(-1)
    #
    #
    #
    #
    #     loss1 = loss_func1(scores[0], target0.cuda(), target1.cuda())
    #     loss2 = loss_func2(scores[1], target1.cuda())
    #
    #
    #
    #     scores3 = scores[2].reshape(scores[2].size()[0] * scores[2].size()[1], scores[2].size()[2])
    #     t2 = target2.reshape(target2.size()[0] * target2.size()[1])
    #
    #
    #
    #     scores3 = torch.index_select(scores3, 0, indicates.cuda())
    #     t2 = torch.index_select(t2, 0, indicates)
    #
    #     loss3 = loss_func3(scores3, t2.cuda())
    #
    #     loss1 = 5 * loss1
    #     loss2 = 5 * loss2
    #     # loss = loss1 + loss2 + loss3
    #     loss = loss3
    #
    #
    #     loss_meter.add(loss.detach().cpu().numpy())
    #     loss_meter1.add(loss1.detach().cpu().numpy())
    #     loss_meter2.add(loss2.detach().cpu().numpy())
    #     loss_meter3.add(loss3.detach().cpu().numpy())
    #
    # model.train()
    #
    # return loss_meter.value()[0],\
    #        loss_meter1.value()[0],\
    #        loss_meter2.value()[0],\
    #        loss_meter3.value()[0]

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

def get_rect_by_value(target,ind):
    x = target[ind][0]
    y = target[ind][1]
    w = target[ind][2]
    h = target[ind][3]
    index = ind % (CEL_NUMS * CEL_NUMS)
    index_x = int(index % CEL_NUMS)
    index_y = int(index / CEL_NUMS)
    sub_x = x * CEL_LEN
    sub_y = y * CEL_LEN
    center_x = sub_x + CEL_LEN * index_x
    center_y = sub_y + CEL_LEN * index_y
    width = w * IMG_WIDTH
    height = h * IMG_HEIGHT
    return [int(center_x-width/2),int(center_y-height/2),int(center_x+width/2),int(center_y+height/2)]

def train():
    if os.path.exists(pkl_name):
        cnn = torch.load(pkl_name).cuda()
        logger.info("load model(%s) success" % pkl_name)
    else:
        cnn = detection1().cuda()
        logger.info("new model success")

    print(cnn)


    mse_l = torch.nn.MSELoss().cuda()
    cross_entry_l = torch.nn.CrossEntropyLoss().cuda()

    break_flag = 0

    for c in range(1000):
        print("while(%d) begin"%c)
        train_data = Detection1DataSet('imgs\\imgs', 'imgs\\labels', nums=100, train=True)
        # val_data = Detection1DataSet('imgs\\imgs', 'imgs\\labels', train=False)

        lr = get_lr()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
        loss_meter = meter.AverageValueMeter()
        loss_meter_pos = meter.AverageValueMeter()
        loss_meter_body_c = meter.AverageValueMeter()
        loss_meter_nobody_c = meter.AverageValueMeter()
        loss_meter_classic = meter.AverageValueMeter()
        previous_loss = 1e100

        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )



        # val_dataloader = torch.utils.data.DataLoader(
        #     val_data,
        #     BATCH_SIZE,
        #     shuffle=False,
        #     num_workers=4
        # )
        for epoch in range(EPOCH):
            loss_meter.reset()
            loss_meter_pos.reset()
            loss_meter_body_c.reset()
            loss_meter_nobody_c.reset()
            loss_meter_classic.reset()

            # print("whole_step=%d"%len(train_dataloader))
            for ii,(data,label0,label1,label2) in enumerate(train_dataloader):
                input = Variable(data).cuda()
                target0 = Variable(label0)
                target1 = Variable(label1)
                target2 = Variable(label2)

                target0 = target0.reshape(target0.size()[0]*target0.size()[1],target0.size()[2])
                target1 = target1.reshape(target1.size()[0] * target1.size()[1])
                target2 = target2.reshape(target2.size()[0] * target2.size()[1])
                indicates = torch.nonzero(target1 == 1).view(-1)
                nonindicates = torch.nonzero(target1 == 0).view(-1)

                ious = torch.zeros((target1.size()[0],1),dtype=torch.float32)

                optimizer.zero_grad()
                score = cnn(input)
                scores = torch.split(score,[4,1,CLASSIC_NUMS],2)

                rect_predict = scores[0]
                rect_predict = rect_predict.reshape(rect_predict.size()[0]*rect_predict.size()[1],rect_predict.size()[2])
                for ind in indicates:
                    rect_truth = get_rect_by_value(target0,ind)
                    rect_prd = get_rect_by_value(rect_predict,ind)
                    iou = compute_iou(rect_truth,rect_prd)
                    ious[ind] = iou

                s0_view = scores[0].reshape(scores[0].size()[0]*scores[0].size()[1],scores[0].size()[2])
                s1_view = scores[1].reshape(scores[1].size()[0]*scores[1].size()[1],scores[1].size()[2])
                s2_view = scores[2].reshape(scores[2].size()[0]*scores[2].size()[1],scores[2].size()[2])

                t0_select = torch.index_select(target0.cuda(),0,indicates.cuda())
                s0_select = torch.index_select(s0_view.cuda(),0,indicates.cuda())


                loss_body_pos = mse_l(s0_select,t0_select)

                body_ious = torch.index_select(ious.cuda(),0,indicates.cuda())
                body_ious_pred = torch.index_select(s1_view.cuda(),0,indicates.cuda())
                loss_body_confidence = mse_l(body_ious_pred,body_ious)

                nobody_ious = torch.index_select(ious.cuda(),0,nonindicates.cuda())
                nobody_ious_pred = torch.index_select(s1_view.cuda(),0,nonindicates.cuda())
                loss_nobody_confidence = mse_l(nobody_ious_pred,nobody_ious)

                body_classics = torch.index_select(target2.cuda(),0,indicates.cuda())
                body_classics_pred = torch.index_select(s2_view.cuda(),0,indicates.cuda())
                loss_classic = cross_entry_l(body_classics_pred,body_classics)
                loss = 5*loss_body_pos + loss_body_confidence+0.5*loss_nobody_confidence+loss_classic

                if ii%100==0 and ii!=0:
                    print("epoch(%d) step(%d) loss1(%0.6f) loss2(%0.6f) loss3(%0.6f) loss(%0.6f) percent(%0.6f)"%(
                        epoch,
                        ii,
                        loss_meter_pos.value()[0],
                        loss_meter_body_c.value()[0],
                        loss_meter_nobody_c.value()[0],
                        loss_meter.value()[0],
                        ii/len(train_dataloader)
                    ))

                loss.backward()
                optimizer.step()

                loss_meter.add(loss.detach().cpu().numpy())
                loss_meter_pos.add(loss_body_pos.detach().cpu().numpy())
                loss_meter_body_c.add(loss_body_confidence.detach().cpu().numpy())
                loss_meter_nobody_c.add(loss_nobody_confidence.detach().cpu().numpy())
                loss_meter_classic.add(loss_classic.detach().cpu().numpy())

            # torch.save(cnn,pkl_name)
            # logger.info("save model(%s) success" % pkl_name)
            torch.cuda.empty_cache()

            # if epoch%8==0:
            #     val_loss,val_loss1,val_loss2,val_loss3 = val(cnn,val_dataloader)
            #
            #     logger.info("epoch(%d) train_loss(%0.6f,%0.6f,%0.6f,%0.6f) val_loss(%0.6f,%0.6f,%0.6f,%0.6f)"%
            #           (epoch,loss_meter1.value()[0],loss_meter2.value()[0],loss_meter3.value()[0],loss_meter.value()[0],
            #            val_loss1,val_loss2,val_loss3,val_loss))
            # else:
            logger.info("while(%d) epoch(%d) train_loss(%0.6f,%0.6f,%0.6f,%0.6f,%0.6f)"%
                  (c,epoch,loss_meter_pos.value()[0],
                   loss_meter_body_c.value()[0],
                   loss_meter_nobody_c.value()[0],
                   loss_meter_classic.value()[0],
                   loss_meter.value()[0]))

            if loss_meter.value()[0]<0.03:
                print("while(%d) epoch(%d) loss(%0.6f)<1 break"%(c,epoch,loss_meter.value()[0]))
                if epoch==0:
                    break_flag += 1
                    print("break flag change(%d)"%break_flag)
                break

            break_flag = 0

            # if loss_meter.value()[0] > previous_loss:
            #     logger.info("lr change from %0.7f -> %0.7f"%(lr,lr*0.95))
            #     lr = lr * 0.95
            #     # save_lr(lr)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
            # previous_loss = loss_meter.value()[0]
            torch.cuda.empty_cache()

        torch.save(cnn, pkl_name)
        if break_flag == 5:
            print("break_flag == 5,train over")
            break

if __name__=='__main__':
    logger = Logger(logname='test3.log', loglevel=1, logger="fox").getlog()
    train()