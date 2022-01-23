# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:20:49 2022

@author: Yuhong
"""

from Unet import Unet

import paddle
from paddle.nn import functional as F
input_channels = 3
out_filiters = 2

net = Unet2(3,2)

layer_state_dict = paddle.load("ASPP+Sep_epoch6.pdparams")
net.set_state_dict(layer_state_dict)
net.eval()

miou_all = []
dice_all = []
count = 0
Percision = paddle.metric.Precision()
Recall = paddle.metric.Recall()

res = []
for i, (image,label) in enumerate(train_loader()):
    image = image.astype(np.float32)         # 测试数据标签
    predict = net(image)    # 预测结果
    res.append(list(paddle.argmax(predict,axis=1)[0].numpy()))
    mIou = mean_iou(predict,label,num_classes=2).numpy()
    miou_all.append(mIou)
