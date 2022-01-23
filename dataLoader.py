# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:17:54 2022

@author: Yuhong
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:15:23 2022

@author: Yuhong
"""


import sys 
sys.path.append('/home/aistudio/external-libraries')
import paddle
from paddle.io import Dataset
import nibabel as nib
import os
from util import listdir
import util
from paddle.vision.transforms import Compose,Resize,CenterCrop,Normalize
from PIL import Image
import paddle.nn.functional as F
from paddle.vision.transforms import functional as F


DATA_SIZE = 80
BATCH_SIZE = 1
data_path = 'data/preprocess/data'
label_path = 'data/preprocess/label'
IMAGE_SIZE = '512X512X*'
CLASS_NUM = 2

train_split_rate = 1

class MyDataset(Dataset):
    def __init__(self, num_samples, data_path,label_path, train_split_rate=0.9,mode = 'train'):
        super(MyDataset,self).__init__()
        self.num_samples = num_samples#返回数据集大小
        self.train_size = int(self.num_samples * train_split_rate)
        if mode == 'train': 
            self.data_paths = util.listdir(data_path)[:self.train_size]
            self.label_paths = util.listdir(label_path)[:self.train_size]
        else:
            self.data_paths = util.listdir(data_path)[self.train_size:]
            self.label_paths = util.listdir(label_path)[self.train_size:]
    def __getitem__(self, index):
        image = np.load(os.path.join(data_path,self.data_paths[index]),allow_pickle=True)
        mask = np.load(os.path.join(label_path,self.label_paths[index]),allow_pickle=True)
        #image = image.astype('uint8')
        
        #image = F.adjust_contrast(image, 1.5)

        return image, mask
    def __len__(self):
        return len(self.label_paths)
train_Dataset = MyDataset(DATA_SIZE, data_path, label_path,mode='train')
test_Dataset = MyDataset(DATA_SIZE, data_path, label_path,mode='test')
train_loader = paddle.io.DataLoader(train_Dataset, batch_size=BATCH_SIZE,shuffle=False)
test_loader = paddle.io.DataLoader(test_Dataset, batch_size=1,shuffle=False)



from paddle import fluid
import paddle.nn.functional as F
def create_loss(predict, label, num_classes=1):
    ''' 创建loss，结合dice和交叉熵 '''
    #predict:BATCH_SIZE,2,512,512
    predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])#batch_size,512,512,2
    predict = fluid.layers.reshape(predict, shape=[-1, num_classes])
    predict = fluid.layers.softmax(predict)
    label = paddle.cast(label,'int64')
    label = fluid.layers.reshape(label, shape=[-1, 1])
    ce_loss = fluid.layers.cross_entropy(predict, label) # 计算交叉熵
    dice_loss = fluid.layers.dice_loss(predict,label)
    return fluid.layers.reduce_mean(ce_loss + dice_loss) # 最后使用的loss是dice和交叉熵的和，单独使用dice一般不是很稳定
    #return fluid.layers.reduce_mean(ce_loss)
def mean_iou(pred, label, num_classes=2):
    ''' 计算miou，评价网络分割结果的指标'''
    pred = fluid.layers.argmax(pred, axis=1)
    pred = fluid.layers.cast(pred, 'int32')
    label = fluid.layers.cast(label, 'int32')
    miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes)
    return miou

def focal_loss(pred, label, num_classes=2):
    pred = fluid.layers.argmax(pred, axis=1)
    pred = fluid.layers.reshape(pred, shape=[BATCH_SIZE, -1, 512, 512])
    pred = fluid.layers.cast(pred, 'float32')
    lable = fluid.layers.cast(label, 'float32')
    return F.sigmoid_focal_loss(pred, label, reduction='mean',alpha=0.75)


def hybird_loss(predict, label):
    focal_loss1 = focal_loss(predict, label, num_classes=2)
    ori_loss = create_loss(predict, label, num_classes=2)
    return focal_loss1 + ori_loss

def Gernel_dice_loss(pre,loss, num_classes=2):
    return 


def deep_super_Fusion_loss(out, label, num_classes=2):
    loss = 0
    loss += focal_loss(out[0], label, num_classes=2)
    loss += focal_loss(out[1], label, num_classes=2)
    loss += focal_loss(out[2], label, num_classes=2)
    loss += focal_loss(out[3], label, num_classes=2)
    loss += focal_loss(out[4], label, num_classes=2)
    return loss
    
def dice_mertic(predict,label,num_classes=2):
    predict = fluid.layers.transpose(predict,perm=[0,2,3,1])
    predict = fluid.layers.reshape(predict,shape=[-1,num_classes])
    predict = fluid.layers.softmax(predict)
    label = paddle.cast(label,'int64')
    label = fluid.layers.reshape(label,shape=[-1,1])
    dice_loss = fluid.layers.dice_loss(predict,label)
    return 1-dice_loss




