import paddle
import paddle.nn as nn
import numpy as np
from paddle import fluid
import paddle.nn.functional as F


class Separable_base_conv(nn.Layer):
    def __init__(self, input_channels, num_filters):
        super(Separable_base_conv,self).__init__()
        self.pointwise_conv = nn.Conv2D(in_channels=input_channels, out_channels=num_filters, kernel_size=1,stride=1)
        self.depthwise_conv = nn.Conv2D(in_channels=num_filters, out_channels=num_filters, kernel_size=3,stride=1, groups=num_filters,padding=1)
    def forward(self, inputs):
        x = self.pointwise_conv(inputs)
        #print(x.shape)
        x = self.depthwise_conv(x)
        #print(x.shape)
        return x
#原始的Xception没有在conv后加BatchNorm,这里加上看看效果
class Norm_conv_block1(nn.Layer):
    def __init__(self, input_channels=3, out_channels=64):
        super(Norm_conv_block1, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2D(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm(32, act="relu"),
            nn.Conv2D(32, out_channels, kernel_size=3, stride = 1, padding=1),
            nn.BatchNorm(out_channels, act="relu")
        )
    def forward(self, inputs):
        x = self.layers(inputs)
        return x

class Sep_conv_block1(nn.Layer):
    def __init__(self, input_channels, num_filters):
        super(Sep_conv_block1, self).__init__()
        
        self.short = nn.Conv2D(in_channels=input_channels, out_channels=num_filters, kernel_size=1, stride=2, padding=0)
        self.SepConv1 = Separable_base_conv(input_channels, num_filters)
        self.SepConv2 = Separable_base_conv(num_filters, num_filters)
        self.Relu = nn.ReLU()
        self.pool = nn.MaxPool2D(kernel_size=3, stride=2,padding=1)
    
    def forward(self, inputs):
        short_con = self.short(inputs)
        x = self.SepConv1(inputs)
        x = self.Relu(x)
        x = self.SepConv2(x)
        x = self.pool(x)
        return paddle.add(short_con, x)
class Sep_conv_block2(nn.Layer):
    def __init__(self, input_channels, num_filters):
        super(Sep_conv_block2, self).__init__()
        self.short = nn.Conv2D(in_channels=input_channels, out_channels=num_filters, kernel_size=1, stride=2,padding=0)
        self.Relu = nn.ReLU()
        self.SepConv1 = Separable_base_conv(input_channels, num_filters)
        self.SepConv2 = Separable_base_conv(num_filters,num_filters)
        self.pool = nn.MaxPool2D(kernel_size=3, stride=2,padding=1)
    def forward(self, inputs):
        short_con = self.short(inputs)
        x = self.Relu(inputs)
        x = self.SepConv1(x)
        x = self.Relu(x)
        x = self.SepConv2(x)
        x = self.pool(x)
        return paddle.add(short_con, x)


class Sep_conv_block3(nn.Layer):
    def __init__(self, input_channels, num_filters):
        super(Sep_conv_block3, self).__init__()
        self.short = nn.Conv2D(in_channels = input_channels, out_channels=num_filters, kernel_size=1,stride=2,padding=0)
        self.Relu = nn.ReLU()
        self.SepConv1 = Separable_base_conv(input_channels, num_filters)
        self.SepConv2 = Separable_base_conv(num_filters, num_filters)
        self.pool = nn.MaxPool2D(kernel_size=3, stride=2,padding=1)
    def forward(self, inputs):
        short_con = self.short(inputs)
        x = self.Relu(inputs)
        x = self.SepConv1(x)
        x = self.Relu(x)
        x = self.SepConv2(x)
        x = self.pool(x)
        return paddle.add(short_con, x)
        

#池化快都拿出来
class XceptUnet(nn.Layer):
    def __init__(self, input_channels=3, num_classes=2):
        super(XceptUnet,self).__init__()
        self.input_channels = input_channels
        self.out_channels = num_classes
        
        self.Norm_conv_block1 = Norm_conv_block1(input_channels, 64)
        self.Sep_conv_block1 = Sep_conv_block1(64,128)
        self.Sep_conv_block2 = Sep_conv_block2(128,256)
        self.Sep_conv_block3 = Sep_conv_block3(256,728)
        feature_dim = 728
        self.deeplabhead = DeeplabHead(feature_dim, num_classes)

    def forward(self, inputs):
        #print('________')
        #print(inputs.shape) # 1 3 512 512
        x = self.Norm_conv_block1(inputs)
        #print(x.shape) # 1 64 256 256
        x = self.Sep_conv_block1(x)
        #print(x.shape) # 1 128 128 128
        x = self.Sep_conv_block2(x)
        #print(x.shape) # 1 256 64 64
        x = self.Sep_conv_block3(x)
        #print(x.shape) # 1 728 32 32
        
        return x

paddle.summary(XceptUnet(), (1, 3, 512, 512))
