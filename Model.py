# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:18:12 2022

@author: Yuhong
"""
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Conv2DTranspose
from paddle.fluid.dygraph.base import to_variable
import paddle.nn as nn
import numpy as np
from paddle import fluid
import paddle.nn.functional as F
class ASPPPooling(nn.Layer):
    def __init__(self,num_channels,num_filters):
        super(ASPPPooling,self).__init__()
        self.adaptive_pool = nn.AdaptiveMaxPool2D(output_size=3)
        self.features = nn.Sequential(
                                        nn.Conv2D(num_channels, num_filters,1),
                                        nn.BatchNorm(num_filters, act="relu")
                                    )
    def forward(self, inputs):
        n1, c1, h1, w1 = inputs.shape
        x = self.adaptive_pool(inputs)
        x = self.features(x)
        x = nn.functional.interpolate(x, (h1, w1), align_corners=False)
        return x
class ASPPConv(nn.Layer):
    def __init__(self,num_channels,num_filters,dilations):
        super(ASPPConv,self).__init__()
        self.asppconv = nn.Sequential(
                            nn.Conv2D(num_channels,num_filters,3,padding=dilations,dilation=dilations),
                            nn.BatchNorm(num_filters, act="relu")
                            )
    def forward(self,inputs):
        x = self.asppconv(inputs)
        return x

#ASPP模块最大的特点是使用了空洞卷积来增大感受野
class ASPPModule(nn.Layer):
    def __init__(self, num_channels, num_filters, rates):
        super(ASPPModule, self).__init__()
        self.features = nn.LayerList()
        #Layer1:1x1卷积
        self.features.append(nn.Sequential(
                                        nn.Conv2D(num_channels, num_filters,1),
                                        nn.BatchNorm(num_filters, act="relu")
                                          )
                            )
        #Layer2:三个空洞卷积模块
        for r in rates:
            self.features.append(ASPPConv(num_channels, num_filters, r))
        #Layer3:适应输出尺寸的池化
        self.features.append(ASPPPooling(num_channels, num_filters))
        #Layer4:将前几层layer concat之后的统一操作
        self.project  = nn.Sequential(
                                    nn.Conv2D(num_filters*(2+len(rates)), num_filters, 1),#TODO
                                    nn.BatchNorm(num_filters, act="relu")
                                     )
    def forward(self, inputs):
        out = []
        for op in self.features:
            out.append(op(inputs))
        x = paddle.concat(x=out,axis=1)
        x = self.project(x)
        return x
#Upgraded ASPP Module作为DeepLabv3的head
class DeeplabHead(nn.Layer):
    def __init__(self, num_channels):
        super(DeeplabHead, self).__init__()
        self.head = nn.Sequential(
                            ASPPModule(num_channels, 256, [2, 4, 6]),
                            nn.Conv2D(256, 512, 3, padding=1),
                            nn.BatchNorm(512, act="relu"),
                            nn.Conv2D(512, num_channels, 1)        
                            )
    def forward(self, inputs):
        x = self.head(inputs)
        return x

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

class DoubleSepConv(nn.Layer):
    def __init__(self, in_channel, out_channel):
        super(DoubleSepConv, self).__init__()
        self.layers = nn.Sequential(
            Separable_base_conv(in_channel, out_channel),
            nn.BatchNorm(out_channel,act='relu'),
            nn.Conv2D(out_channel, out_channel, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm(out_channel,act='relu')
        )
    def forward(self,inputs):
        x = self.layers(inputs)
        return x

class DoubleConv(fluid.dygraph.Layer):
    def __init__(self, inchannel, outchannel):
        super(DoubleConv, self).__init__()
        self.layers = fluid.dygraph.Sequential(
            Conv2D(inchannel, outchannel, filter_size=3,stride=1, padding=1),
            fluid.BatchNorm(outchannel,act='relu'),
            Conv2D(outchannel, outchannel, filter_size=3, stride=1,padding=1),
            fluid.BatchNorm(outchannel, act='relu'),
        )

    def forward(self, x):
        x = self.layers(x)
        return x




class Unet2(fluid.dygraph.Layer):
    def __init__(self, input, out):
        super(Unet2, self).__init__()

        self.c1 = DoubleConv(input, 64)
        self.maxpool1 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c2 = DoubleSepConv(64, 128)
        self.maxpool2 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c3 = DoubleSepConv(128, 256)
        self.maxpool3 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c4 = DoubleSepConv(256, 512)
        self.maxpool4 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c5 = DoubleSepConv(512, 1024)

        feature_dim = 1024      #输出层通道1024
        self.deeplabhead = DeeplabHead(feature_dim)

        self.up6 = Conv2DTranspose(1024, 512, 2, stride=2)
        self.c6 = DoubleSepConv(1024, 512)
        self.up7 = Conv2DTranspose(512, 256, 2, stride=2)
        self.c7 = DoubleSepConv(512, 256)
        self.up8 = Conv2DTranspose(256, 128, 2, stride=2)
        self.c8 = DoubleSepConv(256, 128)
        self.up9 = Conv2DTranspose(128, 64, 2, stride=2)
        self.c9 = DoubleSepConv( 128, 64)

        self.c10 = Conv2D(64, out, 1)
        
        self.short1 = nn.Conv2D(64,128,kernel_size=1,stride=2,padding=0)
        self.short2 = nn.Conv2D(128, 256, kernel_size=1,stride=2,padding=0)
        self.short3 = nn.Conv2D(256, 512, kernel_size=1, stride=2, padding=0)
        self.short4 = nn.Conv2D(512, 1024, kernel_size=1, stride=2, padding=0)


    def forward(self, inputs):
        c1 = self.c1(inputs)
        p1 = self.maxpool1(c1)
        c2 = self.c2(p1)
        p2 = self.maxpool2(c2)

        c3 = self.c3(p2)
        p4 = self.maxpool3(c3)        

        c4 = self.c4(p4)
        p5 = self.maxpool4(c4)
        c5 = self.c5(p5)
        
        c5 = self.deeplabhead(c5)


        up6 = self.up6(c5)
        merge6 = fluid.layers.concat([up6, c4], axis=1)
        c6 = self.c6(merge6)
        up7 = self.up7(c6)
        merge7 = fluid.layers.concat([up7, c3], axis=1)
        c7= self.c7(merge7)
        up8 = self.up8(c7)
        merge8 = fluid.layers.concat([up8, c2], axis=1)
        c8 = self.c8(merge8)
        up9 = self.up9(c8)
        merge9 = fluid.layers.concat([up9 , c1], axis=1)
        c9 = self.c9(merge9)

        c10 = self.c10(c9)
        out = fluid.layers.logsigmoid(c10)
        return out        

print(paddle.summary(Unet2(3,2), (1, 3, 512, 512)))


class ASPPPooling(nn.Layer):
    def __init__(self,num_channels,num_filters):
        super(ASPPPooling,self).__init__()
        self.adaptive_pool = nn.AdaptiveMaxPool2D(output_size=3)
        self.features = nn.Sequential(
                                        nn.Conv2D(num_channels, num_filters,1),
                                        nn.BatchNorm(num_filters, act="relu")
                                    )
    def forward(self, inputs):
        n1, c1, h1, w1 = inputs.shape
        x = self.adaptive_pool(inputs)
        x = self.features(x)
        x = nn.functional.interpolate(x, (h1, w1), align_corners=False)
        return x
class ASPPConv(nn.Layer):
    def __init__(self,num_channels,num_filters,dilations):
        super(ASPPConv,self).__init__()
        self.asppconv = nn.Sequential(
                            nn.Conv2D(num_channels,num_filters,3,padding=dilations,dilation=dilations),
                            nn.BatchNorm(num_filters, act="relu")
                            )
    def forward(self,inputs):
        x = self.asppconv(inputs)
        return x

#ASPP模块最大的特点是使用了空洞卷积来增大感受野
class ASPPModule(nn.Layer):
    def __init__(self, num_channels, num_filters, rates):
        super(ASPPModule, self).__init__()
        self.features = nn.LayerList()
        #Layer1:1x1卷积
        self.features.append(nn.Sequential(
                                        nn.Conv2D(num_channels, num_filters,1),
                                        nn.BatchNorm(num_filters, act="relu")
                                          )
                            )
        #Layer2:三个空洞卷积模块
        for r in rates:
            self.features.append(ASPPConv(num_channels, num_filters, r))
        #Layer3:适应输出尺寸的池化
        self.features.append(ASPPPooling(num_channels, num_filters))
        #Layer4:将前几层layer concat之后的统一操作
        self.project  = nn.Sequential(
                                    nn.Conv2D(num_filters*(2+len(rates)), num_filters, 1),#TODO
                                    nn.BatchNorm(num_filters, act="relu")
                                     )
    def forward(self, inputs):
        out = []
        for op in self.features:
            out.append(op(inputs))
        x = paddle.concat(x=out,axis=1)
        x = self.project(x)
        return x
#Upgraded ASPP Module作为DeepLabv3的head
class DeeplabHead(nn.Layer):
    def __init__(self, num_channels):
        super(DeeplabHead, self).__init__()
        self.head = nn.Sequential(
                            ASPPModule(num_channels, 256, [2, 4, 6]),
                            nn.Conv2D(256, 512, 3, padding=1),
                            nn.BatchNorm(512, act="relu"),
                            nn.Conv2D(512, num_channels, 1)        
                            )
    def forward(self, inputs):
        x = self.head(inputs)
        return x

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

class DoubleSepConv(nn.Layer):
    def __init__(self, in_channel, out_channel):
        super(DoubleSepConv, self).__init__()
        self.layers = nn.Sequential(
            Separable_base_conv(in_channel, out_channel),
            nn.BatchNorm(out_channel,act='relu'),
            nn.Conv2D(out_channel, out_channel, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm(out_channel,act='relu')
        )
    def forward(self,inputs):
        x = self.layers(inputs)
        return x

class DoubleConv(fluid.dygraph.Layer):
    def __init__(self, inchannel, outchannel):
        super(DoubleConv, self).__init__()
        self.layers = fluid.dygraph.Sequential(
            Conv2D(inchannel, outchannel, filter_size=3,stride=1, padding=1),
            fluid.BatchNorm(outchannel,act='relu'),
            Conv2D(outchannel, outchannel, filter_size=3, stride=1,padding=1),
            fluid.BatchNorm(outchannel, act='relu'),
        )

    def forward(self, x):
        x = self.layers(x)
        return x



#增加残差块
class Unet2(fluid.dygraph.Layer):
    def __init__(self, input, out):
        super(Unet2, self).__init__()

        self.c1 = DoubleConv(input, 64)
        self.maxpool1 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c2 = DoubleSepConv(64, 128)
        self.maxpool2 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c3 = DoubleSepConv(128, 256)
        self.maxpool3 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c4 = DoubleSepConv(256, 512)
        self.maxpool4 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c5 = DoubleSepConv(512, 1024)

        feature_dim = 1024      #输出层通道1024
        self.deeplabhead = DeeplabHead(feature_dim)

        self.up6 = Conv2DTranspose(1024, 512, 2, stride=2)
        self.c6 = DoubleSepConv(1024, 512)
        self.up7 = Conv2DTranspose(512, 256, 2, stride=2)
        self.c7 = DoubleSepConv(512, 256)
        self.up8 = Conv2DTranspose(256, 128, 2, stride=2)
        self.c8 = DoubleSepConv(256, 128)
        self.up9 = Conv2DTranspose(128, 64, 2, stride=2)
        self.c9 = DoubleSepConv(128, 64)

        self.c10 = Conv2D(64, out, 1)
        
        self.short1 = nn.Conv2D(64,128,kernel_size=1,stride=2,padding=0)
        self.short2 = nn.Conv2D(128, 256, kernel_size=1,stride=2,padding=0)
        self.short3 = nn.Conv2D(256, 512, kernel_size=1, stride=2, padding=0)
        self.short4 = nn.Conv2D(512, 1024, kernel_size=1, stride=2, padding=0)

#s2没有参与相加，我们认为在浅层的特征提取并没有损失太多的细节特征
    def forward(self, inputs):
        c1 = self.c1(inputs)
        p1 = self.maxpool1(c1)
        
        s1 = self.short1(p1)
        
        c2 = self.c2(p1)
        p2 = self.maxpool2(c2)
        

        #merge2 = fluid.layers.concat([s1, p2], axis=1)
        sum2 = paddle.add(s1, p2)
        s2 = self.short2(sum2)
        

        c3 = self.c3(p2)
        p4 = self.maxpool3(c3)        

        #merge3 = fluid.layers.concat([s2, p4],axis=1)
        sum3 = paddle.add(s2, p4)
        s3 = self.short3(sum3)

        c4 = self.c4(sum3)
        p5 = self.maxpool4(c4)
        
        #merge4 = fluid.layers.concat([s3, p5], axis=1)
        sum4 = paddle.add(s3, p5)
        
        c5 = self.c5(sum4)
        
        c5 = self.deeplabhead(c5)
        up6 = self.up6(c5)
        merge6 = fluid.layers.concat([up6, c4], axis=1)
        c6 = self.c6(merge6)
        up7 = self.up7(c6)
        merge7 = fluid.layers.concat([up7, c3], axis=1)
        c7= self.c7(merge7)
        up8 = self.up8(c7)
        merge8 = fluid.layers.concat([up8, c2], axis=1)
        c8 = self.c8(merge8)
        up9 = self.up9(c8)
        merge9 = fluid.layers.concat([up9 , c1], axis=1)
        c9 = self.c9(merge9)

        c10 = self.c10(c9)
        out = fluid.layers.logsigmoid(c10)
        return out        





