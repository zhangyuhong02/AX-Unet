import paddle
import paddle.nn as nn
#from paddle.vision.models import resnet50
#from paddle.vision.models.resnet import BottleneckBlock
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
class sub_sample_encoder(nn.Layer):
    def __init__(self, input_channel, num_filters):
        super(sub_sample_encoder, self).__init__()
        self.Conv_Block = DoubleConv(input_channel, num_filters)
        self.MaxPool = nn.MaxPool2D(kernel_size=2, stride=2)
    def forward(self, inputs):
        x = self.Conv_Block(inputs)
        x = self.MaxPool(x)
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
    def __init__(self, num_channels, num_classes):
        super(DeeplabHead, self).__init__()
        self.head = nn.Sequential(
                            ASPPModule(num_channels, 256, [12, 24, 36]),
                            nn.Conv2D(256, 512, 3, padding=1),
                            nn.BatchNorm(512, act="relu"),
                            nn.Conv2D(512, num_channels, 1)        
                            )
    def forward(self, inputs):
        x = self.head(inputs)
        return x
class sub_sample_encoder(nn.Layer):
    def __init__(self, input_channel, num_filters):
        super(sub_sample_encoder, self).__init__()
        self.Conv_Block = DoubleConv(input_channel, num_filters)
        self.MaxPool = nn.MaxPool2D(kernel_size=2, stride=2)
    def forward(self, inputs):
        x = self.Conv_Block(inputs)
        x = self.MaxPool(x)
        return x
class DoubleConv(nn.Layer):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2D(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(out_channel, act="relu"),
            nn.Conv2D(out_channel, out_channel, kernel_size=3, stride = 1, padding=1),
            nn.BatchNorm(out_channel, act="relu")
        )

    def forward(self, inputs):
        x = self.layers(inputs)
        return x
class Deeplabv3(nn.Layer):
    def __init__(self, num_classes=2, in_channels=3):
        super(Deeplabv3, self).__init__()
        # resnet50 3->2048
        # resnet50 四层layers = [3 4 6 3]
        # 调用resnet.py模块，空洞卷积[2 4 8 16]
        num_channels = [64,128,256,512,1024]
        num_filters = 128   #Deeplabv3为256
        self.features = nn.LayerList()
        
        ##Unet编码模块
        self.enco1 = sub_sample_encoder(in_channels, num_channels[0])
        self.enco2 = sub_sample_encoder(num_channels[0], num_channels[1])
        self.enco3 = sub_sample_encoder(num_channels[1], num_channels[2])
        self.enco4 = sub_sample_encoder(num_channels[2], num_channels[3])
        self.enco5 = DoubleConv(num_channels[3], num_channels[4])
        self.c1 = DoubleConv(in_channels, 64)
        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.c2 = DoubleConv(64, 128)
        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.c3 = DoubleConv(128, 256)
        self.maxpool3 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.c4 = DoubleConv(256, 512)
        self.maxpool4 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.c5 = DoubleConv(512, 1024)
        
        feature_dim = 1024      #输出层通道1024
        self.deeplabhead = DeeplabHead(feature_dim, num_classes)

        self.up6 = nn.Conv2DTranspose(1024, 512, 2, stride=2)
        self.c6 = DoubleConv(1024, 512)
        self.up7 = nn.Conv2DTranspose(512, 256, 2, stride=2)
        self.c7 = DoubleConv(512, 256)
        self.up8 = nn.Conv2DTranspose(256, 128, 2, stride=2)
        self.c8 = DoubleConv(256, 128)
        self.up9 = nn.Conv2DTranspose(128, 64, 2, stride=2)
        self.c9 = DoubleConv(128, 64)

        self.c10 = nn.Conv2D(64, num_classes, 1)

        self.merge6_conv = nn.Conv2D(1024,1024,kernel_size=3,padding=1,groups=2)
        self.merge7_conv = nn.Conv2D(512,512,kernel_size=3,padding=1,groups=2)
        self.merge8_conv = nn.Conv2D(256,256,kernel_size=3,padding=1,groups=2)
        self.merge9_conv = nn.Conv2D(128,128,kernel_size=3,padding=1,groups=2)

    def forward(self, inputs):
        #x1 = self.enco1(inputs) #n,64,256,256
        #x2 = self.enco2(x1) #n,128,128,128
        #x3 = self.enco3(x2) #n,256,64,64
        #x4 = self.enco4(x3) #n,512,32,32
        #x5 = self.enco5(x4) #n,1024,32,32
        c1 = self.c1(inputs)
        p1 = self.maxpool1(c1)
        c2 = self.c2(p1)
        p2 = self.maxpool2(c2)
        c3 = self.c3(p2)
        p4 = self.maxpool3(c3)
        c4 = self.c4(p4)
        p5 = self.maxpool4(c4)
        c5 = self.c5(p5)
        c5= self.deeplabhead(c5)#ASPP模块进行分类
        # 恢复原图尺寸
        up6 = self.up6(c5)
        merge6 = fluid.layers.concat([up6, c4], axis=1)
        #print(merge6.shape)
        merge6 = self.merge6_conv(merge6)
        c6 = self.c6(merge6)
        up7 = self.up7(c6)
        merge7 = fluid.layers.concat([up7, c3], axis=1)
        #print(merge7.shape)
        merge7 = self.merge7_conv(merge7)
        c7= self.c7(merge7)
        up8 = self.up8(c7)
        merge8 = fluid.layers.concat([up8, c2], axis=1)
        #print(merge8.shape)
        merge8 = self.merge8_conv(merge8)
        c8 = self.c8(merge8)
        up9 = self.up9(c8)
        merge9 = fluid.layers.concat([up9 , c1], axis=1)
        merge9 = self.merge9_conv(merge9)
        c9 = self.c9(merge9)
        c10 = self.c10(c9)
        out = fluid.layers.logsigmoid(c10)
        
        #x = paddle.nn.functional.interpolate(x=x, size=inputs.shape[2::], mode='bilinear', align_corners=True)
        return out

paddle.summary(Deeplabv3(), (1, 3, 512, 512))