import torch
import torch.nn as nn
import torchvision.models as models
from function import AdaIN,get_mean_std

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#训练设备



#############vgg编码器##############
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


############定义镜像解码器#############
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        #layer1
        self.pre11=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1)
        self.relu11=nn.ReLU(inplace=True)
        self.up1=nn.Upsample(scale_factor=2, mode='nearest')
        #layer2
        self.pre21=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1)
        self.relu21=nn.ReLU(inplace=True)
        self.pre22=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1)
        self.relu22=nn.ReLU(inplace=True)
        self.pre23=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1)
        self.relu23=nn.ReLU(inplace=True)
        self.pre24=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24=nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1)
        self.relu24=nn.ReLU(inplace=True)

        self.up2=nn.Upsample(scale_factor=2, mode='nearest')
        #layer3
        self.pre31=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv31=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1)
        self.relu31=nn.ReLU(inplace=True)
        self.pre32=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv32=nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1)
        self.relu32=nn.ReLU(inplace=True)
        self.up3=nn.Upsample(scale_factor=2, mode='nearest')
        #layer4
        self.pre41=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv41=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.relu41=nn.ReLU(inplace=True)
        self.pre42=nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv42=nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1)
        self.relu42=nn.ReLU(inplace=True)

    def forward(self, x):
        x=self.pre11(x)
        x=self.relu11(self.conv11(x))
        x=self.up1(x)
        x=self.pre21(x)
        x=self.relu21(self.conv21(x))
        x=self.pre22(x)
        x=self.relu22(self.conv22(x))
        x=self.pre23(x)
        x=self.relu23(self.conv23(x))
        x=self.pre24(x)
        x=self.relu24(self.conv24(x))
        x=self.up2(x)
        x=self.pre31(x)
        x=self.relu31(self.conv31(x))
        x=self.pre32(x)
        x=self.relu32(self.conv32(x))
        x=self.up3(x)
        x=self.pre41(x)
        x=self.relu41(self.conv41(x))
        x=self.pre42(x)
        x=self.conv42(x)
        return x

##############定义前传网络#############
class FPnet(nn.Module):
    def __init__(self,decoder,test=False):
        super(FPnet, self).__init__()
        self.encoder = vgg
        self.decoder = decoder
        self.mseloss = nn.MSELoss()
        self.encoder.load_state_dict(torch.load('static/vgg_normalised.pth'))
        if test: self.encoder=self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False#对encoder不进行梯度下降

    def encode(self, x,layer=0):#基于vgg19编码提取特定层的特征图
        '''分别提取relu1-1,relu2-1,relu3-1,relu4-1的特征，默认提取relu4-1'''
        layers=[31,4,11,18,31]
        for i in range(layers[layer]):
            x = self.encoder[i](x)
        return x

    def get_content_loss(self,feature1,feature2):
        '''计算内容损失'''
        dis=self.mseloss(feature1,feature2)
        return dis

    def get_style_loss(self, input, target):
        '''计算样式损失'''
        input_mean, input_std = get_mean_std(input)
        target_mean, target_std = get_mean_std(target)
        return self.mseloss(input_mean, target_mean) + self.mseloss(input_std, target_std)

    def forward(self,content,style,alpha=1.0,lamda=10.0,require_loss=True):
        '''一次前传计算损失'''
        if True:
            # content=content.to(device)
            # style=style.to(device)
            content=content.to(torch.device("cpu"))
            style=style.to(torch.device("cpu"))

        style_features=self.encode(style)
        content_features=self.encode(content)

        t=AdaIN(content_features,style_features)
        t=alpha*t+(1-alpha)*content_features

        output=self.decoder(t)
        if not require_loss:return output

        out_features=self.encode(output)
        content_loss=self.get_content_loss(out_features,t)
        style_loss=self.get_style_loss(out_features,style_features)

        for i in range(1,4):
            style_loss+=self.get_style_loss(self.encode(output,layer=i),self.encode(style,layer=i))

        return content_loss + style_loss*lamda , output
