import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)


class UNetWithClassifier(nn.Module):

    def __init__(self,in_channels=4,out_classes=4):
        super().__init__()

        self.enc1=ConvBlock(in_channels,64)
        self.enc2=ConvBlock(64,128)
        self.enc3=ConvBlock(128,256)
        self.enc4=ConvBlock(256,512)

        self.pool=nn.MaxPool2d(2)

        self.bottleneck=ConvBlock(512,1024)

        self.global_pool=nn.AdaptiveAvgPool2d(1)

        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

        self.up4=nn.ConvTranspose2d(1024,512,2,2)
        self.dec4=ConvBlock(1024,512)

        self.up3=nn.ConvTranspose2d(512,256,2,2)
        self.dec3=ConvBlock(512,256)

        self.up2=nn.ConvTranspose2d(256,128,2,2)
        self.dec2=ConvBlock(256,128)

        self.up1=nn.ConvTranspose2d(128,64,2,2)
        self.dec1=ConvBlock(128,64)

        self.final_conv=nn.Conv2d(64,out_classes,1)

    def forward(self,x):

        s1=self.enc1(x)
        p1=self.pool(s1)

        s2=self.enc2(p1)
        p2=self.pool(s2)

        s3=self.enc3(p2)
        p3=self.pool(s3)

        s4=self.enc4(p3)
        p4=self.pool(s4)

        bn=self.bottleneck(p4)

        cls=self.classifier(self.global_pool(bn))

        d4=self.up4(bn)
        d4=torch.cat([d4,s4],1)
        d4=self.dec4(d4)

        d3=self.up3(d4)
        d3=torch.cat([d3,s3],1)
        d3=self.dec3(d3)

        d2=self.up2(d3)
        d2=torch.cat([d2,s2],1)
        d2=self.dec2(d2)

        d1=self.up1(d2)
        d1=torch.cat([d1,s1],1)
        d1=self.dec1(d1)

        seg=self.final_conv(d1)

        return seg,cls