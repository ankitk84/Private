import torch
from torch import nn
from btlbo_unet.cblock import CBLOCK
from btlbo_unet.dag import DAGBLOCK
from btlbo_unet.attention import AttBlock1, AttBlock2, AttBlock3, AttBlock4


# from btlbo_unet.dagblock import DAGBLOCK
class BTLBOUNet22(nn.Module):
    def __init__(self, particle):
        super().__init__()
        self.particle = particle
        self.ffs = 32#16
        
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.convtrans1 = nn.ConvTranspose2d(self.ffs*8,self.ffs*8,kernel_size=2,stride=2)
        self.convtrans2 = nn.ConvTranspose2d(self.ffs*4,self.ffs*4,kernel_size=2,stride=2)
        self.convtrans3 = nn.ConvTranspose2d(self.ffs*2,self.ffs*2,kernel_size=2,stride=2)
        self.convtrans4 = nn.ConvTranspose2d(self.ffs,self.ffs,kernel_size=2,stride=2)
        
        self.upsampl = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.drop = nn.Dropout(p=0.3)
        
        self.cb1 = DAGBLOCK(self.ffs, particle[0:15])
        self.cb2 = DAGBLOCK(self.ffs*2, particle[15:30])
        self.cb3 = DAGBLOCK(self.ffs*4, particle[30:45])
        self.cb4 = DAGBLOCK(self.ffs*8, particle[45:60])        
        self.cb5 = DAGBLOCK(self.ffs*16, particle[60:75])        
        self.cb6 = DAGBLOCK(self.ffs*8, particle[75:90])
        self.cb7 = DAGBLOCK(self.ffs*4, particle[90:105])
        self.cb8 = DAGBLOCK(self.ffs*2, particle[105:120])
        self.cb9 = DAGBLOCK(self.ffs, particle[120:135])
        
        self.c1 = nn.Conv2d(3, self.ffs, 1, stride=1, padding='same')
        self.c2 = nn.Conv2d(self.ffs, self.ffs*2, 1, stride=1, padding='same')
        self.c3 = nn.Conv2d(self.ffs*2, self.ffs*4, 1, stride=1, padding='same')
        self.c4 = nn.Conv2d(self.ffs*4, self.ffs*8, 1, stride=1, padding='same')
        self.c5 = nn.Conv2d(self.ffs*8, self.ffs*16, 1, stride=1, padding='same')
        self.c6 = nn.Conv2d(self.ffs*16, self.ffs*8, 1, stride=1, padding='same')
        self.c7 = nn.Conv2d(self.ffs*8, self.ffs*4, 1, stride=1, padding='same')
        self.c8 = nn.Conv2d(self.ffs*4, self.ffs*2, 1, stride=1, padding='same')
        self.c9 = nn.Conv2d(self.ffs*2, self.ffs, 1, stride=1, padding='same')
        self.c10 = nn.Conv2d(self.ffs, 1, 1, stride=1, padding='same')
        self.sig = nn.Sigmoid() 
        
        self.ab11 = AttBlock1(self.ffs*8)
        self.ab12 = AttBlock2(self.ffs*8)
        self.ab13 = AttBlock3(self.ffs*8)
        self.ab14 = AttBlock4(self.ffs*8)
        
        self.ab21 = AttBlock1(self.ffs*4)
        self.ab22 = AttBlock2(self.ffs*4)
        self.ab23 = AttBlock3(self.ffs*4)
        self.ab24 = AttBlock4(self.ffs*4)
        
        self.ab31 = AttBlock1(self.ffs*2)
        self.ab32 = AttBlock2(self.ffs*2)
        self.ab33 = AttBlock3(self.ffs*2)
        self.ab34 = AttBlock4(self.ffs*2)
        
        self.ab41 = AttBlock1(self.ffs)
        self.ab42 = AttBlock2(self.ffs)
        self.ab43 = AttBlock3(self.ffs)
        self.ab44 = AttBlock4(self.ffs)
      
        
    def forward(self, x):
        n1 = self.c1(x)
        b1 = self.cb1(n1)
        if self.particle[135] == 0:
            p1 = self.maxpool(b1)
        else:
            p1 = self.avgpool(b1)
#         print("after p1")
#         print(p1.shape)
        
        n2 = self.c2(p1)
        b2 = self.cb2(n2)
#         print("after b2")
#         print(b2.shape)
        if self.particle[135] == 0:
            p2 = self.maxpool(b2)
        else:
            p2 = self.avgpool(b2)
#         print("after p2")
#         print(p2.shape)
        
        n3 = self.c3(p2)
        b3 = self.cb3(n3)
#         print("after b3")
#         print(b3.shape)
        if self.particle[135] == 0:
            p3 = self.maxpool(b3)
        else:
            p3 = self.avgpool(b3)
        
        n4 = self.c4(p3)
        b4 = self.cb4(n4)
#         print("after b4")
#         print(b4.shape)
        if self.particle[135] == 0:
            p4 = self.maxpool(b4)
        else:
            p4 = self.avgpool(b4)
        
        n5 = self.c5(p4)
        b5 = self.cb5(n5)
        n6 = self.c6(b5)
        # b5 = self.drop(b5)
#         print("after b5")
#         print(b5.shape)

        if self.particle[136] == 0:
            # print('upsampl')
            up1 = self.upsampl(n6)
#             up1 = self.upsampl1(up1)
        else:
            # print("transpose")
            up1 = self.convtrans1(n6)
#         print("up1")
#         print(up1.shape)
        if self.particle[138] == 0 and self.particle[139] == 0 :
            merge1 = self.ab11(up1,b4)
        elif self.particle[138] == 0 and self.particle[139] == 1:
            merge1 = self.ab12(up1,b4)
        elif self.particle[138] == 1 and self.particle[139] == 0 :
            merge1 = self.ab13(up1,b4)
        elif self.particle[138] == 1 and self.particle[139] == 1 :
            merge1 = self.ab14(up1,b4)
            # merge1 = torch.add(up1, b4)
    
        if self.particle[137] == 0:
            # print('drop')
            merge1 = self.drop(merge1)
        
        
        b6 = self.cb6(merge1)
        n7 = self.c7(b6)
#         print("after b6")
#         print(b6.shape)
        if self.particle[136] == 0:
            up2 = self.upsampl(n7)
#             up2 = self.upsampl2(up2)
        else:
            up2 = self.convtrans2(n7)
#         print("up2")
#         print(up2.shape)
        if self.particle[138] == 0 and self.particle[139] == 0 :
            merge2 = self.ab21(up2, b3)
        elif self.particle[138] == 0 and self.particle[139] == 1 :
            merge2 = self.ab22(up2, b3)
        elif self.particle[138] == 1 and self.particle[139] == 0 :
            merge2 = self.ab23(up2, b3)
        elif self.particle[138] == 1 and self.particle[139] == 1 :
            merge2 = self.ab24(up2, b3)
        # merge2 = torch.add(up2, b3)
#         if self.particle[137] == 0:
#             merge2 = self.drop(merge2)
        
        b7 = self.cb7(merge2)
        n8 = self.c8(b7)
#         print("after b7")
#         print(b7.shape)
        if self.particle[136] == 0:
            up3 = self.upsampl(n8)
#             up3 = self.upsampl3(up3)
        else:
            up3 = self.convtrans3(n8)
#         print("up3")
#         print(up3.shape)
        if self.particle[138] == 0 and self.particle[139] == 0 :
            merge3 = self.ab31(up3, b2)
        elif self.particle[138] == 0 and self.particle[139] == 1 :
            merge3 = self.ab32(up3, b2)
        elif self.particle[138] == 1 and self.particle[139] == 0 :
            merge3 = self.ab33(up3, b2)
        elif self.particle[138] == 1 and self.particle[139] == 1 :
            merge3 = self.ab34(up3, b2)
        # merge3 = torch.add(up3, b2)
#         if self.particle[137] == 0:
#             merge3 = self.drop(merge3)
        
        b8 = self.cb8(merge3)
        n9 = self.c9(b8)
#         print("after b8")
#         print(b8.shape)
        if self.particle[136] == 0:
            up4 = self.upsampl(n9)
#             up4 = self.upsampl4(up4)
        else:
            up4 = self.convtrans4(n9)
#         print("up4")
#         print(up4.shape)
        if self.particle[138] == 0 and self.particle[139] == 0 :
            merge4 = self.ab41(up4, b1)
        elif self.particle[138] == 0 and self.particle[139] == 1 :
            merge4 = self.ab42(up4, b1)
        elif self.particle[138] == 1 and self.particle[139] == 0 :
            merge4 = self.ab43(up4, b1)
        elif self.particle[138] == 1 and self.particle[139] == 1 :
            merge4 = self.ab44(up4, b1)
        # merge4 = torch.add(up4, b1)
#         if self.particle[137] == 0:
#             merge4 = self.drop(merge4)
        
        b9 = self.cb9(merge4)
        n10 = self.c10(b9)
        n10 = self.sig(n10)
#         print("after b9")
#         print(b9.shape)

        # net = self.last_conv(b9+x)
#         net = self.last_layer(net)
        return n10
