import torch
from torch import nn

class AttBlock1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, stride=1, padding='same')
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, up, b):
        n1 = self.conv(up)
        n2 = self.conv(b)
        n3 = self.rel(n1+n2)
        n4 = self.conv(n3)
        n5 = self.sig(n4)
        n6 = n5*up       
        return n6
    
class AttBlock2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, stride=1, padding='same')
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, up, b):
        n1 = self.conv(up)
        n2 = self.conv(b)
        n3 = self.sig(n1*n2)
        n4 = self.conv(n3)   
        n5 = self.rel(n4+up)   
        return n5
    
class AttBlock3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, stride=1, padding='same')
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, up, b):
        n1 = self.conv(up)
        n2 = self.conv(b)
        n3 = self.rel(n1+n2)
        n4 = self.conv(n3)
        n5 = self.sig(n4)
        n6 = n5*b       
        return n6
    
class AttBlock4(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, stride=1, padding='same')
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, up, b):
        n1 = self.conv(up)
        n2 = self.conv(b)
        n3 = self.sig(n1*n2)
        n4 = self.conv(n3)   
        n5 = self.rel(n4+b)   
        return n5