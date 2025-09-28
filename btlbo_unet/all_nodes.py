from torch import nn

class NODE0(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                nn.Conv2d(inc, outc, 1, stride=1, padding='same'),
                nn.ReLU()
            )                                   

    def forward(self, x):
        n = self.n0(x)
        return n
    

class NODE1(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n1 = nn.Sequential(
               nn.Conv2d(inc, outc, 1, stride=1, padding='same'),
                nn.Mish()
            )                                   

    def forward(self, x):
        n = self.n1(x)
        return n



class NODE2(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n2 = nn.Sequential(
                nn.Conv2d(inc, outc, 1, stride=1, padding='same'),
                nn.InstanceNorm2d(outc)
            )                                   

    def forward(self, x):
        n = self.n2(x)
        return n

class NODE3(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                nn.Conv2d(inc, outc, 1, stride=1, padding='same'),
                nn.BatchNorm2d(outc)
                )                                   

    def forward(self, x):
        n = self.n0(x)
        return n


class NODE4(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n1 = nn.Sequential(
               nn.Conv2d(inc, outc, 3, stride=1, padding='same'),
        nn.ReLU()
            )                                   

    def forward(self, x):
        n = self.n1(x)
        return n



class NODE5(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n2 = nn.Sequential(
                 nn.Conv2d(inc, outc, 3, stride=1, padding='same'),
        nn.Mish()
            )                                   

    def forward(self, x):
        n = self.n2(x)
        return n

class NODE6(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                 nn.Conv2d(inc, outc, 3, stride=1, padding='same'),
        nn.InstanceNorm2d(outc),
        nn.ReLU()
            )                                   

    def forward(self, x):
        n = self.n0(x)
        return n


class NODE7(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n1 = nn.Sequential(
               nn.Conv2d(inc, outc, 3, stride=1, padding='same'),
        nn.InstanceNorm2d(outc),
        nn.Mish()
            )                                   

    def forward(self, x):
        n = self.n1(x)
        return n



class NODE8(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n2 = nn.Sequential(
                nn.Conv2d(inc, outc, 3, stride=1, padding='same'),
        nn.BatchNorm2d(outc),
        nn.ReLU()
            )                                   

    def forward(self, x):
        n = self.n2(x)
        return n

class NODE9(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                 nn.Conv2d(inc, outc, 3, stride=1, padding='same'),
        nn.BatchNorm2d(outc),
        nn.Mish()
            )                                   

    def forward(self, x):
        n = self.n0(x)
        return n

class NODE10(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n1 = nn.Sequential(
              nn.Conv2d(inc, outc, 5, stride=1, padding='same'),
        nn.ReLU()
            )                                   

    def forward(self, x):
        n = self.n1(x)
        return n



class NODE11(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n2 = nn.Sequential(
                 nn.Conv2d(inc, outc, 5, stride=1, padding='same'),
        nn.Mish()
            )                                   

    def forward(self, x):
        n = self.n2(x)
        return n

class NODE12(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                nn.Conv2d(inc, outc, 5, stride=1, padding='same'),
        nn.InstanceNorm2d(outc),
        nn.ReLU()
            )                                   

    def forward(self, x):
        n = self.n0(x)
        return n


class NODE13(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n1 = nn.Sequential(
               nn.Conv2d(inc, outc, 5, stride=1, padding='same'),
        nn.InstanceNorm2d(outc),
        nn.Mish()
            )                                   

    def forward(self, x):
        n = self.n1(x)
        return n



class NODE14(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n2 = nn.Sequential(
                nn.Conv2d(inc, outc, 5, stride=1, padding='same'),
        nn.BatchNorm2d(outc),
        nn.ReLU()
            )                                   

    def forward(self, x):
        n = self.n2(x)
        return n

class NODE15(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                 nn.Conv2d(inc, outc, 5, stride=1, padding='same'),
        nn.BatchNorm2d(outc),
        nn.Mish()
            )                                   

    def forward(self, x):
        n = self.n0(x)
        return n


class NODE16(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n1 = nn.Sequential(
                     nn.ReLU(),
        nn.Conv2d(inc, outc, 1, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n1(x)
        return n



class NODE17(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n2 = nn.Sequential(
                 nn.Mish(),
        nn.Conv2d(inc, outc, 1, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n2(x)
        return n


class NODE18(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                       nn.InstanceNorm2d(inc),
        nn.Conv2d(inc, outc, 1, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n0(x)
        return n


class NODE19(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n1 = nn.Sequential(
               nn.BatchNorm2d(inc),
        nn.Conv2d(inc, outc, 1, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n1(x)
        return n



class NODE20(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n2 = nn.Sequential(
                nn.ReLU(),
        nn.Conv2d(inc, outc, 3, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n2(x)
        return n

class NODE21(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                  nn.Mish(),
        nn.Conv2d(inc, outc, 3, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n0(x)
        return n



class NODE22(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n1 = nn.Sequential(
               nn.InstanceNorm2d(inc),
        nn.ReLU(),
        nn.Conv2d(inc, outc, 3, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n1(x)
        return n



class NODE23(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n2 = nn.Sequential(
                 nn.InstanceNorm2d(inc),
        nn.Mish(),
        nn.Conv2d(inc, outc, 3, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n2(x)
        return n

class NODE24(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                nn.BatchNorm2d(inc),
        nn.ReLU(),
        nn.Conv2d(inc, outc, 3, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n0(x)
        return n


class NODE25(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n1 = nn.Sequential(
               nn.BatchNorm2d(inc),
        nn.Mish(),
        nn.Conv2d(inc, outc, 3, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n1(x)
        return n



class NODE26(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n2 = nn.Sequential(
                       nn.ReLU(),
        nn.Conv2d(inc, outc, 5, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n2(x)
        return n

    
class NODE27(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                      nn.Mish(),
        nn.Conv2d(inc, outc, 5, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n0(x)
        return n


class NODE28(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                nn.InstanceNorm2d(inc),
        nn.ReLU(),
        nn.Conv2d(inc, outc, 5, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n0(x)
        return n


class NODE29(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n1 = nn.Sequential(
                 nn.InstanceNorm2d(inc),
        nn.Mish(),
        nn.Conv2d(inc, outc, 5, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n1(x)
        return n



class NODE30(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n2 = nn.Sequential(
              nn.BatchNorm2d(inc),
        nn.ReLU(),
        nn.Conv2d(inc, outc, 5, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n2(x)
        return n

    
    
class NODE31(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.n0 = nn.Sequential(
                     nn.BatchNorm2d(inc),
        nn.Mish(),
        nn.Conv2d(inc, outc, 5, stride=1, padding='same')
            )                                   

    def forward(self, x):
        n = self.n0(x)
        return n
    
    
