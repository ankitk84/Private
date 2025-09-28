import torch
from torch import nn
from btlbo_unet.all_nodes import *

class CBLOCK(nn.Module):
    def __init__(self, inc, outc, sparticle):
        super().__init__()
        n0 = NODE0(inc, outc)
        on0 = NODE0(outc, outc)
        
        n1 = NODE1(inc, outc)
        on1 = NODE1(outc, outc)
        
        n2 = NODE2(inc, outc)
        on2 = NODE2(outc, outc)
        
        n3 = NODE3(inc, outc)
        on3 = NODE3(outc, outc)
        
        n4 = NODE4(inc, outc)
        on4 = NODE4(outc, outc)
        
        n5 = NODE5(inc, outc)
        on5 = NODE5(outc, outc)
        
        n6 = NODE6(inc, outc)
        on6 = NODE6(outc, outc)
        
        n7 = NODE7(inc, outc)
        on7 = NODE7(outc, outc)
        
        n8 = NODE8(inc, outc)
        on8 = NODE8(outc, outc)
        
        n9 = NODE9(inc, outc)
        on9 = NODE9(outc, outc)
        
        n10 = NODE10(inc, outc)
        on10 = NODE10(outc, outc)
        
        n11 = NODE11(inc, outc)
        on11 = NODE11(outc, outc)
        
        n12 = NODE12(inc, outc)
        on12 = NODE12(outc, outc)
        
        n13 = NODE13(inc, outc)
        on13 = NODE13(outc, outc)
        
        n14 = NODE14(inc, outc)
        on14 = NODE14(outc, outc)
        
        n15 = NODE15(inc, outc)
        on15 = NODE15(outc, outc)
        
        n16 = NODE16(inc, outc)
        on16 = NODE16(outc, outc)
        
        n17 = NODE17(inc, outc)
        on17 = NODE17(outc, outc)
        
        n18 = NODE18(inc, outc)
        on18 = NODE18(outc, outc)
        
        n19 = NODE19(inc, outc)
        on19 = NODE19(outc, outc)
        
        n20 = NODE20(inc, outc)
        on20 = NODE20(outc, outc)
        
        n21 = NODE21(inc, outc)
        on21 = NODE21(outc, outc)
        
        n22 = NODE22(inc, outc)
        on22 = NODE22(outc, outc)
        
        n23 = NODE23(inc, outc)
        on23 = NODE23(outc, outc)
        
        n24 = NODE24(inc, outc)
        on24 = NODE24(outc, outc)
        
        n25 = NODE25(inc, outc)
        on25 = NODE25(outc, outc)
        
        n26 = NODE26(inc, outc)
        on26 = NODE26(outc, outc)
        
        n27 = NODE27(inc, outc)
        on27 = NODE27(outc, outc)
        
        n28 = NODE28(inc, outc)
        on28 = NODE28(outc, outc)
        
        n29 = NODE29(inc, outc)
        on29 = NODE29(outc, outc)
        
        n30 = NODE30(inc, outc)
        on30 = NODE30(outc, outc)
        
        n31 = NODE31(inc, outc)
        on31 = NODE31(outc, outc)
        
        self.sparticle = sparticle
        self.block = locals()['n'+str(self.todec(sparticle[0:5]))]
        self.oblock = locals()['on'+str(self.todec(sparticle[0:5]))]
        
    def todec(self,b):
    #     print(b)
        return int(''.join(map(lambda x: str(int(x)), b)), 2)
        
    def forward(self, n0):
        gg = self.sparticle[5:15]  #next 10bits for coinc, outcection [1,0,1,0,0,1,1,0,0,1]   

        #n1
        n1 = self.block(n0)

        #n2
        if gg[0] == 1:
            n2 = self.oblock(n1)
        else:
            n2 = self.block(n0)

        #n3
        if gg[1] == 1 and gg[2] == 1:
            n3 = self.oblock(torch.add(n1,n2))
        elif gg[1] == 1 and gg[2] == 0:
            n3 = self.oblock(n1)
        elif gg[1] == 0 and gg[2] == 1:
            n3 = self.oblock(n2)
        elif gg[1] == 0 and gg[2] == 0:
            n3 = self.block(n0)


        #n4
        if gg[3] == 1 and gg[4] == 1 and gg[5] == 1:
            n12 = torch.add(n1,n2)
            n4 = self.oblock(torch.add(n12,n3))
        elif gg[3] == 0 and gg[4] == 1 and gg[5] == 1:
            n4 = self.oblock(torch.add(n2,n3))
        elif gg[3] == 1 and gg[4] == 0 and gg[5] == 1:
            n4 = self.oblock(torch.add(n1,n3))
        elif gg[3] == 1 and gg[4] == 1 and gg[5] == 0:
            n4 = self.oblock(torch.add(n1,n2))
        elif gg[3] == 0 and gg[4] == 0 and gg[5] == 1:
            n4 = self.oblock(n3)
        elif gg[3] == 0 and gg[4] == 1 and gg[5] == 0:
            n4 = self.oblock(n2)
        elif gg[3] == 1 and gg[4] == 0 and gg[5] == 0:
            n4 = self.oblock(n1)
        elif gg[3] == 0 and gg[4] == 0 and gg[5] == 0:
            n4 = self.block(n0)


        #n5
        if gg[6] == 1 and gg[7] == 1 and gg[8] == 1 and gg[9] == 1:
            n5 = self.oblock(n1+n2+n3+n4)
        elif gg[6] == 1 and gg[7] == 1 and gg[8] == 1 and gg[9] == 0:
            n5 = self.oblock(n1+n2+n3)
        elif gg[6] == 1 and gg[7] == 1 and gg[8] == 0 and gg[9] == 1:
            n5 = self.oblock(n1+n2+n4)
        elif gg[6] == 1 and gg[7] == 0 and gg[8] == 1 and gg[9] == 1:
            n13 = torch.add(n1, n3)
            n5 = self.oblock(torch.add(n13,n4))
        elif gg[6] == 0 and gg[7] == 1 and gg[8] == 1 and gg[9] == 1:
            n5 = self.oblock(n2+n3+n4)
        elif gg[6] == 1 and gg[7] == 1 and gg[8] == 0 and gg[9] == 0:
            n5 = self.oblock(torch.add(n1,n2))
        elif gg[6] == 1 and gg[7] == 0 and gg[8] == 1 and gg[9] == 0:
            n5 = self.oblock(torch.add(n1,n3))
        elif gg[6] == 0 and gg[7] == 1 and gg[8] == 1 and gg[9] == 0:
            n5 = self.oblock(torch.add(n2,n3))
        elif gg[6] == 1 and gg[7] == 0 and gg[8] == 0 and gg[9] == 1:
            n5 = self.oblock(torch.add(n1,n4))
        elif gg[6] == 0 and gg[7] == 1 and gg[8] == 0 and gg[9] == 1:
            n5 = self.oblock(torch.add(n2,n4))
        elif gg[6] == 0 and gg[7] == 0 and gg[8] == 1 and gg[9] == 1:
            n5 = self.oblock(torch.add(n3,n4))
        elif gg[6] == 1 and gg[7] == 0 and gg[8] == 0 and gg[9] == 0:
            n5 = self.oblock(n1)
        elif gg[6] == 0 and gg[7] == 1 and gg[8] == 0 and gg[9] == 0:
            n5 = self.oblock(n1)
        elif gg[6] == 0 and gg[7] == 0 and gg[8] == 1 and gg[9] == 0:
            n5 = self.oblock(n3)
        elif gg[6] == 0 and gg[7] == 0 and gg[8] == 0 and gg[9] == 1:
            n5 = self.oblock(n4)
        elif gg[6] == 0 and gg[7] == 0 and gg[8] == 0 and gg[9] == 0:
            n5 = self.block(n0)



        #output
        ops = [n5]
        if gg[0] == gg[1] == gg[3] == gg[6] == 0:
            ops.append(n1)
        if gg[2] == gg[4] == gg[7] == 0:
            ops.append(n2)
        if gg[5] == gg[8] == 0:
            ops.append(n3)
        if gg[9] == 0:
            ops.append(n4)
    #     print(len(ops))


        out1 = ops[0]
        for i in range(1, len(ops)):
            out1 = torch.add(out1, ops[i]) 
    #         print("out1")
    #         print(out1.shape)
        return out1