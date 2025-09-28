import torch
from torch import nn
from btlbo_unet.all_nodes import *

class DAGBLOCK(nn.Module):
    def __init__(self, channels, sparticle):
        super().__init__()
        n0 = NODE0(channels, channels)        
        n1 = NODE1(channels, channels)        
        n2 = NODE2(channels, channels)        
        n3 = NODE3(channels, channels)        
        n4 = NODE4(channels, channels)        
        n5 = NODE5(channels, channels)        
        n6 = NODE6(channels, channels)        
        n7 = NODE7(channels, channels)
        n8 = NODE8(channels, channels)
        n9 = NODE9(channels, channels)
        n10 = NODE10(channels, channels)    
        n11 = NODE11(channels, channels)        
        n12 = NODE12(channels, channels)        
        n13 = NODE13(channels, channels)       
        n14 = NODE14(channels, channels)
        n15 = NODE15(channels, channels)
        n16 = NODE16(channels, channels)
        n17 = NODE17(channels, channels)      
        n18 = NODE18(channels, channels)   
        n19 = NODE19(channels, channels)  
        n20 = NODE20(channels, channels)
        n21 = NODE21(channels, channels)
        n22 = NODE22(channels, channels)
        n23 = NODE23(channels, channels)
        n24 = NODE24(channels, channels)
        n25 = NODE25(channels, channels)
        n26 = NODE26(channels, channels) 
        n27 = NODE27(channels, channels)   
        n28 = NODE28(channels, channels)
        n29 = NODE29(channels, channels) 
        n30 = NODE30(channels, channels) 
        n31 = NODE31(channels, channels)
        
        self.sparticle = sparticle
        self.block = locals()['n'+str(self.todec(sparticle[0:5]))]   #first 5 bits for block selection [0,1,1,0,0]
        
        
        # self.oblock = locals()['on'+str(self.todec(sparticle[0:5]))]
        #print(block)
         
    
    
    def todec(self,b):
    #     print(b)
        return int(''.join(map(lambda x: str(int(x)), b)), 2)

    
    def forward(self, n0):
        gg = self.sparticle[5:15]  #next 10bits for cochannels, channelsection [1,0,1,0,0,1,1,0,0,1]  
        #input=n0

        #n1
        n1 = self.block(n0)

        #n2
        if gg[0] == 1:
            n2 = self.block(n1)
        else:
            n2 = self.block(n0)

        #n3
        if gg[1] == 1 and gg[2] == 1:
            n3 = self.block(torch.add(n1,n2))
        elif gg[1] == 1 and gg[2] == 0:
            n3 = self.block(n1)
        elif gg[1] == 0 and gg[2] == 1:
            n3 = self.block(n2)
        elif gg[1] == 0 and gg[2] == 0:
            n3 = self.block(n0)


        #n4
        if gg[3] == 1 and gg[4] == 1 and gg[5] == 1:
            n12 = torch.add(n1,n2)
            n4 = self.block(torch.add(n12,n3))
        elif gg[3] == 0 and gg[4] == 1 and gg[5] == 1:
            n4 = self.block(torch.add(n2,n3))
        elif gg[3] == 1 and gg[4] == 0 and gg[5] == 1:
            n4 = self.block(torch.add(n1,n3))
        elif gg[3] == 1 and gg[4] == 1 and gg[5] == 0:
            n4 = self.block(torch.add(n1,n2))
        elif gg[3] == 0 and gg[4] == 0 and gg[5] == 1:
            n4 = self.block(n3)
        elif gg[3] == 0 and gg[4] == 1 and gg[5] == 0:
            n4 = self.block(n2)
        elif gg[3] == 1 and gg[4] == 0 and gg[5] == 0:
            n4 = self.block(n1)
        elif gg[3] == 0 and gg[4] == 0 and gg[5] == 0:
            n4 = self.block(n0)


        #n5
        if gg[6] == 1 and gg[7] == 1 and gg[8] == 1 and gg[9] == 1:
            n12 = torch.add(n1,n2)
            n123 = torch.add(n12,n3)
            n1234 = torch.add(n123,n4)
            n5 = self.block(n1234)
        elif gg[6] == 1 and gg[7] == 1 and gg[8] == 1 and gg[9] == 0:
            n5 = self.block(n1+n2+n3)
        elif gg[6] == 1 and gg[7] == 1 and gg[8] == 0 and gg[9] == 1:
            n5 = self.block(n1+n2+n4)
        elif gg[6] == 1 and gg[7] == 0 and gg[8] == 1 and gg[9] == 1:
            n13 = torch.add(n1, n3)
            n5 = self.block(n13+n4)
        elif gg[6] == 0 and gg[7] == 1 and gg[8] == 1 and gg[9] == 1:
            n5 = self.block(n2+n3+n4)
        elif gg[6] == 1 and gg[7] == 1 and gg[8] == 0 and gg[9] == 0:
            n5 = self.block(torch.add(n1,n2))
        elif gg[6] == 1 and gg[7] == 0 and gg[8] == 1 and gg[9] == 0:
            n5 = self.block(torch.add(n1,n3))
        elif gg[6] == 0 and gg[7] == 1 and gg[8] == 1 and gg[9] == 0:
            n5 = self.block(torch.add(n2,n3))
        elif gg[6] == 1 and gg[7] == 0 and gg[8] == 0 and gg[9] == 1:
            n5 = self.block(torch.add(n1,n4))
        elif gg[6] == 0 and gg[7] == 1 and gg[8] == 0 and gg[9] == 1:
            n5 = self.block(torch.add(n2,n4))
        elif gg[6] == 0 and gg[7] == 0 and gg[8] == 1 and gg[9] == 1:
            n5 = self.block(torch.add(n3,n4))
        elif gg[6] == 1 and gg[7] == 0 and gg[8] == 0 and gg[9] == 0:
            n5 = self.block(n1)
        elif gg[6] == 0 and gg[7] == 1 and gg[8] == 0 and gg[9] == 0:
            n5 = self.block(n2)
        elif gg[6] == 0 and gg[7] == 0 and gg[8] == 1 and gg[9] == 0:
            n5 = self.block(n3)
        elif gg[6] == 0 and gg[7] == 0 and gg[8] == 0 and gg[9] == 1:
            n5 = self.block(n4)
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
