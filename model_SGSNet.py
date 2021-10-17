import torch
from torch import nn

def conv_block_3d_out(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm3d(out_dim),
        activation, )

def conv_trans_block_3d(in_dim, out_dim, activation, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size, stride, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )

def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

class Heatmap_attention2(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, activation2):
        super(Heatmap_attention2, self).__init__()
        self.W_1 = nn.Sequential(
                        nn.Conv3d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0),#3 1
                        nn.BatchNorm3d(mid_dim),)
        self.relu = nn.ReLU(inplace=True)
        self.psi = conv_block_3d_out(mid_dim,out_dim,activation2)#sigmoid

    def forward(self, add1, add2 ,add3, out2):
        W_1 = self.W_1(add1+add2+add3)
        psi = self.relu(W_1)
        psi_out = self.psi(psi)

        return out2*psi_out# + out2

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1 ,L=8):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv3d(features, features, kernel_size=(5+i*2,5+i*2,3), stride=stride, padding=(2+i,2+i,1), groups=G),#kernel_size=3+i*2   padding=1+i
                nn.BatchNorm3d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):   
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1).mean(-1)
        fea_z = self.fc(fea_s)         
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

#Selective Branch Module(SBM)
class SBConv(nn.Module):
    def __init__(self, features, WH=32, M=2, G=8, r=2, stride=1 ,L=8):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SBConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, branch1, branch2):  
        branch1=branch1.unsqueeze(dim=1)
        branch2=branch2.unsqueeze(dim=1)
        feas = torch.cat([branch1, branch2], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1).mean(-1)
        fea_z = self.fc(fea_s)         
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v
    
class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH=32, M=2, G=8, r=2, mid_features=None, stride=1, L=8):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        G = mid_features
        self.feas = nn.Sequential(
            nn.Conv3d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm3d(mid_features),
            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm3d(mid_features),
            nn.Conv3d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm3d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm3d(out_features)
            )
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)

def skunet_block_2_3d(in_dim, out_dim, activation=False):
    return nn.Sequential(
        SKUnit(in_dim, out_dim),
        nn.ReLU(), 
        SKUnit(out_dim, out_dim),
        nn.ReLU(), )


class SGSNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(SGSNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        # Down sampling
        self.down_1 = skunet_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = skunet_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = skunet_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = skunet_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)

        # Up sampling of SDB
        self.trans_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.up_1 = skunet_block_2_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.trans_1_add = conv_trans_block_3d(self.num_filters * 4, self.num_filters, activation, kernel_size=5, stride=4)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.up_2 = skunet_block_2_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.trans_2_add = conv_trans_block_3d(self.num_filters * 2, self.num_filters, activation, stride=2)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.up_3 = skunet_block_2_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.out1 = nn.Conv3d(self.num_filters, 1, kernel_size=1, stride=1, padding=0)

        # Up sampling of MSB
        self.trans_1_seg = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.up_1_seg = skunet_block_2_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.trans_2_seg = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.up_2_seg = skunet_block_2_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.trans_3_seg = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.up_3_seg = skunet_block_2_3d(self.num_filters * 2, self.num_filters * 1, activation)

		# Structure Attention Module (SAM)
        self.heatmap_att = Heatmap_attention2(self.num_filters , self.num_filters//2, 1, self.sigmoid)
        self.SBConv_1 = SBConv(self.num_filters * 4)
        self.SBConv_2 = SBConv(self.num_filters * 2)
        self.SBConv_3 = SBConv(self.num_filters)
        self.SBConv_out = SBConv(self.num_filters)

        self.out2 = nn.Conv3d(self.num_filters, out_dim, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) 
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)

        # Up sampling of SDB
        trans_1 = self.trans_1(down_4)
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1) 
        concat_2 = torch.cat([trans_2, down_2], dim=1) 
        up_2 = self.up_2(concat_2) 

        trans_3 = self.trans_3(up_2)  
        concat_3 = torch.cat([trans_3, down_1], dim=1) 
        up_3 = self.up_3(concat_3)
        out1 = self.out1(up_3)

        # Up sampling of MSB
        trans_1_seg = self.trans_1_seg(down_4) 
        concat_1_seg = torch.cat([trans_1_seg, down_3], dim=1) 
        up_1_seg = self.up_1_seg(concat_1_seg)

        trans_2_seg = self.trans_2_seg(up_1_seg)
        concat_2_seg = torch.cat([trans_2_seg, down_2], dim=1) 
        up_2_seg = self.up_2_seg(concat_2_seg) 

        trans_3_seg = self.trans_3_seg(up_2_seg) 
        concat_3_seg = torch.cat([trans_3_seg, down_1], dim=1) 
        up_3_seg = self.up_3_seg(concat_3_seg) 

        # Structure Attention Module (SAM)
        SBConv_1 = self.SBConv_1(up_1, up_1_seg)
        trans_1_add = self.trans_1_add(SBConv_1)
        SBConv_2 = self.SBConv_2(up_2, up_2_seg)
        trans_2_add = self.trans_2_add(SBConv_2) 
        SBConv_3 = self.SBConv_3(up_3, up_3_seg)
        heatmap_att = self.heatmap_att(trans_1_add,trans_2_add,SBConv_3,up_3_seg)
        # Output
        out2 = self.out2(heatmap_att+up_3_seg)#(SBConv_out)  
        return out1,out2