import torch
import torch.nn as nn
import math

from torch.autograd import Variable
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from graph.tools import get_groups


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            bias=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=True)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x) + self.bias
        x = self.bn(x)
        return x


class RHT_MTC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 dilations=[1, 2],  # 多个分支的膨胀率列表
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += self.residual(x)
        return out


class residual_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(residual_conv, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, dim=4):  # N, C, T, V
        if dim == 3:
            N, C, L = x.size()
            pass
        else:
            N, C, T, V = x.size()
            x = x.mean(dim=-2, keepdim=False)  # N, C, V
        x = self.get_graph_feature(x, self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        if dim == 3:
            pass
        else:
            x = repeat(x, 'n c v -> n c t v', t=T)
        return x

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = - xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # N, V, k
        return idx

    def get_graph_feature(self, x, k, idx=None):
        N, C, V = x.size()
        if idx is None:
            idx = self.knn(x, k=k)  # 用于计算输入张量 x 中每个节点的最近邻索引。k 是最近邻的数量。
        idx_base = torch.arange(0, N).view(-1, 1, 1) * V
        idx_base = idx_base.to('cuda')
        idx = idx.to('cuda')
        idx = idx + idx_base
        idx = idx.view(-1)
        x = rearrange(x, 'n c v -> n v c')
        feature = rearrange(x, 'n v c -> (n v) c')[idx, :]
        feature = feature.view(N, V, k, C)
        x = repeat(x, 'n v c -> n v k c', k=k)
        feature = torch.cat((feature - x, x), dim=3)
        feature = rearrange(feature, 'n v k c -> n c v k')
        return feature


class A_CRHTA(nn.Module):
    def __init__(self, in_channels, num_layers, CoM):
        super(A_CRHTA, self).__init__()
        self.num_layers = num_layers
        groups = get_groups(dataset='NTU', CoM=CoM)
        for i, group in enumerate(groups):  # 对获取的组信息进行一些处理，将其转换为 groups 列表。
            group = [i - 1 for i in group]
            groups[i] = group
        inter_channels = in_channels // 4
        self.layers = [groups[i] + groups[i + 1] for i in range(len(groups) - 1)]
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.edge_conv = EdgeConv(inter_channels, inter_channels, k=3)
        self.aggregate = nn.Conv1d(inter_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, L, T, V = x.size()
        x_t = x.max(dim=-2, keepdim=False)[0]
        x_t = self.conv_down(x_t)
        x_sampled = []
        for i in range(self.num_layers):
            s_t = x_t[:, :, i, self.layers[i]]
            s_t = s_t.mean(dim=-1, keepdim=True)
            x_sampled.append(s_t)
        x_sampled = torch.cat(x_sampled, dim=2)
        att = self.edge_conv(x_sampled, dim=3)
        att = self.aggregate(att).view(N, C, L, 1, 1)
        out = (x * self.sigmoid(att)).sum(dim=2, keepdim=False)
        return out

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class CRHT_DGC(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, residual=True, att=False, CoM=8):
        super(CRHT_DGC, self).__init__()
        self.num_layers = A.shape[0] # num_layers 4
        self.num_subset = A.shape[1] # num_subset 3
        self.att = att

        inter_channels = out_channels // (self.num_subset + 1)
        self.adaptive = adaptive

        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            raise ValueError()

        self.conv_down = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range(self.num_layers): # num_layers 4
            self.conv_d = nn.ModuleList()
            self.conv_down.append(nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, kernel_size=1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True)
            ))
            for j in range(self.num_subset): # num_subset 3
                self.conv_d.append(nn.Sequential(
                    nn.Conv2d(inter_channels, inter_channels, kernel_size=1),
                    nn.BatchNorm2d(inter_channels)
                ))
            self.conv_d.append(CTRGC(inter_channels, inter_channels))  # 调用 CTRGC
            self.conv.append(self.conv_d)

        if self.att:
            self.aha = A_CRHTA(out_channels, num_layers=self.num_layers, CoM=CoM)

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        self.bn = nn.BatchNorm2d(out_channels)

        # 7개 conv layer
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    # HD_Gconv
    def forward(self, x):
        A = self.PA
        out = []
        # 多层的图卷积操作
        for i in range(self.num_layers):
            y = []  # 初始化空列表 y： y = [] 用于存储每一层的输出。
            x_down = self.conv_down[i](x)  # 对输入 x 进行下采样操作
            for j in range(self.num_subset):
                # 图卷积操作： 使用 torch.einsum 对 x_down 和邻接矩阵 A 进行乘法，得到 z。
                z = torch.einsum('n c t u, v u -> n c t v', x_down, A[i, j])
                z = self.conv[i][j](z)  # 进行卷积操作，得到卷积输出，并添加到列表 y。
                y.append(z)
            y_edge = self.conv[i][-1](x_down)  # 对下采样结果进行额外的卷积操作，并将结果添加到列表 y
            y.append(y_edge)
            y = torch.cat(y, dim=1)  # 将列表 y 中的结果在维度 1 上拼接
            out.append(y)
        out = torch.stack(out, dim=2)  # 将每一层的输出在维度 2 上进行堆叠，得到最终输出 out。
        if self.att:
            out = self.aha(out)
        else:
            out = out.sum(dim=2, keepdim=False)
        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)
        return out


class STGC_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True,
                 kernel_size=5, dilations=[1, 2], att=True, CoM=8):
        super(STGC_Block, self).__init__()
        self.gcn1 = CRHT_DGC(in_channels, out_channels, A, adaptive=adaptive, att=att, CoM=CoM)
        self.tcn1 = RHT_MTC(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = residual_conv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class CRHTA_ADGCN(nn.Module):
    def __init__(self, num_class=21, num_point=16, num_person=1, in_channels=3,
                 drop_out=0, adaptive=True):
        super(CRHTA_ADGCN, self).__init__()
        Graph = import_class('graph.ntu_rgb_d_hierarchy.Graph')
        graph_args = {'labeling_mode': 'spatial'}
        self.graph = Graph(**graph_args)
        A, CoM = self.graph.A
        self.dataset = 'NTU'
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channels = 64

        self.l1 = STGC_Block(3, base_channels, A, residual=False, adaptive=adaptive, att=False, CoM=CoM)
        self.l2 = STGC_Block(base_channels, base_channels, A, adaptive=adaptive, CoM=CoM)
        self.l3 = STGC_Block(base_channels, base_channels, A, adaptive=adaptive, CoM=CoM)
        self.l4 = STGC_Block(base_channels, base_channels * 2, A, stride=2, adaptive=adaptive, CoM=CoM)
        self.l5 = STGC_Block(base_channels * 2, base_channels * 2, A, adaptive=adaptive, CoM=CoM)
        self.l6 = STGC_Block(base_channels * 2, base_channels * 2, A, adaptive=adaptive, CoM=CoM)
        self.l7 = STGC_Block(base_channels * 2, base_channels * 4, A, stride=2, adaptive=adaptive, CoM=CoM)
        self.l8 = STGC_Block(base_channels * 4, base_channels * 4, A, adaptive=adaptive, CoM=CoM)
        self.l9 = STGC_Block(base_channels * 4, base_channels * 4, A, adaptive=adaptive, CoM=CoM)

        self.fc = nn.Linear(base_channels * 4, num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> n (m v c) t')
        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)

