import torch.nn as nn
import torch.nn.functional as F
from models_pooling import GPP
from core_qnn.quaternion_layers import QuaternionConv as QuaternionConv2d
from core_qnn.quaternion_layers import QuaternionLinear

class ResnetBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, quaternionized=False, leakyrelu=False):
        super(ResnetBasicBlock, self).__init__()
        if quaternionized:
            layer_conv2d = QuaternionConv2d
        else:
            layer_conv2d = nn.Conv2d
        self.conv1 = layer_conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1) #, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = layer_conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1) #, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if leakyrelu:
            self.lastactivation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.lastactivation = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                layer_conv2d(in_channels=in_planes, out_channels=self.expansion*planes, kernel_size=1, stride=stride), #bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.lastactivation(out)
        return out


class ResnetCNN_v2(nn.Module):
    def __init__(self, n_out, cnn_cfg=None, input_channels=4,
                 pooling_type='spp', pooling_levels=3, quaternionized=False, small_version=False):
        super(ResnetCNN_v2, self).__init__()
        self.quaternionized = quaternionized
        self.phoc_size_actual = n_out
        if quaternionized:
            layer_linear = QuaternionLinear
            self.phoc_size = ((n_out // 4) + 1) * 4
        else:
            layer_linear = nn.Linear
            self.phoc_size = n_out

        if not small_version:
            pooling_factor = 512
        else:
            pooling_factor = 256

        if pooling_type == 'spp':
            pooling_output_size = sum([4**level for level in range(pooling_levels)]) * pooling_factor
        elif pooling_type == 'tpp':
            pooling_output_size = (2**pooling_levels - 1) * pooling_factor
        elif pooling_type == 'gpp':
            pooling_output_size = sum([h*w for h in pooling_levels[0] for w in pooling_levels[1]]) * pooling_factor

        if not small_version:
            self.main = nn.Sequential(
                ResnetBasicBlock(in_planes=input_channels, planes=64, stride=1, quaternionized=quaternionized, leakyrelu=True),
                ResnetBasicBlock(in_planes=64, planes=128, stride=1, quaternionized=quaternionized, leakyrelu=True),
                ResnetBasicBlock(in_planes=128, planes=256, stride=1, quaternionized=quaternionized, leakyrelu=True),
                ResnetBasicBlock(in_planes=256, planes=256, stride=1, quaternionized=quaternionized, leakyrelu=True),
                ResnetBasicBlock(in_planes=256, planes=256, stride=1, quaternionized=quaternionized, leakyrelu=True),
                ResnetBasicBlock(in_planes=256, planes=512, stride=1, quaternionized=quaternionized, leakyrelu=True),
                ResnetBasicBlock(in_planes=512, planes=pooling_factor, stride=1, quaternionized=quaternionized, leakyrelu=True),
                GPP(gpp_type=pooling_type, levels=pooling_levels),
                nn.ReLU(),
                layer_linear(pooling_output_size, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                layer_linear(1024, self.phoc_size)
            )
        else:
            self.main = nn.Sequential(
                ResnetBasicBlock(in_planes=input_channels, planes=64, stride=1, quaternionized=quaternionized, leakyrelu=True),
                ResnetBasicBlock(in_planes=64, planes=128, stride=1, quaternionized=quaternionized, leakyrelu=True),
                ResnetBasicBlock(in_planes=128, planes=pooling_factor, stride=1, quaternionized=quaternionized, leakyrelu=True),
                GPP(gpp_type=pooling_type, levels=pooling_levels),
                nn.ReLU(),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                layer_linear(pooling_output_size, self.phoc_size)
            )
    
    def forward(self, x):
        return(self.main(x)[:, :self.phoc_size_actual])
