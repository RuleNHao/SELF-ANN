"""
@Author: Penghao Tian <rulenhao@mail.ustc.edu.cn>
@Date: 2022/10/5 04:27
@Description: 训练过程中需要用到的模型
"""


import torch
import einops
from torch import nn


class Conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(ch_out),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(ch_out),
                                  nn.ReLU(inplace=True))
        pass
    def forward(self, x):
        x = self.conv(x)
        return x
    pass


class Up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, up_row, up_col):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(size=(up_row, up_col)),
                                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(ch_out),
                                nn.ReLU(inplace=True))
        pass
    def forward(self, x):
        x = self.up(x)
        return x
    pass


class Max_pool(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        pass
    def forward(self, x):
        x = self.maxpool(x)
        return x
    pass


class EncodeDecode(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.Conv0 = Conv_block(ch_in=1, ch_out=32)
        self.Conv1 = Conv_block(ch_in=32, ch_out=64)
        self.Conv2 = Conv_block(ch_in=64, ch_out=128)
        self.Conv3 = Conv_block(ch_in=128, ch_out=256)
        self.Conv4 = Conv_block(ch_in=256, ch_out=512)
        self.Conv5 = Conv_block(ch_in=512, ch_out=1024)
        
        self.Max0 = Max_pool(kernel_size=(2,2), stride=(2,2))
        self.Max1 = Max_pool(kernel_size=(2,2), stride=(2,2))
        self.Max2 = Max_pool(kernel_size=(2,2), stride=(2,2))
        self.Max3 = Max_pool(kernel_size=(2,2), stride=(2,2))
        self.Max4 = Max_pool(kernel_size=(2,2), stride=(2,2))
        
        self.Up0 = Up_conv(ch_in=1024, ch_out=512, up_row=4, up_col=4) # x4
        self.Up1 = Up_conv(ch_in=512, ch_out=256, up_row=8, up_col=8) # x3
        self.Up2 = Up_conv(ch_in=256, ch_out=128, up_row=16, up_col=16) # x2
        self.Up3 = Up_conv(ch_in=128, ch_out=64, up_row=32, up_col=32) # x1
        self.Up4 = Up_conv(ch_in=64, ch_out=32, up_row=64, up_col=64) # x0
        
        self.Up_conv0 = Conv_block(ch_in=512, ch_out=512)
        self.Up_conv1 = Conv_block(ch_in=256, ch_out=256)
        self.Up_conv2 = Conv_block(ch_in=128, ch_out=128)
        self.Up_conv3 = Conv_block(ch_in=64, ch_out=64)
        self.Up_conv4 = Conv_block(ch_in=32, ch_out=32)
        
        self.Conv1x1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        
        pass
    
    def forward(self, x):
        
        # encode
        x0 = self.Conv0(x)
        x1 = self.Conv1(self.Max0(x0))
        x2 = self.Conv2(self.Max1(x1))
        x3 = self.Conv3(self.Max2(x2))
        x4 = self.Conv4(self.Max3(x3))
        x5 = self.Conv5(self.Max4(x4))
        
        # decode
        d4 = self.Up_conv0(self.Up0(x5))
        d3 = self.Up_conv1(self.Up1(d4))
        d2 = self.Up_conv2(self.Up2(d3))
        d1 = self.Up_conv3(self.Up3(d2))
        d0 = self.Up_conv4(self.Up4(d1))
        
        out = self.Conv1x1(d0)
        return out
    
    pass


class Trans1DTo2DLayerWith1Channel(nn.Module):
    
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        out = einops.rearrange(x, "b (c w h) -> b c w h", c=1, w=64, h=64)
        return out
    
    pass


class Trans1DTo2DLayerWith3Channel(nn.Module):
    
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        out = einops.rearrange(x, "b (c w h) -> b c w h", c=3, w=64, h=64)
        return out
    
    pass


class Trans2DTo1DLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        out = einops.rearrange(x, "b c w h -> b (c w h)")
        return out
    
    pass


class FirstLinear(nn.Module):
    
    def __init__(self, feature_in):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(int(feature_in), 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64*32),
                                    nn.ReLU(),
                                    nn.Linear(64*32, 64*64*3),
                                    nn.ReLU(),)
        pass
    
    def forward(self, x):
        return self.dense1(x)
    
    pass


class LastLinear(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(1000, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1))
        pass
    
    def forward(self, x):
        return self.dense1(x)
    
    pass



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers: list,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation = None,
        norm_layer = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    pass


class ResNet(nn.Module):
    
    def __init__(self, block_name: str, layers: list):
        super().__init__()
        if block_name == "BasicBlock":
            self.net = _ResNet(BasicBlock, layers)
            pass
        elif block_name == "Bottleneck":
            self.net = _ResNet(Bottleneck, layers)
            pass
        else:
            raise OSError("Unknown block")
            pass
    
    def forward(self, x):
        return self.net(x)
    
    pass


def CustomLoss01(y_input, y_target):
    threshold = 0.2
    alpha1 = 0.5
    alpha2 = 1.

    weight = torch.full_like(y_target, alpha2)
    weight[y_target < threshold] = alpha1

    l1_loss = nn.L1Loss(reduction="none")(y_input, y_target)
    weight_loss = l1_loss * weight
    loss = weight_loss.sum() / weight.sum()
    return loss

