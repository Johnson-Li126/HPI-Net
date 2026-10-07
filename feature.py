import torch
import torch.nn as nn

# 定义L2正则化的系数
L2_REG = 1e-5

# 卷积层定义
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias,
                           padding_mode='zeros')
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if bias:
            nn.init.constant_(self.conv.bias, 0)
        self.regularizer = nn.Identity()  # PyTorch不直接支持L2正则化，一般在优化器或训练循环外手动添加

    def forward(self, x):
        return self.conv(x)

# 卷积+BN+ReLU模块
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True):
        super(ConvBNReLU, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 平均池化+卷积模块
class AvgPoolConv(nn.Module):
    def __init__(self, pool_size, filters):
        super(AvgPoolConv, self).__init__()
        self.avgpool = nn.AvgPool2d(pool_size)
        self.conv = Conv2d(2 * filters, filters, 1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        return x

# 基础块定义
class BasicBlock(nn.Module):
    def __init__(self, filters, stride, padding, dilation_rate):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBNReLU(filters, filters, 3, stride, padding, dilation=dilation_rate)
        self.conv2 = ConvBNReLU(filters, filters, 3, stride, padding, dilation=dilation_rate, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = self.relu(x)  # 使用ReLU函数代替实例化对象，以避免重复应用
        return x

# 构建连续的基础块
def make_blocks(filters, stride, padding, dilation_rate, num):
    blocks = nn.Sequential(*[BasicBlock(filters, stride, padding, dilation_rate) for _ in range(num)])
    return blocks

# 特征提取模块
class FeatureExtraction(nn.Module):
    def __init__(self, in_c, filters):
        super(FeatureExtraction, self).__init__()
        self.conv0_1 = ConvBNReLU(in_c, filters, 5, stride=2, padding=2)
        self.conv0_2 = ConvBNReLU(filters, 2 * filters, 5, stride=2, padding=2)

        self.conv1_0 = make_blocks(2 * filters, 1, 1, 1, 4)
        self.conv1_1 = make_blocks(2 * filters, 1, 2, 2, 2)
        self.conv1_2 = make_blocks(2 * filters, 1, 4, 4, 2)
        self.conv1_3 = make_blocks(2 * filters, 1, 1, 1, 2)

        self.branch0 = AvgPoolConv(1, filters)
        self.branch1 = AvgPoolConv(2, filters)
        self.branch2 = AvgPoolConv(4, filters)

    def forward(self, x):
        x = self.conv0_1(x)  # (1, 1, 1024, 1024) --> (1, 16, 512, 512)
        x = self.conv0_2(x)  # (1, 16, 512, 512) --> (1, 32, 256, 256)

        x = self.conv1_0(x)  # (1, 32, 256, 256) --> (1, 32, 256, 256)

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)

        x0 = self.branch0(x)  # (1, 32, 256, 256) --> (1, 16, 256, 256)
        x1 = self.branch1(x)  # (1, 32, 256, 256) --> (1, 16, 128, 128)
        x2 = self.branch2(x)  # (1, 32, 256, 256) --> (1, 16, 64, 64)

        return [x0, x1, x2]  # 返回不同尺度的特征图 