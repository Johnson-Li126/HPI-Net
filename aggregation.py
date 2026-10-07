import torch
import torch.nn as nn

L2 = 1e-5
alpha = 0.2

class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.conv(x)

class Conv3dBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(Conv3dBNReLU, self).__init__()
        self.conv = Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=alpha, inplace=False) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class TransConv3dBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, activation=True):
        super(TransConv3dBNReLU, self).__init__()
        self.transconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out', nonlinearity='relu')
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=alpha, inplace=False) if activation else nn.Identity()

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class Hourglass(nn.Module):
    def __init__(self, filters):
        super(Hourglass, self).__init__()
        self.conv1 = Conv3dBNReLU(filters, filters, 3, 1, padding=1, activation=True)
        self.conv2 = Conv3dBNReLU(filters, filters, 3, 1, padding=1, activation=True)
        self.conv3 = Conv3dBNReLU(filters, 2*filters, 3, 2, padding=1, activation=True)
        self.conv4 = Conv3dBNReLU(2*filters, 2*filters, 3, 1, padding=1, activation=True)
        self.conv5 = Conv3dBNReLU(2*filters, 2*filters, 3, 2, padding=1, activation=True)
        self.conv6 = Conv3dBNReLU(2*filters, 2*filters, 3, 1, padding=1, activation=True)
        self.conv7 = TransConv3dBNReLU(2*filters, 2*filters, 4, 2, padding=1, output_padding=0, activation=True)
        self.conv8 = TransConv3dBNReLU(2*filters, filters, 4, 2, padding=1, output_padding=0, activation=True)

    def forward(self, x):
        x1 = self.conv1(x)  # (1, 16, 32, 256, 256) --> (1, 16, 32, 256, 256)
        x1 = self.conv2(x1)  # (1, 16, 32, 256, 256) --> (1, 16, 32, 256, 256)
        x2 = self.conv3(x1)  # (1, 16, 32, 256, 256) --> (1, 32, 16, 128, 128)
        x2 = self.conv4(x2)  # (1, 32, 16, 128, 128) --> (1, 32, 16, 128, 128)
        x3 = self.conv5(x2)  # (1, 32, 16, 128, 128) --> (1, 32, 8, 64, 64)
        x3 = self.conv6(x3)  # (1, 32, 8, 64, 64) --> (1, 32, 8, 64, 64)
        x4 = self.conv7(x3)  # (1, 32, 8, 64, 64) --> (1, 32, 16, 128, 128)
        x4 += x2  # (1, 32, 16, 128, 128) --> (1, 32, 16, 128, 128)
        x5 = self.conv8(x4)  # (1, 32, 16, 128, 128) --> (1, 16, 32, 256, 256)
        x5 += x1
        return x5