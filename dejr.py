import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.functional import interpolate, leaky_relu
import torch.nn.functional as F


class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x

class Conv2x_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x
    
class ConvBnLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBnLeakyReLU, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

class UpConvBnLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBnLeakyReLU, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
    def forward(self, x):
        x = self.upconv(x)
        return x
    
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = ConvBnLeakyReLU(in_channels, 16, kernel_size=3)
        self.conv2 = ConvBnLeakyReLU(16, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 通道注意力机制
        att = self.conv1(x)
        att = self.conv2(att)
        att = self.sigmoid(att)
        
        # 加权输入特征
        out = x * att
        return out

def context_upsample(disp_low, up_weights):
    # cv (b,1,h,w), sp (b,9,4*h,4*w)
    b, c, h, w = disp_low.shape 
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)
    disp = (disp_unfold*up_weights).sum(1)
    return disp

def warp_right_to_left(right_image, disparity):
    """
    使用视差图将右视图变换为左视图
    
    Args:
        right_image: 右视图, 形状为 [B, 3, H, W]
        disparity: 视差图, 形状为 [B, 1, H, W]
        
    Returns:
        warped_left_image: 变换后的左视图, 形状为 [B, 3, H, W]
    """
    B, _, H, W = right_image.size()
    
    # 创建网格坐标
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, H-1, H, device=right_image.device),
        torch.linspace(0, W-1, W, device=right_image.device),
        indexing='ij'
    )
    
    # 转换为归一化坐标 [-1, 1]
    x_grid = (2.0 * x_grid / (W - 1)) - 1.0
    y_grid = (2.0 * y_grid / (H - 1)) - 1.0
    
    # 扩展到batch维度
    grid = torch.stack((x_grid, y_grid), dim=-1).repeat(B, 1, 1, 1)  # [B, H, W, 2]
    
    # 根据视差调整x坐标 (视差是水平方向的偏移)
    # 将视差从像素单位转换为归一化单位 [-1, 1]
    disparity_normalized = disparity * (2.0 / (W - 1))
    
    # 调整网格的x坐标
    # 注意：视差表示从左到右，所以对右视图采样时需要减去视差
    grid[:, :, :, 0] = grid[:, :, :, 0] - disparity_normalized.squeeze(1)
    # 使用网格对右视图进行采样
    warped_left_image = F.grid_sample(
        right_image, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    
    return warped_left_image

class DEJR(nn.Module):
    """
    精细化模块，用于对视差图进行细化处理。
    """
    def __init__(self, in_c, out_c, filters):
        super(DEJR, self).__init__()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(int((in_c-1)//4), filters, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(filters, filters, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(filters), nn.ReLU()
            )
        self.spx = nn.Sequential(nn.ConvTranspose2d(filters*2, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, filters, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(filters, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )
        
        # 编码器部分 - 下采样路径
        self.conv1 = ConvBnLeakyReLU(in_c, filters, stride=1)  # 保持原始分辨率
        self.conv2 = ConvBnLeakyReLU(filters, filters, stride=2)           # H/2, W/2
        self.conv3 = ConvBnLeakyReLU(filters, filters, stride=2)          # H/4, W/4
        self.conv4 = ConvBnLeakyReLU(filters, filters, stride=2)         # H/8, W/8
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ConvBnLeakyReLU(filters, filters, stride=1),
            ConvBnLeakyReLU(filters, filters, stride=1),
            ConvBnLeakyReLU(filters, filters, stride=1)
        )
        
        # 解码器部分 - 上采样路径（带跳跃连接）
        self.upconv3 = UpConvBnLeakyReLU(filters, filters)
        self.iconv3 = ConvBnLeakyReLU(filters*2, filters, stride=1)
        
        self.upconv2 = UpConvBnLeakyReLU(filters, filters)
        self.iconv2 = ConvBnLeakyReLU(filters*2, filters, stride=1)  
        
        self.upconv1 = UpConvBnLeakyReLU(filters, filters)
        self.iconv1 = ConvBnLeakyReLU(filters*2, filters, stride=1)

        # 残差预测层，输出视差的残差调整
        self.disp_refine = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.1),
            nn.Conv2d(filters, 1, kernel_size=3, padding=1)
        )
        
        # 注意力模块 - 用于加权集成各种输入特征
        self.attention = AttentionModule(in_c)

    def forward(self, inputs):
        """
        Forward pass for the simplified refinement module
        :param inputs: [disparity_4x, left_img, right_img, l0, gx, gy, var0]
        :return: refined disparity map
        """
        assert len(inputs) == 7

        disp_4x = inputs[0].permute(0, 3, 1, 2)  # (N, 1, H/4, W/4)
        left = inputs[1]  # (N, 1, H, W)
        right = inputs[2]
        feat_4x = inputs[3]
        var_4x = inputs[-1].permute(0, 3, 1, 2)
        scale_factor = left.size(2) / disp_4x.size(2)

        stem_2x = self.stem_2(left)
        xspx = self.spx_4(feat_4x)
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        disp_scaled = context_upsample(disp_4x, spx_pred.float()).unsqueeze(1) * scale_factor 

        # 调整视差图尺寸以匹配RGB图像
        var_resized = interpolate(var_4x, size=(left.size(2), left.size(3)), mode='bilinear', align_corners=True) * (scale_factor**2)
        # var_log = torch.log1p(var_resized) 
        # var_scaled = var_log / (torch.mean(var_log) + 1e-6)
        var_scaled = (var_resized - var_resized.min()) / (var_resized.max() - var_resized.min() + 1e-8)

        warped_left = warp_right_to_left(right, disp_scaled)
        error_map = torch.abs(warped_left - left)
    
        concat = torch.cat((disp_scaled, left, inputs[-3], inputs[-2], error_map, var_scaled), dim=1)
        concat_attn = self.attention(concat)
        # 编码器前向传播
        conv1 = self.conv1(concat_attn)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # 瓶颈层
        bottle = self.bottleneck(conv4)
        # 解码器前向传播（带跳跃连接）
        upconv3 = self.upconv3(bottle)
        iconv3 = self.iconv3(torch.cat([upconv3, conv3], dim=1))
        upconv2 = self.upconv2(iconv3)
        iconv2 = self.iconv2(torch.cat([upconv2, conv2], dim=1))
        upconv1 = self.upconv1(iconv2)
        iconv1 = self.iconv1(torch.cat([upconv1, conv1], dim=1))
        
        # 预测视差残差并与原始视差相加得到细化后的视差
        disp_res = self.disp_refine(iconv1)
        refined_disp = disp_scaled + disp_res
        
        return refined_disp, error_map