import torch
import torch.nn as nn

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x
 
class DDAC(nn.Module):
    def __init__(self, min_disp=-128.0, max_disp=64.0, filters=16, cv='gwc', is_gate=True, alpha=0.5):
        super(DDAC, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.num_groups = filters
        self.cv = cv
        self.is_gate = is_gate

        if self.is_gate:
            self.gate_conv = nn.Sequential(
                nn.Conv3d(2*filters, filters, 3, padding=1),
                nn.BatchNorm3d(filters),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(filters, filters, 3, padding=1),
            )
            self.gate = None
        elif alpha is not None:
            # 使用可学习参数控制融合比例
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            print('Choose one of the fusion strategy.')

        if self.cv == 'gwc':
            self.coor_stem = BasicConv(filters, filters, is_3d=True, kernel_size=3, stride=1, padding=1)

    def compute_weights(self, disp_diff, upsampled_var):
        B, H, W, _ = disp_diff.shape
        # 计算每个样本的归一化因子
        disp_scale = torch.mean(disp_diff.view(B, -1), dim=1) + 1.0  # 避免除零
        var_scale = torch.mean(upsampled_var.view(B, -1), dim=1) + 1.0
        
        # 扩展维度以便广播
        disp_scale = disp_scale.view(B, 1, 1, 1)
        var_scale = var_scale.view(B, 1, 1, 1)
        
        # 计算归一化后的权重
        weight = torch.exp(-((disp_diff/disp_scale).pow(2) + (upsampled_var/var_scale)))
        return weight

    def groupwise_correlation(self, fea1, fea2, num_groups):
        B, C, H, W = fea1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
        assert cost.shape == (B, num_groups, H, W)
        return cost

    def concat(self, left, right):
        cost_volume = []
        for i in range(self.min_disp, self.max_disp):
            if i < 0:
                # pad_left = abs(i)
                # pad_right = 0
                left_cv = left[:, :, :, :i]
                right_cv = right[:, :, :, -i:]
                padding = (-i, 0, 0, 0, 0, 0)
            elif i > 0:
                # pad_left = 0
                # pad_right = i
                left_cv = left[:, :, :, i:]
                right_cv = right[:, :, :, :-i]
                padding = (0, i, 0, 0, 0, 0)
            else:
                # pad_left = 0
                # pad_right = 0
                left_cv = left
                right_cv = right
                padding = (0, 0, 0, 0, 0, 0)
            
            # Concatenate along the channel dimension
            # concatenated = torch.cat((left, right), dim=1)
            concatenated = left_cv - right_cv
            # Pad to maintain the same spatial dimensions across the cost volume
            padded_concatenated = nn.functional.pad(concatenated, padding, mode='constant', value=0)
            cost_volume.append(padded_concatenated)
        cost_volume = torch.stack(cost_volume, dim=2)
        return cost_volume
    
    def gwc(self, left, right):
        B, C, H, W = left.shape
        volume = left.new_zeros([B, self.num_groups, self.max_disp-self.min_disp, H, W])
        for i in range(self.min_disp, self.max_disp):
            idx = i - self.min_disp
            if i > 0:
                volume[:, :, idx, :, i:] = self.groupwise_correlation(left[:, :, :, i:], right[:, :, :, :-i], self.num_groups)
            elif i < 0:
                abs_i = abs(i)
                volume[:, :, idx, :, :-abs_i] = self.groupwise_correlation(left[:, :, :, :-abs_i], right[:, :, :, abs_i:], self.num_groups)
            else:
                volume[:, :, idx, :, :] = self.groupwise_correlation(left, right, self.num_groups)
        cost_volume = self.coor_stem(volume.contiguous())
        return cost_volume
    
    def forward(self, inputs, prev_disp=None, prev_prob=None, prev_var=None):
        assert len(inputs) == 2
        left = inputs[0]
        right = inputs[1]
        B, C, H, W = left.shape

        if self.cv == 'concat':
            cost_volume = self.concat(left, right)
        elif self.cv == 'gwc':
            cost_volume = self.gwc(left, right)
        else:
            print("invalid cost volume construction methods! Please change the config.")

        # 如果提供了上一尺度的信息，融合到当前代价体
        if prev_disp is not None and prev_prob is not None and prev_var is not None:            
            # 视差尺度变换（上一尺度到当前尺度）
            # 例如：从1/16尺度到1/8尺度，视差值需要乘以2
            scale_factor = 2  # 假设尺度变化是2倍
            current_H, current_W = cost_volume.shape[3], cost_volume.shape[4]
            # 上采样视差值并调整尺度
            upsampled_disp = nn.functional.interpolate(prev_disp.permute(0, 3, 1, 2), 
                                         size=(current_H, current_W), 
                                         mode='bilinear', 
                                         align_corners=True) * scale_factor
            upsampled_disp = upsampled_disp.permute(0, 2, 3, 1)  # [B, H, W, 1]
            # 上采样方差
            upsampled_var = nn.functional.interpolate(prev_var.permute(0, 3, 1, 2), 
                                        size=(current_H, current_W), 
                                        mode='bilinear', 
                                        align_corners=True) * (scale_factor**2)
            upsampled_var = upsampled_var.permute(0, 2, 3, 1)  # [B, H, W, 1]

            # 计算每个候选视差值的权重
            weight_volume = torch.zeros_like(cost_volume)
            for d_idx, d_val in enumerate(range(self.min_disp, self.max_disp)):
                disp_diff = torch.abs(upsampled_disp - d_val)  # [B, H, W, 1] # 计算当前视差候选值与上采样后视差的差异
                sigma = torch.clamp(upsampled_var, min=0.0001)  # 防止除零
                weight = self.compute_weights(disp_diff, sigma) # 计算加权因子：exp(-diff²+(2*var))，视差差异小且方差小的位置得到较高权重
                weight_volume[:, :, d_idx, :, :] = weight.permute(0, 3, 1, 2)
            
            # 归一化权重，保证权重和为1
            weight_sum = torch.sum(weight_volume, dim=2, keepdim=True) + 1e-9
            weight_volume = weight_volume / weight_sum
        
            if self.is_gate:
                # 门控融合机制
                weighted_cost = cost_volume * weight_volume
                gate_input = torch.cat([cost_volume, weighted_cost], dim=1)  # 将原始代价体和加权代价体拼接
                gate = torch.sigmoid(self.gate_conv(gate_input))  # 计算门控系数 (0-1之间)
                # 应用门控融合
                enhanced_cost_volume = cost_volume * (1 - gate) + weighted_cost * gate
            else:
                # 将权重融合到代价体中（加权方式）权重越高表示该视差候选值更可能是正确的
                enhanced_cost_volume = cost_volume * (1.0 - self.alpha) + cost_volume * weight_volume * self.alpha # 添加可调节的融合系数的策略
            return enhanced_cost_volume

        return cost_volume