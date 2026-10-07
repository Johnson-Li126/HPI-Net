import torch
import torch.nn as nn

class Estimation_prob_var(nn.Module):
    def __init__(self, min_disp=-112.0, max_disp=16.0, filters=16):
        """
        初始化Estimation模型，用于估计深度图。
        :param min_disp: 最小视差值，默认为-112.0。
        :param max_disp: 最大视差值，默认为16.0。
        """
        super(Estimation_prob_var, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.conv = nn.Conv3d(in_channels=filters, out_channels=1, kernel_size=3,
                           stride=1, padding=1, bias=False)  # 使用He初始化
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, inputs):
        x = self.conv(inputs)  # [N, 1, D, H, W]
        x = torch.squeeze(x, dim=1)  # [N, D, H, W]
        x = x.permute(0, 2, 3, 1)  # [N, H, W, D]
        assert x.shape[-1] == self.max_disp - self.min_disp, "视差范围不匹配"
        
        # 生成候选视差值序列
        candidates = torch.linspace(start=self.min_disp, end=self.max_disp - 1, steps=self.max_disp - self.min_disp)
        # 计算softmax概率分布
        probabilities = torch.softmax(-x, dim=-1)
        # 根据概率分布计算加权平均视差值
        disparities = torch.sum(candidates.to(x.device) * probabilities.to(x.device), dim=-1, keepdim=True)  # [N, H, W, 1]
        # 根据概率分布和候选视差序列计算候选视差方差和标准差
        variances = torch.sum(probabilities.to(x.device) * (candidates.to(x.device) - disparities)**2, dim=-1, keepdim=True)
        return disparities, probabilities, variances