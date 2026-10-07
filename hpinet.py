import sys
import os
sys.path.append('/data/ljx/DE-DG-0203/Code')
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from Code.model.HPINet.feature import FeatureExtraction
from Code.model.HPINet.ddac import DDAC
from Code.model.HPINet.aggregation import Hourglass
from Code.model.HPINet.computation import Estimation_prob_var
from Code.model.HPINet.dejr import DEJR
from Code.model.HPINet.eecf import EECF
import os
import numpy as np

class HPINet(nn.Module):
    def __init__(self, height=1024, width=1024, channel=1, min_disp=-128.0, max_disp=64.0, filters=16, cv_type='gwc', cascade_num=2):
        super(HPINet, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.filter = filters
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
    
        self.feature_extraction = FeatureExtraction(in_c=channel, filters=filters)
        self.cost_concatenation_0 = DDAC(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4, filters=filters, cv=cv_type)
        self.cost_concatenation_1 = DDAC(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8, filters=filters, cv=cv_type)
        self.cost_concatenation_2 = DDAC(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16, filters=filters, cv=cv_type)
        self.hourglass_0 = Hourglass(filters=filters)
        self.hourglass_1 = Hourglass(filters=filters)
        self.hourglass_2 = Hourglass(filters=filters)
        self.estimation_2 = Estimation_prob_var(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16, filters=filters)
        self.feature_fusion_1 = EECF(units=filters, cascade_num=cascade_num)
        self.hourglass_3 = Hourglass(filters=filters)
        self.estimation_1 = Estimation_prob_var(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8, filters=filters)
        self.feature_fusion_2 = EECF(units=filters, cascade_num=cascade_num)
        self.hourglass_4 = Hourglass(filters=filters)
        self.estimation_0 = Estimation_prob_var(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4, filters=filters)
        self.refiner = DEJR(in_c=1+1+channel*4, out_c=1, filters=filters)
        


    def forward(self, left_image, right_image, gx, gy):
        bs, c,_ ,_ = left_image.shape
        if c != self.channel and c == 1:
            left_image = left_image.expand(-1, 3, -1, -1)
            right_image = right_image.expand(-1, 3, -1, -1)
            gx = gx.expand(-1, 3, -1, -1)
            gy = gy.expand(-1, 3, -1, -1)
        l0, l1, l2 = self.feature_extraction(left_image)  # (B, C, 1024, 1024) --> (B, 16, 256, 256), (B, 16, 128, 128), (B, 16, 64, 64)
        r0, r1, r2 = self.feature_extraction(right_image)
        
        cost_volume2 = self.cost_concatenation_2([l2, r2])  # (1, 16, 8, 64, 64)  16/16-(-112/16)
        agg_cost2 = self.hourglass_2(cost_volume2)
        disparity2, prob2, var2 = self.estimation_2(agg_cost2)  # (N, H/16, W/16, 1)

        cost_volume1 = self.cost_concatenation_1([l1, r1], disparity2, prob2, var2)  # (1, 16, 16, 128, 128)  16/8-(-112/8)
        agg_cost1 = self.hourglass_1(cost_volume1)
        fusion_cost1 = self.feature_fusion_1([agg_cost2, agg_cost1])
        agg_fusion_cost1 = self.hourglass_3(fusion_cost1)
        disparity1, prob1, var1 = self.estimation_1(agg_fusion_cost1)  # (N, H/8, W/8, 1)

        cost_volume0 = self.cost_concatenation_0([l0, r0], disparity1, prob1, var1)  # (1, 16, 32, 256, 256)  16/4-(-112/4) 
        agg_cost0 = self.hourglass_0(cost_volume0)
        fusion_cost0 = self.feature_fusion_2([agg_fusion_cost1, agg_cost0])
        agg_fusion_cost0 = self.hourglass_4(fusion_cost0)
        disparity0, prob0, var0 = self.estimation_0(agg_fusion_cost0)  # (N, H/4, W/4, 1)

        final_disp, error_map = self.refiner([disparity0, left_image, right_image, l0, gx, gy, var0])  # HDDANet3_whu_new1

        return disparity2, disparity1, disparity0, final_disp, error_map
    

if __name__ == '__main__':
    
    left = torch.rand(2, 1, 1024, 1024).cuda()
    right = torch.rand(2, 1, 1024, 1024).cuda()
    gx = torch.rand(2, 1, 1024, 1024).cuda()
    gy = torch.rand(2, 1, 1024, 1024).cuda()
    model = HPINet(left.shape[2], right.shape[3], left.shape[1]).cuda()
    _, _, _, final_disp, error_map = model(left, right, gx, gy)
    print(final_disp.shape)