import sys
import os
sys.path.append('/root/autodl-tmp')
import torch
import torch.nn as nn
import torch.nn.functional as F

class CAM(nn.Module):
    def __init__(self, units):
        super(CAM, self).__init__()
        self.cam = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.Sigmoid()
        )
    def forward(self, cv):
        return self.cam(cv)
    
class LapEdgeExtractor(nn.Module):
    def __init__(self):
        super(LapEdgeExtractor, self).__init__()
        self.kernel_2d = torch.tensor([[1., 4., 6., 4., 1],
                                [4., 16., 24., 16., 4.],
                                [6., 24., 36., 24., 6.],
                                [4., 16., 24., 16., 4.],
                                [1., 4., 6., 4., 1.]])

    def gauss_kernel_3d(self, channels=1):
        """Create 3D Gaussian kernel"""
        kernel_2d = self.kernel_2d / 256.
        # Expand to 3D
        kernel_3d = kernel_2d.unsqueeze(0).repeat(5, 1, 1) / 5
        kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1, 1)
        return kernel_3d

    def conv_gauss_3d(self, volume, kernel):
        """Convolve 3D volume with Gaussian kernel"""
        volume = F.pad(volume, (2, 2, 2, 2, 2, 2), mode='replicate')
        out = F.conv3d(volume, kernel, groups=volume.shape[1])
        return out
    
    def make_laplace_3d(self, volume):
        """Create 3D Laplacian edge map"""
        channels = volume.shape[1]
        kernel_3d = self.gauss_kernel_3d(channels).to(volume.device)
        filtered = self.conv_gauss_3d(volume, kernel_3d)
        diff = volume - filtered
        return diff
    
    def forward(self, cv):
        edge_features = self.make_laplace_3d(cv)
        return edge_features

class EECF(nn.Module):
    def __init__(self, units, cascade_num=3):
        super(EECF, self).__init__()
        self.cascade_num = cascade_num
        # Original fusion modules
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
        self.add = lambda x, y: x + y
        self.multiply = lambda x, y: x * y
        self.edge_extractor = LapEdgeExtractor()
        # Create dynamic CAM modules based on cascade_num
        self.cam_paths = nn.ModuleList([CAM(units) for _ in range(cascade_num)])
        # Edge CAM只需要cascade_num-1个，因为最后一次不需要edge enhancement
        self.cam_edges = nn.ModuleList([CAM(units) for _ in range(max(1, cascade_num))])  # 至少保留1个

    def fusion(self, v, cv1, cv2):
        shape = cv1.shape
        v1 = 1.0 - v
        v = v.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        v1 = v1.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        v = v.expand(-1, -1, shape[2], shape[3], shape[4])
        v1 = v1.expand(-1, -1, shape[2], shape[3], shape[4])
        
        x3 = self.multiply(cv1, v)
        x4 = self.multiply(cv2, v1)
        fusion_cv = self.add(x3, x4)

        return fusion_cv
    
    def edge_enhanced(self, v, cv):
        shape = cv.shape
        edge_attn = v.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        edge_attn = edge_attn.expand(-1, -1, shape[2], shape[3], shape[4])
        weighted_edges = self.multiply(cv, edge_attn)
        edge_enhanced_cv = self.add(cv, weighted_edges)

        return edge_enhanced_cv

    def forward(self, inputs):
        assert len(inputs) == 2
        low_res_cv, high_res_cv = inputs[0], inputs[1]
        x1 = self.upsample(low_res_cv)  # [N, C, D/16, H/16, W/16] --> [N, C, D/8, H/8, W/8]
        # Initialize variables for cascading
        current_fusion = None
        current_edge_enhanced = high_res_cv
        # Extract edge features once
        edge_features = self.edge_extractor(high_res_cv)
        
        # Cascade loop
        for i in range(self.cascade_num):
            if i == 0:
                # First iteration: fusion between upsampled low_res and high_res
                v_path = self.cam_paths[i](self.add(x1, high_res_cv))
                current_fusion = self.fusion(v_path, x1, high_res_cv)
                
                # Edge enhancement
                v_edge = self.cam_edges[i](edge_features)
                current_edge_enhanced = self.edge_enhanced(v_edge, high_res_cv)
            else:
                # Subsequent iterations: fusion between previous fusion and edge enhanced
                v_path = self.cam_paths[i](self.add(current_fusion, current_edge_enhanced))
                new_fusion = self.fusion(v_path, current_fusion, current_edge_enhanced)
                
                # Edge enhancement (only if not the last iteration)
                if i < self.cascade_num - 1:
                    v_edge = self.cam_edges[i](current_edge_enhanced)
                    current_edge_enhanced = self.edge_enhanced(v_edge, current_edge_enhanced)
                
                current_fusion = new_fusion
        
        return current_fusion