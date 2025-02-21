import torch
import torch.nn as nn
import torch.nn.functional as F
from emergence import (
    MultiScaleEmergenceModule,
    EmergenceCore,
    CrossModalAttention,
    ScaleInteractionModule,
    BidirectionalEmergenceCore
)

class DualEmergenceOptimizer(nn.Module):
    def __init__(self, feature_dim, temperature=0.1, alpha=0.5, consistency_threshold=0.05, 
                 patience=3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.below_threshold_count = 0  
        self.should_stop = False
        # 复用emergence.py中的核心组件
        self.emergence_core = EmergenceCore(feature_dim)
        self.cross_modal_attention = CrossModalAttention(feature_dim)
        self.scale_interaction = ScaleInteractionModule([feature_dim, feature_dim * 2])
        self.bidirectional_core = BidirectionalEmergenceCore(feature_dim)
        
        # E-Step: 分布估计
        self.distribution_estimator = CriticalDistributionEstimator(
            feature_dim, self.emergence_core, self.cross_modal_attention
        )
        
        # M-Step: 参数优化
        self.parameter_optimizer = AdaptiveParameterOptimizer(
            feature_dim, self.scale_interaction
        )
        
        # 双向一致性损失
        self.consistency_loss = SymmetricConsistencyLoss()

    def check_consistency(self, loss_value): 
        if loss_value < self.consistency_threshold:
            self.below_threshold_count += 1
            if self.below_threshold_count >= self.patience:
                self.should_stop = True
        else:
            self.below_threshold_count = 0
        return self.should_stop

    def forward(self, text_features, image_features):
        # 1. E-Step: 估计分布
        distribution, emergence_state = self.distribution_estimator(text_features, image_features)
        
        # 2. M-Step: 优化参数
        current_params = torch.cat([text_features, image_features], dim=-1)
        optimized_params, scale_weights = self.parameter_optimizer(distribution, current_params)
        
        # 3. 拆分参数
        split_size = text_features.size(-1)
        text_params, image_params = torch.split(optimized_params, [split_size, split_size], dim=-1)
        
        # 4. 双向涌现
        text_emerged, image_emerged = self.bidirectional_core(text_params, image_params)
        
        # 5. 计算一致性损失
        consistency_loss = self.consistency_loss(text_emerged, image_emerged, distribution)
        
        # 6. 加权损失
        emergence_state = torch.sigmoid(emergence_state)
        emergence_weighted_loss = consistency_loss * emergence_state
        
        should_stop = self.check_consistency(consistency_loss.item()
        return text_emerged, image_emerged, emergence_weighted_loss, consistency_loss, should_stop


class CriticalDistributionEstimator(nn.Module):
    """E-Step: 估计当前分布"""
    def __init__(self, feature_dim, emergence_core, cross_attention):
        super().__init__()
        self.emergence_core = emergence_core
        self.cross_attention = cross_attention
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, text_features, image_features):
        # 使用emergence_core检测涌现状态
        emergence_state = self.emergence_core(torch.cat([text_features, image_features], dim=-1))
        
        # 使用cross_attention计算模态间注意力
        attn_output = self.cross_attention(text_features, image_features)
        
        # 根据涌现状态调整分布
        temp = torch.clamp(self.temperature, min=1e-3)
        distribution = F.softmax(attn_output / (temp * emergence_state), dim=-1)
        
        return distribution, emergence_state


class AdaptiveParameterOptimizer(nn.Module):
    """M-Step: 优化参数"""
    def __init__(self, feature_dim, scale_interaction):
        super().__init__()
        self.scale_interaction = scale_interaction
        
        # 参数预测网络
        self.param_predictor = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, distribution, current_params):
        # 使用scale_interaction进行多尺度特征融合
        scale_weights = self.scale_interaction([distribution, current_params])
        
        # 预测参数更新
        param_update = self.param_predictor(torch.cat([distribution, current_params], dim=-1))
        
        # 应用尺度权重
        optimized_params = current_params + scale_weights * param_update
        
        return optimized_params, scale_weights


class SymmetricConsistencyLoss(nn.Module):
    """双向一致性损失"""
    def __init__(self):
        super().__init__()
        
    def forward(self, text_features, image_features, distribution):
        # 计算双向 KL 散度
        text_dist = F.softmax(text_features, dim=-1)
        img_dist = F.softmax(image_features, dim=-1)
        m = 0.5 * (text_dist + img_dist)
        jsd = 0.5 * (F.kl_div(text_dist.log(), m, reduction='batchmean') +
                     F.kl_div(img_dist.log(), m, reduction='batchmean'))
        return jsd

