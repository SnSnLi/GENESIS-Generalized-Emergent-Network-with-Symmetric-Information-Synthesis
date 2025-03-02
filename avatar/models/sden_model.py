import torch
import torch.nn as nn
from emergence import (
    MultiScaleEmergenceModule,
    BidirectionalEmergenceCore,
    CrossModalAttention
)
from topology import (
    DynamicTopologyCoupler,
    EntropyController,
    CriticalDynamicsController
)
from dual import DualEmergenceOptimizer

class SymmetricDynamicEmergenceNetwork(nn.Module):
    def __init__(
        self,
        dim=512,
        num_heads=8,
        num_layers=4,
        temperature=0.1
    ):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        
        # 1. 多尺度涌现模块
        self.emergence_module = MultiScaleEmergenceModule(
            dims=[dim, dim*2, dim*4]
        )
        self.bidirectional_core = BidirectionalEmergenceCore(dim=dim)
        self.cross_modal = CrossModalAttention(dim=dim)
        
        # 2. 动态拓扑耦合模块
        self.topology_coupler = DynamicTopologyCoupler(dim=dim, num_heads=num_heads)
        self.entropy_controller = EntropyController(dim=dim)
        self.critical_controller = CriticalDynamicsController(dim=dim)
        
        # 3. 对偶优化模块
        self.dual_optimizer = DualEmergenceOptimizer(dim=dim)
        
        # 4. 输出头
        self.classifier = nn.Linear(dim, dim)

    def emergence_forward(self, text_features, image_features):
        """多尺度涌现前向传播"""
        text_emerged, image_emerged = self.bidirectional_core(text_features, image_features)
        fused_features = self.cross_modal(text_emerged, image_emerged)
        multi_scale_features = self.emergence_module(fused_features)
        return multi_scale_features
        
    def topology_forward(self, emerged_features):
        """动态拓扑耦合前向传播"""
        topo_output = self.topology_coupler(emerged_features)
        controlled_features, entropy_weights = self.entropy_controller(topo_output['output'])
        critical_features = self.critical_controller(controlled_features)
        return {
            'features': critical_features,
            'entropy_weights': entropy_weights,
            'adj_matrix': topo_output['adj_matrix']
        }
        
    def dual_forward(self, topo_features, labels=None):
        """对偶优化前向传播"""
        optimized_features = self.dual_optimizer(topo_features['features'], labels=labels if self.training else None)
        logits = self.classifier(optimized_features)
        return logits, optimized_features
        
    def forward(self, text_features, image_features, labels=None):
        # 1. 多尺度涌现
        emerged_features = self.emergence_forward(text_features, image_features)
        
        # 2. 动态拓扑耦合
        topo_features = self.topology_forward(emerged_features)
        entropy_ranking = torch.argsort(topo_features['entropy_weights'], dim=-1)
        
        # 3. 对偶优化
        logits, final_features = self.dual_forward(topo_features, labels)
        
        return {
            'emergence_features': {
                'final_features': final_features,
                'emerged_raw': emerged_features,
            },
            'auxiliary_features': {
                'topo_features': topo_features['features'],
                'entropy_weights': topo_features['entropy_weights'],
                'adjacency': topo_features['adj_matrix'],
                'entropy_ranking': entropy_ranking
            },
            'logits': logits
        }

    def get_features(self):
        """获取中间特征用于分析或可视化"""
        return {
            'emerged_features': self.emerged_features,
            'topo_features': self.topo_features,
            'final_features': self.final_features
        }

    @torch.no_grad()
    def extract_features(self, text_features, image_features):
        """提取特征(无梯度)"""
        self.eval()
        output = self.forward(text_features, image_features)
        return output['emergence_features']['final_features']