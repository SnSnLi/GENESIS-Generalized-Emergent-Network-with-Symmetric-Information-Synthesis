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
from distill import EmergentKnowledgeDistillation

class SymmetricDynamicEmergenceNetwork(nn.Module):
    def __init__(
        self,
        dim=512,
        num_heads=8,
        num_layers=4,
        temperature=0.1,
        use_distill=True
    ):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        
        # 1. 多尺度涌现模块
        self.emergence_module = MultiScaleEmergenceModule(
            dims=[dim, dim*2, dim*4]
        )
        self.bidirectional_core = BidirectionalEmergenceCore(
            dim=dim
        )
        self.cross_modal = CrossModalAttention(
            dim=dim
        )
        
        # 2. 动态拓扑耦合模块
        self.topology_coupler = DynamicTopologyCoupler(
            dim=dim,
            num_heads=num_heads
        )
        self.entropy_controller = EntropyController(
            dim=dim
        )
        self.critical_controller = CriticalDynamicsController(
            dim=dim
        )
        
        # 3. 对偶优化模块
        self.dual_optimizer = DualEmergenceOptimizer(
            dim=dim
        )
        
        # 4. 知识蒸馏模块(可选)
        self.use_distill = use_distill
        if use_distill:
            self.distiller = EmergentKnowledgeDistillation(
                teacher_model=self,  # 将自身作为教师模型
                student_model=None,  # 学生模型需要外部设置
                feature_dim=dim,
                temperature=temperature
            )
        
        # 5. 输出头
        self.classifier = nn.Linear(dim, dim)
        
    def emergence_forward(self, text_features, image_features):
        """多尺度涌现前向传播"""
        # 1. 双向涌现特征提取
        text_emerged, image_emerged = self.bidirectional_core(
            text_features, image_features
        )
        
        # 2. 跨模态注意力
        fused_features = self.cross_modal(
            text_emerged, image_emerged
        )
        
        # 3. 多尺度涌现
        multi_scale_features = self.emergence_module(fused_features)
        
        return multi_scale_features
        
    def topology_forward(self, emerged_features):
        """动态拓扑耦合前向传播"""
        # 1. 动态拓扑耦合
        topo_output = self.topology_coupler(emerged_features)
        
        # 2. 熵控制
        controlled_features, entropy_weights = self.entropy_controller(
            topo_output['output']
        )
        
        # 3. 临界动力学控制
        critical_features = self.critical_controller(controlled_features)
        
        return {
            'features': critical_features,
            'entropy_weights': entropy_weights,
            'adj_matrix': topo_output['adj_matrix']
        }
        
    def dual_forward(self, topo_features, labels=None):
        """对偶优化前向传播"""
        # 1. 特征优化
        optimized_features = self.dual_optimizer(
            topo_features['features'],
            labels=labels if self.training else None
        )
        
        # 2. 分类预测
        logits = self.classifier(optimized_features)
        
        return logits, optimized_features
        
    def forward(self, text_features, image_features, labels=None, student_model=None):
        """完整的前向传播流程"""
        # 1. 多尺度涌现
        emerged_features = self.emergence_forward(
            text_features, image_features
        )
        
        # 2. 动态拓扑耦合
        topo_features = self.topology_forward(emerged_features)
        
        # 3. 对偶优化
        logits, final_features = self.dual_forward(
            topo_features, labels
        )
        
        # 4. 知识蒸馏(如果启用)
        if self.training and self.use_distill and student_model is not None:
            self.distiller.student = student_model
            distill_output = self.distiller(
                inputs=(text_features, image_features),
                labels=labels
            )
            return {
                'logits': logits,
                'features': final_features,
                'topo_features': topo_features,
                'distill_loss': distill_output['loss'],
                'student_logits': distill_output['student_logits']
            }
            
        return {
            'logits': logits,
            'features': final_features,
            'topo_features': topo_features
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
        return output['features']