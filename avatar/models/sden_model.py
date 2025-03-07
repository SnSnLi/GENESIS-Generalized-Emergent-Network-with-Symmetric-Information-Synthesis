import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_modules import (
    MultiHeadAttention,
    EmergenceCore,
    BidirectionalEmergenceCore,
    CrossModalAttention
)
from .emergence import MultiScaleEmergenceModule
from .topology import (
    DynamicTopologyCoupler,
    EntropyController,
    CriticalDynamicsController
)
from .dual import DualEmergenceOptimizer

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
        
        self.emergence_module = MultiScaleEmergenceModule(dims=[dim, dim*2, dim*4])
        self.bidirectional_core = BidirectionalEmergenceCore(dim=dim)
        self.cross_modal = CrossModalAttention(dim=dim)
        
        self.topology_coupler = DynamicTopologyCoupler(dim=dim, num_heads=num_heads)
        self.entropy_controller = EntropyController(dim=dim)
        self.critical_controller = CriticalDynamicsController(dim=dim)
        
        self.dual_optimizer = DualEmergenceOptimizer(dim=dim)
        
        self.classifier = nn.Linear(dim, dim)

        # 新增对比学习温度参数
        self.contrastive_temp = nn.Parameter(torch.ones(1) * 0.07)

    def contrastive_loss(self, text_feat, image_feat, entropy_weights):
        # 计算对比损失，熵权重调整正负样本的贡献
        text_norm = F.normalize(text_feat, dim=-1)
        image_norm = F.normalize(image_feat, dim=-1)
        logits = torch.matmul(text_norm, image_norm.T) / self.contrastive_temp
        labels = torch.arange(text_norm.size(0)).to(text_norm.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        return contrastive_loss * entropy_weights.mean()

    def emergence_forward(self, text_features, image_features):
        """多尺度涌现前向传播"""
        # 先通过 BidirectionalEmergenceCore 获取独立的文本和图像特征
        text_emerged, image_emerged = self.bidirectional_core(text_features, image_features)
        # 调用 MultiScaleEmergenceModule，获取多尺度特征和熵权重
        final_text, final_image, global_emerged, entropy_weights = self.emergence_module(text_emerged, image_emerged)
        # 融合文本和图像特征
        fused_features = self.cross_modal(final_text, final_image)
        return fused_features, final_text, final_image, entropy_weights
        
    def topology_forward(self, emerged_features, entropy_weights):
        """动态拓扑耦合前向传播"""
        topo_output = self.topology_coupler(emerged_features)
        # 使用熵控制进一步调整特征
        controlled_features, topo_entropy_weights = self.entropy_controller(topo_output['output'])
        # 结合多尺度熵权重和拓扑熵权重
        combined_entropy_weights = (entropy_weights + topo_entropy_weights) / 2
        critical_features = self.critical_controller(controlled_features)
        return {
            'features': critical_features,
            'entropy_weights': combined_entropy_weights,
            'adj_matrix': topo_output['adj_matrix']
        }
        
    def dual_forward(self, topo_features, labels=None):
        """对偶优化前向传播"""
        optimized_features = self.dual_optimizer(topo_features['features'], labels=labels if self.training else None)
        logits = self.classifier(optimized_features)
        return logits, optimized_features
        
    def forward(self, text_features=None, image_features=None, labels=None):
        # 检查输入模态
        if text_features is not None and image_features is not None:
            # 双向涌现（训练时使用）
            fused_features, final_text, final_image, entropy_weights = self.emergence_forward(
                text_features, image_features
            )
            topo_features = self.topology_forward(fused_features, entropy_weights)
            entropy_ranking = torch.argsort(topo_features['entropy_weights'], dim=-1)
            logits, final_features = self.dual_forward(topo_features, labels)
            
            if self.training:
                # 训练时的损失计算
                text_emerged = self.forward_text(text_features)
                image_emerged = self.forward_image(image_features)
                consistency_loss = -F.cosine_similarity(
                    text_emerged.mean(dim=1), image_emerged.mean(dim=1)
                ).mean()
                contrastive_loss = self.contrastive_loss(final_text, final_image, topo_features['entropy_weights'])
                total_loss = consistency_loss + contrastive_loss
            else:
                total_loss = None
        elif text_features is not None:
            # 单向涌现（仅文本）
            text_emerged = self.forward_text(text_features)
            topo_features = self.topology_forward(text_emerged, torch.ones_like(text_emerged[..., 0]))
            entropy_ranking = torch.argsort(topo_features['entropy_weights'], dim=-1)
            logits, final_features = self.dual_forward(topo_features, labels)
            total_loss = None
        elif image_features is not None:
            # 单向涌现（仅图像）
            image_emerged = self.forward_image(image_features)
            topo_features = self.topology_forward(image_emerged, torch.ones_like(image_emerged[..., 0]))
            entropy_ranking = torch.argsort(topo_features['entropy_weights'], dim=-1)
            logits, final_features = self.dual_forward(topo_features, labels)
            total_loss = None
        else:
            raise ValueError("At least one of text_features or image_features must be provided")
    
        return {
            'emergence_features': {
                'final_features': final_features,
                'emerged_raw': fused_features if text_features is not None and image_features is not None else None,
            },
            'auxiliary_features': {
                'topo_features': topo_features['features'],
                'entropy_weights': topo_features['entropy_weights'],
                'adjacency': topo_features['adj_matrix'],
                'entropy_ranking': entropy_ranking
            },
            'logits': logits,
            'consistency_loss': total_loss
        }

    def forward_text(self, text_features):
        """单模态文本涌现"""
        text_emerged = self.bidirectional_core.text_emergence(text_features)
        return text_emerged
    
    def forward_image(self, image_features):
        """单模态图像涌现"""
        image_emerged = self.bidirectional_core.image_emergence(image_features)
        return image_emerged

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

    def set_inference_mode(self, mode=True):
        self.dual_optimizer.inference_mode = mode

class SDENModel(nn.Module):
    def __init__(self, feature_dim=512, temperature=0.1):
        super().__init__()
        self.dual_optimizer = DualEmergenceOptimizer(feature_dim, temperature)
        self.emergence = SymmetricDynamicEmergenceNetwork(dim=feature_dim)
        self.is_training = True  # 添加训练标志
        
    def set_training_mode(self, mode=True):
        self.is_training = mode
        self.dual_optimizer.is_training = mode
        self.emergence.is_training = mode
        
    def forward(self, x):
        # 根据is_training标志设置模式
        self.set_training_mode(self.is_training)
        return self.dual_optimizer(x)