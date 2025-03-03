import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from emergence import MultiHeadAttention, EmergenceCore, BidirectionalEmergenceCore

class PhaseMapper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mapper = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        
    def forward(self, x):
        return self.mapper(x)

class QuantumFluctuationAttention(nn.Module):
    def __init__(self, dim, heads=8, temperature=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        noise = torch.randn_like(q) * torch.sigmoid(self.temperature)
        q = q + noise
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim_head)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(out)

class CriticalDynamicsController(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.order_param = nn.Parameter(torch.ones(1))
        self.controller = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
    def forward(self, x):
        critical_state = self.controller(x)
        return critical_state * torch.sigmoid(self.order_param)

class AdaptiveGraphGenerator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        self.edge_predictor = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.threshold = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x):
        node_feats = self.node_encoder(x)
        batch_size, num_nodes, _ = node_feats.shape
        node_pairs = torch.cat([
            node_feats.unsqueeze(2).expand(-1, -1, num_nodes, -1),
            node_feats.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        ], dim=-1)
        edge_probs = F.gumbel_softmax(self.edge_predictor(node_pairs), tau=1, hard=False)
        adj_matrix = (edge_probs > self.threshold).float()
        return adj_matrix, node_feats

class EntropyController(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.entropy_estimator = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.LayerNorm(dim//2),
            nn.GELU(),
            nn.Linear(dim//2, 1)
        )
        self.target_entropy = nn.Parameter(torch.ones(1) * math.log(dim))
        self.temperature = nn.Parameter(torch.ones(1))
        
    def compute_entropy(self, features):
        kernel = torch.exp(-torch.cdist(features, features) ** 2 / (2 * self.temperature ** 2))
        probs = kernel / kernel.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
        return entropy
    
    def forward(self, features, adj_matrix=None):
        current_entropy = self.compute_entropy(features)
        estimated_entropy = self.entropy_estimator(features)
        entropy_diff = current_entropy - self.target_entropy
        control_signal = torch.sigmoid(-entropy_diff / self.temperature)
        controlled_features = features * control_signal.unsqueeze(-1)
        if adj_matrix is not None:
            controlled_adj = adj_matrix * control_signal.unsqueeze(-1).unsqueeze(-1)
            return controlled_features, control_signal, controlled_adj
        return controlled_features, control_signal

class DynamicTopologyCoupler(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.phase_mapper = PhaseMapper(dim)
        self.graph_gen = AdaptiveGraphGenerator(dim)
        self.entropy_ctrl = EntropyController(dim)
        self.feature_extractor = BidirectionalEmergenceCore(dim)
        self.mha = MultiHeadAttention(dim, num_heads)
        self.quantum_attention = QuantumFluctuationAttention(dim)
        self.critical_controller = CriticalDynamicsController(dim)
        self.fusion_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.fusion_gate = nn.Linear(dim * 2, dim)  # 添加融合门控网络

    def forward(self, text_feat=None, image_feat=None):
        # 单模态或多模态处理
        if text_feat is not None and image_feat is not None:
            # 多模态联合涌现
            text_final, image_final = self.feature_extractor(text_feat, image_feat)
            x = (text_final + image_final) / 2
        elif text_feat is not None:
            # 单模态（文本）
            x = text_feat
        elif image_feat is not None:
            # 单模态（图像）
            x = image_feat
        else:
            raise ValueError("At least one of text_feat or image_feat must be provided")

        # 相空间映射
        phase_features = self.phase_mapper(x)
        
        # 动态图生成
        adj_matrix, node_features = self.graph_gen(phase_features)
        
        # 熵控制
        controlled_features, entropy_weights, controlled_adj = self.entropy_ctrl(node_features, adj_matrix)
        
        # 注意力机制
        mha_output, _ = self.mha(controlled_features)
        quantum_output = self.quantum_attention(controlled_features)
        
        # 自适应特征融合
        w = torch.sigmoid(self.fusion_weight)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([mha_output, quantum_output], dim=-1)))
        fused_features = gate * mha_output + (1 - gate) * quantum_output
        
        # 临界动力学控制
        critical_state = self.critical_controller(fused_features)
        
        return {
            'output': critical_state,
            'text_features': text_final if text_feat is not None and image_feat is not None else None,
            'image_features': image_final if text_feat is not None and image_feat is not None else None,
            'adj_matrix': controlled_adj,
            'entropy_weights': entropy_weights,
            'attention_weights': w.item()
        }

def build_topology_network(dim=256, num_heads=8):
    return DynamicTopologyCoupler(dim, num_heads)