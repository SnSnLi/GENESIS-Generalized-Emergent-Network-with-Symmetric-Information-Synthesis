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
        
        # 添加量子噪声
        noise = torch.randn_like(q) * torch.sigmoid(self.temperature)
        q = q + noise
        
        # 计算注意力
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
        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        
        # 边预测器
        self.edge_predictor = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # 动态阈值
        self.threshold = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x):
        # 1. 节点编码
        node_feats = self.node_encoder(x)
        
        # 2. 生成边概率
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
        # 使用核密度估计计算熵
        kernel = torch.exp(-torch.cdist(features, features) ** 2 / (2 * self.temperature ** 2))
        probs = kernel / kernel.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
        return entropy
    
    def forward(self, features, adj_matrix=None):
        # 1. 计算当前熵
        current_entropy = self.compute_entropy(features)
        estimated_entropy = self.entropy_estimator(features)
        
        # 2. 计算熵差
        entropy_diff = current_entropy - self.target_entropy
        
        # 3. 生成控制信号
        control_signal = torch.sigmoid(-entropy_diff / self.temperature)
        
        # 4. 应用熵控制
        controlled_features = features * control_signal.unsqueeze(-1)
        
        if adj_matrix is not None:
            # 如果提供邻接矩阵,同时调整图结构
            controlled_adj = adj_matrix * control_signal.unsqueeze(-1).unsqueeze(-1)

            return controlled_features, control_signal, controlled_adj
        
        return controlled_features, control_signal

class DynamicTopologyCoupler(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # 基础组件
        self.phase_mapper = PhaseMapper(dim)
        self.graph_gen = AdaptiveGraphGenerator(dim)
        self.entropy_ctrl = EntropyController(dim)
        self.feature_extractor = BidirectionalEmergenceCore(dim)
        # 注意力机制
        self.mha = MultiHeadAttention(dim, num_heads)  # 从emergence.py导入
        self.quantum_attention = QuantumFluctuationAttention(dim)
        
        # 临界动力学控制
        self.critical_controller = CriticalDynamicsController(dim)
        
        # 特征融合权重
        self.fusion_weight = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, text_feat, image_feat):
        text_final, image_final = self.feature_extractor(text_feat, image_feat)
        x = (text_final + image_final) / 2
        # 1. 相空间映射
        phase_features = self.phase_mapper(x)
        
        # 2. 动态图生成
        adj_matrix, node_features = self.graph_gen(phase_features)
        
        # 3. 熵控制
        controlled_features, entropy_weights, controlled_adj = self.entropy_ctrl(
            node_features, adj_matrix
        )
       
        mha_output, _ = self.mha(controlled_features)
        #  量子涨落注意力
        quantum_output = self.quantum_attention(controlled_features)
        #  自适应特征融合
        w = torch.sigmoid(self.fusion_weight)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([mha_output, quantum_output], dim=-1)))
        fused_features = gate * mha_output + (1 - gate) * quantum_output
        
        #  临界动力学控制
        critical_state = self.critical_controller(fused_features)
        
        return {
            'output': critical_state,
            'text_features': text_final,
            'image_features': image_final,
            'adj_matrix': controlled_adj,
            'entropy_weights': entropy_weights,
            'attention_weights': w.item()
        }

def build_topology_network(dim=256, num_heads=8):
    return DynamicTopologyCoupler(dim, num_heads)