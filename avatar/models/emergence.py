import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.query_transform = nn.Linear(dim, dim)
        self.key_transform = nn.Linear(dim, dim)
        self.value_transform = nn.Linear(dim, dim)
        self.final_linear = nn.Linear(dim, dim)
        
    def forward(self, text_feat, image_feat):
        batch_size, seq_len, _ = text_feat.size()
        _, num_regions, _ = image_feat.size()

        text_query = self.query_transform(text_feat).view(batch_size, seq_len, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        image_key = self.key_transform(image_feat).view(batch_size, num_regions, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        image_value = self.value_transform(image_feat).view(batch_size, num_regions, self.num_heads, self.dim // self.num_heads).transpose(1, 2)

        attention_scores = torch.matmul(text_query, image_key.transpose(-2, -1)) / (self.dim // self.num_heads) ** 0.5
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, image_value).transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.final_linear(weighted_values)
        return output + text_feat

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.proj(out), attn
        
class PhaseTransitionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(dim*2, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim*2),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, 1)
        )
        self.temperature = nn.Parameter(torch.tensor([0.1]))
        
    def forward(self, text_feat, image_feat):
        joint_feat = torch.cat([text_feat, image_feat], dim=-1)
        energy = self.energy_net(joint_feat) / self.temperature
        return torch.sigmoid(energy)

class LocalFeatureAligner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.phase_transition = PhaseTransitionLayer(dim)
        self.cross_attention = CrossModalAttention(dim)
        
    def forward(self, text_feat, image_feat):
        energy = self.phase_transition(text_feat, image_feat)
        aligned_feat = self.cross_attention(text_feat, image_feat) * energy
        return aligned_feat

class DynamicTopoNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.graph_gen = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        self.attention = nn.MultiheadAttention(dim, num_heads=8)
        
    def forward(self, x):
        x_permuted = x.permute(1, 0, 2)  # [seq_len, batch, dim]
        attn_out, _ = self.attention(x_permuted, x_permuted, x_permuted)
        attn_out = attn_out.permute(1, 0, 2)  # 恢复原始维度 [batch, seq, dim]
        adj_matrix = torch.matmul(self.graph_gen(attn_out), x.transpose(-2, -1))
        return F.softmax(adj_matrix, dim=-1)

class EntropyGateLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim//2), 
            nn.ReLU(),
            nn.Linear(dim//2, 1)   
        )
        
    def forward(self, graph):
        entropy = -torch.sum(graph * torch.log(graph.clamp(min=1e-10)), dim=-1, keepdim=True)
        gate = torch.sigmoid(self.mlp(entropy))
        return graph * gate

class SemanticGraphBuilder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dynamic_topology = DynamicTopoNet(dim)
        self.entropy_gate = EntropyGateLayer(dim)
        
    def forward(self, local_feat):
        initial_graph = self.dynamic_topology(local_feat)
        evolved_graph = self.entropy_gate(initial_graph)
        return evolved_graph

class EmergenceCore(nn.Module):
    """核心涌现模块 - 作为基础组件"""
    def __init__(self, dim):
        super().__init__()
        self.phase_transition = PhaseTransitionLayer(dim)
        self.critical_transformer = CriticalTransformer(dim)
        self.emergence_predictor = EmergencePredictor(dim)
        self.cross_modal = CrossModalAttention(dim)
        
    def forward(self, x, context=None):
        # 相变层
        if context is not None:
            phase_state = self.phase_transition(x, context).expand_as(x)
            x = x * phase_state
            
        # 临界转换
        critical_state = self.critical_transformer(x)
        
        # 如果有上下文，进行跨模态注意力
        if context is not None:
            critical_state = self.cross_modal(critical_state, context)
            
        # 涌现预测
        emerged = self.emergence_predictor(critical_state)
        return emerged
        
class CriticalTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, 8)
        self.critical_mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        
    def forward(self, x):
        x_permuted = x.permute(1, 0, 2)
        attn_out, _ = self.self_attention(x_permuted, x_permuted, x_permuted)
        attn_out = attn_out.permute(1, 0, 2)
        return self.critical_mlp(attn_out) + x

class EmergencePredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.ReLU(),
            nn.Linear(dim*2, dim)
        )
        
    def forward(self, critical_state):
        return self.predictor(critical_state)

class GlobalEmergenceLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.critical_transformer = CriticalTransformer(dim)
        self.emergence_predictor = EmergencePredictor(dim)
        
    def forward(self, semantic_graph):
        critical_state = self.critical_transformer(semantic_graph)
        emerged_feat = self.emergence_predictor(critical_state)
        return emerged_feat

class ScaleInteractionModule(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(len(dims)))
        
    def forward(self, features):
        weights = F.softmax(self.scale_weights, dim=0)
        return sum(w * f for w, f in zip(weights, features))

class BidirectionalEmergenceCore(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.text_emergence = EmergenceCore(dim)
        self.image_emergence = EmergenceCore(dim)
        self.cross_modal = CrossModalAttention(dim)
        self.fusion = nn.Sequential(
            nn.Linear(dim*2, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        
    def forward(self, text_feat, image_feat):
        # 单模态涌现
        text_emerged = self.text_emergence(text_feat)
        image_emerged = self.image_emergence(image_feat)
        
        # 跨模态涌现
        text_cross = self.cross_modal(text_emerged, image_emerged)
        image_cross = self.cross_modal(image_emerged, text_emerged)
        
        # 特征融合
        text_final = self.fusion(torch.cat([text_emerged, text_cross], dim=-1))
        image_final = self.fusion(torch.cat([image_emerged, image_cross], dim=-1))
        
        return text_final, image_final

class MultiScaleEmergenceModule(nn.Module):
    def __init__(self, dims=[256, 512, 1024]):
        super().__init__()
        self.micro_layer = LocalFeatureAligner(dims[0])
        self.meso_layer = SemanticGraphBuilder(dims[1])
        self.macro_layer = GlobalEmergenceLayer(dims[2])
        self.bidirectional = BidirectionalEmergenceCore(dims[2])
        self.scale_interaction = ScaleInteractionModule(dims)
        self.micro_to_meso = nn.Linear(dims[0], dims[1])
        
    def forward(self, text_feat, image_feat):
       
        local_feat = self.micro_layer(text_feat, image_feat)
        
        local_feat = self.micro_to_meso(local_feat)
        
        semantic_graph = self.meso_layer(local_feat)
        
        global_emerged = self.macro_layer(semantic_graph)
        
        # 双向涌现
        text_emerged, image_emerged = self.bidirectional(text_feat, image_feat)
        
        # 多尺度特征交互
        final_text = self.scale_interaction([local_feat, semantic_graph, text_emerged])
        final_image = self.scale_interaction([local_feat, semantic_graph, image_emerged])
        
        return final_text, final_image, global_emerged

# 完整模型封装
class EmergenceModel(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.multi_scale = MultiScaleEmergenceModule([dim, dim*2, dim*4])
        
    def forward(self, text_feat, image_feat):
        return self.multi_scale(text_feat, image_feat)
