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
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
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
    def __init__(self, dim, noise_scale=0.1):
        super().__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(dim*2, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim*2),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, 1)
        )
        self.temperature = nn.Parameter(torch.ones(dim) * 0.1)  # 可学习温度参数
        self.noise_scale = noise_scale  # 控制量子噪声幅度
        
    def forward(self, text_feat, image_feat):
        joint_feat = torch.cat([text_feat, image_feat], dim=-1)
        energy = self.energy_net(joint_feat) / self.temperature.view(1, 1, -1)
        phase = torch.sigmoid(energy)  # 阶段状态，0-1之间
        
        # 量子波动：低能量时注入受控噪声（仅在训练时）
        if self.training:
            noise_mask = (phase < 0.5).float()  # 能量低于阈值时触发
            noise = torch.randn_like(text_feat) * self.noise_scale * noise_mask.expand_as(text_feat)
            text_feat = text_feat + noise
            image_feat = image_feat + noise  # 双向噪声注入
        
        return phase, text_feat, image_feat  # 返回阶段状态和扰动后的特征

class LocalFeatureAligner(nn.Module):
    def __init__(self, dim, noise_scale=0.1):
        super().__init__()
        self.phase_transition = PhaseTransitionLayer(dim, noise_scale)
        self.cross_attention = CrossModalAttention(dim)
        
    def forward(self, text_feat, image_feat):
        phase, text_feat_perturbed, image_feat_perturbed = self.phase_transition(text_feat, image_feat)
        aligned_feat = self.cross_attention(text_feat_perturbed, image_feat_perturbed)
        return aligned_feat * phase.expand_as(aligned_feat)  # 能量调制

class DynamicTopoNet(nn.Module):
    def __init__(self, dim, noise_scale=0.1):
        super().__init__()
        self.graph_gen = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        self.attention = nn.MultiheadAttention(dim, num_heads=8)
        self.noise_scale = noise_scale  # 量子噪声幅度
        
    def forward(self, x):
        # 自注意力生成初始特征
        x_permuted = x.permute(1, 0, 2)
        attn_out, _ = self.attention(x_permuted, x_permuted, x_permuted)
        attn_out = attn_out.permute(1, 0, 2)
        
        # 量子波动：注入噪声增强拓扑适应性（仅训练时）
        if self.training:
            noise = torch.randn_like(attn_out) * self.noise_scale
            attn_out = attn_out + noise
        
        gen_out = self.graph_gen(attn_out)
        adj_matrix = torch.matmul(gen_out, gen_out.transpose(-2, -1))  # 双向图
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
        entropy = -torch.sum(graph * torch.log(torch.clamp(graph, min=1e-15, max=1.0)), dim=-1, keepdim=True)
        gate = torch.sigmoid(self.mlp(entropy))
        return graph * gate

class SemanticGraphBuilder(nn.Module):
    def __init__(self, dim, noise_scale=0.1):
        super().__init__()
        self.dynamic_topology = DynamicTopoNet(dim, noise_scale)
        self.entropy_gate = EntropyGateLayer(dim)
        
    def forward(self, local_feat):
        initial_graph = self.dynamic_topology(local_feat)
        evolved_graph = self.entropy_gate(initial_graph)
        return evolved_graph

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

class EmergenceCore(nn.Module):
    def __init__(self, dim, noise_scale=0.1):
        super().__init__()
        self.phase_transition = PhaseTransitionLayer(dim, noise_scale)
        self.critical_transformer = CriticalTransformer(dim)
        self.emergence_predictor = EmergencePredictor(dim)
        self.cross_modal = CrossModalAttention(dim)
        
    def forward(self, x, context=None):
        if context is not None:
            phase_state, x_perturbed, context_perturbed = self.phase_transition(x, context)
            x = x_perturbed * phase_state.expand_as(x)
            context = context_perturbed
        critical_state = self.critical_transformer(x)
        if context is not None:
            critical_state = self.cross_modal(critical_state, context)
        emerged = self.emergence_predictor(critical_state)
        return emerged

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
        self.mlp = nn.Sequential(
            nn.Linear(sum(dims), sum(dims) // 2),
            nn.GELU(),
            nn.Linear(sum(dims) // 2, dims[-1])
        )
        
    def forward(self, features):
        concat_feats = torch.cat(features, dim=-1)
        return self.mlp(concat_feats)

class BidirectionalEmergenceCore(nn.Module):
    def __init__(self, dim, noise_scale=0.1):
        super().__init__()
        self.text_emergence = EmergenceCore(dim, noise_scale)
        self.image_emergence = EmergenceCore(dim, noise_scale)
        self.cross_modal = CrossModalAttention(dim)
        self.fusion = nn.Sequential(
            nn.Linear(dim*2, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        
    def forward(self, text_feat, image_feat):
        text_emerged = self.text_emergence(text_feat, image_feat)
        image_emerged = self.image_emergence(image_feat, text_feat)
        text_context = self.cross_modal(text_emerged, image_emerged)
        image_context = self.cross_modal(image_emerged, text_emerged)
        text_final = self.fusion(torch.cat([text_emerged, text_context], dim=-1))
        image_final = self.fusion(torch.cat([image_emerged, image_context], dim=-1))
        return text_final, image_final

class MultiScaleEmergenceModule(nn.Module):
    def __init__(self, dims, noise_scale=0.1):
        super().__init__()
        self.micro_layer = LocalFeatureAligner(dims[0], noise_scale)
        self.meso_layer = SemanticGraphBuilder(dims[1], noise_scale)
        self.macro_layer = GlobalEmergenceLayer(dims[2])
        self.bidirectional = BidirectionalEmergenceCore(dims[2], noise_scale)
        self.scale_interaction = ScaleInteractionModule(dims)
        self.micro_to_meso = nn.Linear(dims[0], dims[1])
        
    def forward(self, text_feat, image_feat):
        local_feat = self.micro_layer(text_feat, image_feat)
        local_feat = self.micro_to_meso(local_feat)
        semantic_graph = self.meso_layer(local_feat)
        global_emerged = self.macro_layer(semantic_graph)
        text_emerged, image_emerged = self.bidirectional(text_feat, image_feat)
        final_text = self.scale_interaction([local_feat, semantic_graph, text_emerged])
        final_image = self.scale_interaction([local_feat, semantic_graph, image_emerged])
        return final_text, final_image, global_emerged

class EmergenceModel(nn.Module):
    def __init__(self, dim=512, text_input_dim=768, image_input_dim=1024, num_classes=None, modality_dims=None, noise_scale=0.1):
        super().__init__()
        if modality_dims is None:
            modality_dims = {'text': text_input_dim, 'image': image_input_dim}
        self.projections = nn.ModuleDict({k: nn.Linear(v, dim) for k, v in modality_dims.items()})
        self.multi_scale = MultiScaleEmergenceModule([dim, dim*2, dim*4], noise_scale)
        
        # 可选的任务头（例如分类）
        self.num_classes = num_classes
        if num_classes:
            self.classifier = nn.Linear(dim*4, num_classes)
        
    def forward(self, text_feat=None, image_feat=None, **modalities):
        if text_feat is not None and image_feat is not None:
            feats = {
                'text': self.projections['text'](text_feat),
                'image': self.projections['image'](image_feat)
            }
        else:
            feats = {k: self.projections[k](v) for k, v in modalities.items()}
        
        final_text, final_image, global_emerged = self.multi_scale(feats['text'], feats['image'])
        
        if self.num_classes:
            cls_input = global_emerged.mean(dim=1)  # 池化
            logits = self.classifier(cls_input)
            return final_text, final_image, global_emerged, logits
        return final_text, final_image, global_emerged
    
    def consistency_loss(self, final_text, final_image):
        """辅助损失：鼓励跨模态一致性"""
        return -F.cosine_similarity(final_text.mean(dim=1), final_image.mean(dim=1)).mean()
