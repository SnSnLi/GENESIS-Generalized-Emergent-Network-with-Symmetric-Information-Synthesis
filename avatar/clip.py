import PIL
import torch
from typing import List, Union
from transformers import AutoTokenizer, AutoProcessor, CLIPModel, CLIPTextConfig
from avatar.models.emergence import EmergenceModel
from avatar.utils.format import format_checked
from avatar.tools.tool import Tool


class GetCLIPTextEmbedding(Tool):
    """
    Class to get CLIP text embeddings.

    Args:
        emb_model (str): The pre-trained CLIP model to use. Default is "openai/clip-vit-large-patch14".
        batch_size (int): The batch size for processing. Default is 4.
        use_cuda (bool): Whether to use CUDA for processing. Default is True
        **kwargs: Additional arguments.
    """

    def __init__(self, 
                 emb_model: str = "openai/clip-vit-large-patch14", 
                 batch_size: int = 4, 
                 use_cuda: bool = True,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.model = CLIPModel.from_pretrained(emb_model)
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model)
        text_config = CLIPTextConfig.from_pretrained(emb_model)
        self.max_length = text_config.max_position_embeddings
        if use_cuda:
            self.model = self.model.cuda()

    @format_checked
    def __call__(self, string: Union[str, List[str]]) -> torch.Tensor:
        """
        Generates CLIP text embeddings for the given string or list of strings.

        Args:
            string (Union[str, List[str]]): The input string or list of strings to embed.

        Returns:
            torch.Tensor: The generated embeddings.
        """
        if isinstance(string, str):
            string = [string]
        assert all(len(s) > 0 for s in string), 'Every string in the list to be embedded should be non-empty'

        print(f'get_clip_text_embedding - input {string}')
        text_embeds = []
        for text_batch in [string[i:i + self.batch_size] for i in range(0, len(string), self.batch_size)]:
            with torch.no_grad():
                inputs = self.tokenizer(text_batch, padding="max_length", truncation=True, 
                                        max_length=self.max_length, return_tensors="pt")
                inputs = {k: v.cuda() if self.use_cuda else v for k, v in inputs.items()}
                text_batch_embs = self.model.get_text_features(**inputs).cpu()
            text_embeds.append(text_batch_embs.view(len(text_batch), -1))
        text_embeds = torch.cat(text_embeds, dim=0)

        print(f'get_clip_text_embedding - output shape {text_embeds.size()}')
        return text_embeds

    def __str__(self):
        return 'get_clip_text_embedding(string: Union[str, List[str]]) -> embedding: torch.Tensor'

    def __repr__(self):
        return ("Embed a string or list of N strings into a tensor of size (N, hidden_dim). For efficiency, "
                "include multiple strings in the list at once, rather than calling the function separately "
                "for each string.")

class CLIPFeatureExtractor(nn.Module):
    """
    统一的CLIP特征提取器，集成了文本和图像特征提取，并连接到EmergenceModel
    """
    def __init__(
        self,
        emb_model: str = "openai/clip-vit-large-patch14",
        batch_size: int = 4,
        emergence_dim: int = 768,  # CLIP-large的输出维度
        use_cuda: bool = True
    ):
        super().__init__()
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        
        # 初始化CLIP模型和相关组件
        self.clip_model = CLIPModel.from_pretrained(emb_model)
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model)
        self.processor = AutoProcessor.from_pretrained(emb_model)
        self.text_config = CLIPTextConfig.from_pretrained(emb_model)
        self.max_length = self.text_config.max_position_embeddings
        
        # 初始化EmergenceModel
        self.emergence_model = EmergenceModel(dim=emergence_dim)
        
        if use_cuda:
            self.clip_model = self.clip_model.cuda()
            self.emergence_model = self.emergence_model.cuda()

    def extract_text_features(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """提取文本特征"""
        if isinstance(texts, str):
            texts = [texts]
            
        text_embeds = []
        for text_batch in [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]:
            with torch.no_grad():
                inputs = self.tokenizer(text_batch, padding="max_length", truncation=True,
                                      max_length=self.max_length, return_tensors="pt")
                if self.use_cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                text_batch_embs = self.clip_model.get_text_features(**inputs)
                if not self.use_cuda:
                    text_batch_embs = text_batch_embs.cpu()
            text_embeds.append(text_batch_embs)
        
        return torch.cat(text_embeds, dim=0)

    def extract_image_features(self, images: List[PIL.Image.Image]) -> torch.Tensor:
        """提取图像特征"""
        image_embeds = []
        for image_batch in [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]:
            with torch.no_grad():
                inputs = self.processor(images=image_batch, return_tensors="pt")
                if self.use_cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                image_batch_embs = self.clip_model.get_image_features(**inputs)
                if not self.use_cuda:
                    image_batch_embs = image_batch_embs.cpu()
            image_embeds.append(image_batch_embs)
        
        return torch.cat(image_embeds, dim=0)

    def forward(self, texts: Union[str, List[str]], images: List[PIL.Image.Image]):
        """
        完整的前向传播过程：
        1. 提取文本和图像特征
        2. 将特征传入EmergenceModel
        """
        # 1. 提取特征
        text_features = self.extract_text_features(texts)
        image_features = self.extract_image_features(images)
        
        # 2. 确保特征维度匹配（增加序列长度维度）
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)  # [B, 1, D]
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)  # [B, 1, D]
            
        # 3. 通过EmergenceModel处理特征
        final_text, final_image, global_emerged = self.emergence_model(text_features, image_features)
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'final_text': final_text,
            'final_image': final_image,
            'global_emerged': global_emerged
        }
