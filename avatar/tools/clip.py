import PIL
import torch
from typing import List, Union, Dict
from transformers import AutoTokenizer, AutoProcessor, CLIPModel, CLIPTextConfig
from avatar.utils.format import format_checked
from avatar.tools.tool import Tool
from avatar.models.emergence import EmergenceModel

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

class GetCLIPImageEmbedding(Tool):
    """
    Class to get CLIP image embeddings.

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
        self.processor = AutoProcessor.from_pretrained(emb_model)
        if use_cuda:
            self.model = self.model.cuda()

    @format_checked
    def __call__(self, image_lst: List[PIL.Image.Image]) -> torch.Tensor:
        """
        Generates CLIP image embeddings for the given list of images.

        Args:
            image_lst (List[PIL.Image.Image]): The list of images to embed.

        Returns:
            torch.Tensor: The generated embeddings.
        """
        print(f'get_clip_image_embedding - len(image_lst) {len(image_lst)}')
        image_embs = []
        for image_batch in [image_lst[i:i + self.batch_size] for i in range(0, len(image_lst), self.batch_size)]:
            with torch.no_grad():
                inputs = self.processor(images=image_batch, return_tensors="pt")
                inputs = {k: v.cuda() if self.use_cuda else v for k, v in inputs.items()}
                image_batch_embs = self.model.get_image_features(**inputs).cpu()
            image_embs.append(image_batch_embs.view(len(image_batch), -1))
        image_embs = torch.cat(image_embs, dim=0)

        print(f'get_clip_image_embedding - output shape {image_embs.size()}')
        return image_embs

    def __str__(self):
        return 'get_clip_image_embedding(image_lst: List[PIL.Image.Image]) -> embedding: torch.Tensor'

    def __repr__(self):
        return ("Embed a list of images into a tensor of size (len(image_lst), hidden_dim). "
                "For example, get_image_embedding([image1, image2]) returns a tensor of size (2, hidden_dim).")

class CLIPFeatureExtractorTool(Tool):
    """
    统一的CLIP特征提取工具，集成了文本和图像特征提取，并连接到EmergenceModel。
    继承自 Tool，通过 __call__ 方法调用。

    Args:
        emb_model (str): The pre-trained CLIP model to use. Default is "openai/clip-vit-large-patch14".
        batch_size (int): The batch size for processing. Default is 4.
        emergence_dim (int): The output dimension for EmergenceModel. Default is 768.
        use_cuda (bool): Whether to use CUDA for processing. Default is True.
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
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        # 初始化CLIP模型和相关组件
        self.clip_model = CLIPModel.from_pretrained(emb_model)
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model)
        self.processor = AutoProcessor.from_pretrained(emb_model)
        self.text_config = CLIPTextConfig.from_pretrained(emb_model)
        self.max_length = self.text_config.max_position_embeddings

        # 初始化EmergenceModel
        self.emergence_model = EmergenceModel(dim=emergence_dim)

        # 移动到设备
        if use_cuda:
            self.clip_model = self.clip_model.to(self.device)
            self.emergence_model = self.emergence_model.to(self.device)

    def _extract_text_features(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """提取文本特征"""
        if isinstance(texts, str):
            texts = [texts]
            
        text_embeds = []
        for text_batch in [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]:
            with torch.no_grad():
                inputs = self.tokenizer(text_batch, padding="max_length", truncation=True,
                                        max_length=self.max_length, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_batch_embs = self.clip_model.get_text_features(**inputs)
                if not self.use_cuda:
                    text_batch_embs = text_batch_embs.cpu()
            text_embeds.append(text_batch_embs)
        
        return torch.cat(text_embeds, dim=0)

    def _extract_image_features(self, images: List[PIL.Image.Image]) -> torch.Tensor:
        """提取图像特征"""
        image_embeds = []
        for image_batch in [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]:
            with torch.no_grad():
                inputs = self.processor(images=image_batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_batch_embs = self.clip_model.get_image_features(**inputs)
                if not self.use_cuda:
                    image_batch_embs = image_batch_embs.cpu()
            image_embeds.append(image_batch_embs)
        
        return torch.cat(image_embeds, dim=0)

    @format_checked
    def __call__(self, texts: Union[str, List[str]] = None, images: List[PIL.Image.Image] = None) -> Dict[str, torch.Tensor]:
        """
        提取文本和图像特征，并通过EmergenceModel处理。

        Args:
            texts (Union[str, List[str]]): The input string or list of strings to embed.
            images (List[PIL.Image.Image]): The input list of PIL images to embed.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing raw and processed features.
        """
        result = {}

        # 提取文本特征
        if texts is not None:
            text_features = self._extract_text_features(texts)
            # 确保特征维度匹配（增加序列长度维度）
            if text_features.dim() == 2:
                text_features = text_features.unsqueeze(1)  # [B, 1, D]
            result['text_features'] = text_features

        # 提取图像特征
        if images is not None:
            image_features = self._extract_image_features(images)
            # 确保特征维度匹配（增加序列长度维度）
            if image_features.dim() == 2:
                image_features = image_features.unsqueeze(1)  # [B, 1, D]
            result['image_features'] = image_features

        # 通过 EmergenceModel 处理特征
        if texts is not None and images is not None:
            # 多模态：联合涌现
            final_text, final_image, global_emerged = self.emergence_model(
                text_features.to(self.device),
                image_features.to(self.device)
            )
            result.update({
                'final_text': final_text,
                'final_image': final_image,
                'global_emerged': global_emerged
            })
        elif texts is not None:
            # 单模态：只处理文本
            text_emerged = self.emergence_model.forward_text(text_features.to(self.device))
            result['final_text'] = text_emerged
        elif images is not None:
            # 单模态：只处理图像
            image_emerged = self.emergence_model.forward_image(image_features.to(self.device))
            result['final_image'] = image_emerged

        return result

    def __str__(self):
        return 'clip_feature_extractor_tool(texts: Union[str, List[str]], images: List[PIL.Image.Image]) -> Dict[str, torch.Tensor]'

    def __repr__(self):
        return ("Extract CLIP features for texts and/or images, and process them with EmergenceModel. "
                "Returns a dictionary with raw and processed features.")