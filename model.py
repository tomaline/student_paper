"""
模型定义
=================
论文中使用的所有神经网络模块：
- BERTClassifier: 基于 BERT 的文本分类器（用于 Stage-1 和 Stage-2 微调）
- BERTFeatureExtractor: 冻结参数的 BERT，用于抽取语义特征
- BipartiteGCNEncoder: 用于实例级二分图的 GCN 编码器
- CrossRelationAttention: 基于查询的双线性交叉关系注意力机制
- DualGraphGNN: 完整的 GNN 模型（字图 + 词图 + 融合 + 分类器）
"""

import os
import logging
from typing import List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import jieba

from config import Config


# ============================================================================
# BERT 相关模块
# ============================================================================

class BERTClassifier(nn.Module):
    """
    基于 BERT 的文本分类器，用于端到端微调。

    参数:
        model_name (str): 预训练 BERT 模型名称，默认为 "bert-base-chinese"。
        num_classes (int): 分类类别数，默认为 10。
        dropout (float): Dropout 概率，默认为 0.1。
    """

    def __init__(self, model_name: str = "bert-base-chinese", num_classes: int = 10, dropout: float = 0.1):
        super(BERTClassifier, self).__init__()
        # 加载预训练 BERT 模型
        self.bert = BertModel.from_pretrained(model_name)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 分类全连接层，将 BERT 输出映射到类别数
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        # 类别权重，用于类别不平衡损失加权；None 表示不使用加权
        self.class_weights = None

    def set_class_weights(self, weights: torch.Tensor):
        """设置类别权重张量，形状应为 (num_classes,)。"""
        self.class_weights = weights

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算交叉熵损失，若设置了类别权重则使用加权交叉熵。"""
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
        return loss_fct(logits, labels)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播。

        参数:
            input_ids: 输入文本的 token ID 张量。
            attention_mask: 注意力掩码张量。
            labels: 真实标签张量，若为 None 则只返回 logits。

        返回:
            若 labels 不为 None，返回 (loss, logits)；否则返回 logits。
        """
        # 通过 BERT 获取输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 提取 [CLS] 对应的池化输出 (B, hidden_size)
        pooled_output = outputs.pooler_output
        # 对池化输出应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 分类层预测 logits (B, num_classes)
        logits = self.classifier(pooled_output)
        # 若提供了标签，则根据配置计算损失
        if labels is not None:
            loss = self._compute_loss(logits, labels)
            return loss, logits
        return logits

    def get_mean_pooled_embeddings(self, input_ids, attention_mask):
        """
        从最后一层隐藏状态提取均值池化嵌入。
        公式: h = (1/|x|) * sum_t h_t  （带掩码的平均）

        参数:
            input_ids: 输入文本的 token ID 张量。
            attention_mask: 注意力掩码张量。

        返回:
            均值池化后的嵌入张量，形状为 (B, D)。
        """
        # 获取 BERT 最后一层隐藏状态 (B, T, D)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, T, D)
        # 将 attention_mask 扩展维度以匹配隐藏状态维度
        mask_expanded = attention_mask.unsqueeze(-1).float()
        # 对有效 token 的隐藏状态求和
        summed = (last_hidden * mask_expanded).sum(dim=1)
        # 计算有效 token 数量，防止除零
        counts = mask_expanded.sum(dim=1).clamp(min=1)
        # 返回均值池化结果 (B, D)
        return summed / counts  # (B, D)

    def get_cls_embeddings(self, input_ids, attention_mask):
        """
        提取 [CLS] token 的嵌入向量。

        参数:
            input_ids: 输入文本的 token ID 张量。
            attention_mask: 注意力掩码张量。

        返回:
            [CLS] token 对应的嵌入张量，形状为 (B, D)。
        """
        # 获取 BERT 最后一层隐藏状态
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取每个样本第一个位置（[CLS]）的向量
        return outputs.last_hidden_state[:, 0, :]


def adapt_tokenizer_for_domain(
    tokenizer: BertTokenizer,
    model: BertModel,
    texts: List[str],
    max_new_tokens: int = 1000,
    min_freq: int = 3,
    logger: Optional[logging.Logger] = None,
) -> Tuple[BertTokenizer, BertModel, List[str]]:
    """
    在领域语料上对 BERT Tokenizer 进行自适应扩展。

    流程:
        1. 使用 jieba 对训练语料分词，统计词频；
        2. 筛选高频且当前词表中未覆盖的领域词；
        3. 将 Top-K 领域词加入 tokenizer，并同步 resize BERT 嵌入层。

    参数:
        tokenizer: 原始 BERT 分词器
        model: 原始 BERT 模型（将被修改以扩展嵌入层）
        texts: 领域语料文本列表
        max_new_tokens: 最多新增 token 数量
        min_freq: 领域词最小出现频次
        logger: 日志记录器

    返回:
        (扩展后的 tokenizer, 扩展后的 model, 新增 token 列表)
    """
    logger = logger or logging.getLogger(__name__)

    # 统计词频
    word_counter: Counter = Counter()
    for text in tqdm(texts, desc="统计领域词频"):
        if not text:
            continue
        words = list(jieba.cut(text))
        # 过滤掉纯标点、纯数字、空白等无意义片段
        filtered = [
            w.strip() for w in words
            if w.strip() and len(w.strip()) > 1 and not w.strip().isspace()
        ]
        word_counter.update(filtered)

    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = []
    for word, count in word_counter.most_common():
        if len(new_tokens) >= max_new_tokens:
            break
        if count < min_freq:
            break
        if word not in existing_vocab:
            new_tokens.append(word)

    if new_tokens:
        num_added = tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(
            "Tokenizer 领域自适应：新增 %d 个领域 token（候选 %d 个），当前词表大小 %d",
            num_added, len(new_tokens), len(tokenizer),
        )
    else:
        logger.info("Tokenizer 领域自适应：未检测到需要新增的领域 token")

    return tokenizer, model, new_tokens


class BERTFeatureExtractor(nn.Module):
    """
    冻结参数的 BERT，用于语义特征抽取。
    加载 Stage-2 微调后的最佳权重并冻结所有参数。

    参数:
        model_name (str): 预训练 BERT 模型名称或本地路径。
        device (str): 运行设备，默认为 "cpu"。
        tokenizer_path (Optional[str]): 若提供，从该路径加载已适应领域的分词器；
            否则从 model_name 加载。
    """

    def __init__(self, model_name: str, device: str = "cpu", tokenizer_path: Optional[str] = None):
        super().__init__()
        # 加载预训练 BERT 并转移到指定设备
        self.bert = BertModel.from_pretrained(model_name).to(device)
        # 加载对应的分词器（优先使用领域自适应后的 tokenizer）
        if tokenizer_path is not None and os.path.isdir(tokenizer_path):
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # 记录设备信息
        self.device = device
        # 最大序列长度
        self.max_length = 512

    def load_finetuned_weights(self, checkpoint_path: str):
        """
        加载 Stage-2 最佳 BERT 权重。

        参数:
            checkpoint_path (str): 检查点文件路径。
        """
        # 检查检查点文件是否存在
        if os.path.exists(checkpoint_path):
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            # 检查点可能来自 BERTClassifier，因此只提取 bert 部分
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            # 过滤出以 "bert." 开头的键，并移除前缀
            bert_state = {k.replace("bert.", "", 1): v for k, v in state_dict.items() if k.startswith("bert.")}
            # 加载到当前 bert 模型，允许部分加载
            self.bert.load_state_dict(bert_state, strict=False)
            logging.getLogger(__name__).info("已从 %s 加载微调后的 BERT", checkpoint_path)
        else:
            logging.getLogger(__name__).warning("检查点 %s 不存在，使用预训练 BERT 权重。", checkpoint_path)

    def freeze(self):
        """
        冻结所有 BERT 参数，使其在训练中不更新。
        """
        # 遍历所有参数并关闭梯度计算
        for param in self.bert.parameters():
            param.requires_grad = False
        # 设置为评估模式，关闭 Dropout 等训练行为
        self.bert.eval()

    @torch.no_grad()
    def extract_mean_pooled(self, texts: List[str], batch_size: int = 64, max_length: int = 512) -> np.ndarray:
        """
        对一批文本抽取均值池化的 BERT 特征。

        参数:
            texts (List[str]): 输入文本列表。
            batch_size (int): 批处理大小，默认为 64。
            max_length (int): 最大序列长度，默认为 512。

        返回:
            均值池化后的特征矩阵，形状为 (num_texts, hidden_size)。
        """
        # 设置为评估模式
        self.bert.eval()
        all_embeddings = []
        # 按 batch_size 分批处理文本
        for i in tqdm(range(0, len(texts), batch_size), desc="抽取 BERT 特征"):
            batch_texts = texts[i:i + batch_size]
            # 使用分词器对批次文本进行编码
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            # 将输入张量转移到设备
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # 前向传播获取 BERT 输出
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (B, T, D)
            # 均值池化计算
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            embeddings = (summed / counts).cpu().numpy()
            all_embeddings.append(embeddings)

        # 垂直拼接所有批次的嵌入结果
        return np.vstack(all_embeddings)


# ============================================================================
# GNN 相关模块
# ============================================================================

class BipartiteGCNEncoder(nn.Module):
    """
    用于二分图的 GCN 编码器。
    处理（可能为分块对角形式的）批量图中的所有节点。

    参数:
        in_channels (int): 输入特征维度。
        hidden_channels (int): 隐藏层特征维度。
        num_layers (int): GCN 层数，默认为 2。
        dropout (float): Dropout 概率，默认为 0.5。
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        # 记录层数
        self.num_layers = num_layers
        # 记录 dropout 概率
        self.dropout = dropout

        # 构建 GCN 卷积层列表
        self.convs = nn.ModuleList()
        # 构建批归一化层列表
        self.bns = nn.ModuleList()

        # 第一层：输入维度 -> 隐藏维度
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 后续层：隐藏维度 -> 隐藏维度
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (torch.Tensor): 节点特征张量，形状为 (num_nodes, in_channels)。
            edge_index (torch.Tensor): 边索引张量，形状为 (2, num_edges)。
            batch (Optional[torch.Tensor]): 批次分配张量，用于批归一化。

        返回:
            经过 GCN 编码后的节点特征张量，形状为 (num_nodes, hidden_channels)。
        """
        # 逐层进行图卷积、批归一化、ReLU 激活和 Dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)          # 图卷积
            x = self.bns[i](x)               # 批归一化
            x = F.relu(x)                    # ReLU 激活
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        return x


class CrossRelationAttention(nn.Module):
    """
    基于查询的双线性交叉关系注意力机制。

    输入:
        h: (B, D) 原始 BERT 特征（作为 Query 来源）
        z_c: (B, D) 来自字图 GCN 的文档特征（作为 Key/Value）
        z_w: (B, D) 来自词图 GCN 的文档特征（作为 Key/Value）

    输出:
        z_out: (B, D) 融合后的文档特征

    参数:
        dim (int): 特征维度。
    """

    def __init__(self, dim: int):
        super().__init__()
        # 记录特征维度
        self.dim = dim
        # Query 的线性变换
        self.W_q = nn.Linear(dim, dim)
        # Key 的线性变换（字图和词图共享）
        self.W_k = nn.Linear(dim, dim)
        # Value 的线性变换（字图和词图共享）
        self.W_v = nn.Linear(dim, dim)
        # 可学习的双线性交互矩阵
        self.B = nn.Parameter(torch.randn(dim, dim) * 0.02)
        # 层归一化，用于残差连接后的稳定
        self.layer_norm = nn.LayerNorm(dim)

        # 用于追踪 alpha（字图/词图原始注意力分数）和 beta（归一化注意力权重）
        # 在训练过程中的变化
        self.track_beta = False
        self.register_buffer("alpha_sum", torch.zeros(2))
        self.register_buffer("alpha_count", torch.zeros(1))
        self.register_buffer("beta_sum", torch.zeros(2))
        self.register_buffer("beta_count", torch.zeros(1))

    def forward(self, h: torch.Tensor, z_c: torch.Tensor, z_w: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算交叉关系注意力并融合字图和词图特征。

        参数:
            h (torch.Tensor): 原始 BERT 特征，形状为 (B, D)。
            z_c (torch.Tensor): 字图文档特征，形状为 (B, D)。
            z_w (torch.Tensor): 词图文档特征，形状为 (B, D)。

        返回:
            融合后的文档特征张量，形状为 (B, D)。
        """
        # 分别计算 Query、Key、Value
        q = self.W_q(h)
        k_c = self.W_k(z_c)
        k_w = self.W_k(z_w)
        v_c = self.W_v(z_c)
        v_w = self.W_v(z_w)

        # 通过双线性矩阵对 Key 进行变换
        Bk_c = torch.matmul(k_c, self.B.t())
        Bk_w = torch.matmul(k_w, self.B.t())

        # 计算字图和词图的注意力分数（点积注意力）
        alpha_c = (q * Bk_c).sum(dim=-1, keepdim=True)
        alpha_w = (q * Bk_w).sum(dim=-1, keepdim=True)

        # 拼接注意力分数并应用 Softmax 归一化
        alphas = torch.cat([alpha_c, alpha_w], dim=-1)
        betas = F.softmax(alphas, dim=-1)
        beta_c = betas[:, 0:1]
        beta_w = betas[:, 1:2]

        # 若启用追踪，累加当前 batch 的平均 alpha 和 beta 值
        if self.track_beta:
            with torch.no_grad():
                self.alpha_sum += alphas.mean(dim=0).detach()
                self.alpha_count += 1
                self.beta_sum += betas.mean(dim=0).detach()
                self.beta_count += 1

        # 加权融合字图和词图的 Value
        z_fuse = beta_c * v_c + beta_w * v_w
        # 残差连接后进行层归一化
        z_out = self.layer_norm(z_fuse + h)
        return z_out

    def set_beta_tracking(self, enabled: bool = True):
        """启用或禁用 alpha/beta 追踪。"""
        self.track_beta = enabled

    def reset_attention_stats(self):
        """清空已累加的 alpha/beta 统计信息。"""
        self.alpha_sum.zero_()
        self.alpha_count.zero_()
        self.beta_sum.zero_()
        self.beta_count.zero_()

    # 保留旧名称以兼容历史代码
    reset_beta_stats = reset_attention_stats

    def get_avg_alpha(self) -> Tuple[float, float]:
        """
        获取当前已累加 alpha 的平均值。

        返回:
            Tuple[float, float]: (字图平均 alpha_c, 词图平均 alpha_w)
        """
        count = self.alpha_count.item()
        if count == 0:
            return 0.0, 0.0
        avg = self.alpha_sum / count
        return float(avg[0].item()), float(avg[1].item())

    def get_avg_beta(self) -> Tuple[float, float]:
        """
        获取当前已累加 beta 的平均值。

        返回:
            Tuple[float, float]: (字图平均权重 beta_c, 词图平均权重 beta_w)
        """
        count = self.beta_count.item()
        if count == 0:
            return 0.0, 0.0
        avg = self.beta_sum / count
        return float(avg[0].item()), float(avg[1].item())


class DualGraphGNN(nn.Module):
    """
    完整的实例级图神经网络模型：
    - 接受 PyG Batch 对象（由实例图拼接成的分块对角图）
    - 双二分图 GNN 编码器分别在字图和词图上传播
    - 通过 doc_mask 提取文档节点
    - 交叉关系注意力融合字图与词图的文档特征

    参数:
        config (Config): 全局配置对象。
        bert_dim (int): BERT 特征维度，默认为 768。
        num_classes (int): 分类类别数，默认为 10。
    """

    def __init__(self, config: Config, bert_dim: int = 768, num_classes: int = 10, fusion_type: str = "cross_attention"):
        super().__init__()
        # 保存配置和维度信息
        self.config = config
        self.bert_dim = bert_dim
        self.num_classes = num_classes
        # 融合方式: cross_attention / concat / add
        self.fusion_type = fusion_type

        # 字图 GCN 编码器
        self.char_encoder = BipartiteGCNEncoder(
            in_channels=bert_dim,
            hidden_channels=config.gnn.hidden_channels,
            num_layers=config.gnn.num_layers,
            dropout=config.gnn.dropout,
        )
        # 词图 GCN 编码器
        self.word_encoder = BipartiteGCNEncoder(
            in_channels=bert_dim,
            hidden_channels=config.gnn.hidden_channels,
            num_layers=config.gnn.num_layers,
            dropout=config.gnn.dropout,
        )

        # 交叉关系注意力融合模块（仅 fusion_type="cross_attention" 时使用）
        if fusion_type == "cross_attention":
            self.cross_attention = CrossRelationAttention(dim=config.gnn.hidden_channels)
        else:
            self.cross_attention = None

        # 将 BERT 特征投影到 GNN 隐藏维度
        self.h_proj = nn.Linear(bert_dim, config.gnn.hidden_channels)

        # 最终分类器输入维度
        if fusion_type == "concat":
            classifier_in_dim = 2 * config.gnn.hidden_channels
        else:
            classifier_in_dim = config.gnn.hidden_channels
        self.classifier = nn.Linear(classifier_in_dim, num_classes)

    def _extract_doc_features(self, batch: Batch, encoded: torch.Tensor) -> torch.Tensor:
        """
        从批量图中提取文档节点特征。

        参数:
            batch (Batch): PyG 的 Batch 对象，包含 doc_mask。
            encoded (torch.Tensor): 编码后的所有节点特征。

        返回:
            文档节点特征张量。
        """
        # 使用 doc_mask 布尔索引筛选出文档节点
        return encoded[batch.doc_mask]

    def forward(self, char_batch: Batch, word_batch: Batch, bert_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播：分别编码字图和词图，融合后分类。

        参数:
            char_batch (Batch): 字图批量数据。
            word_batch (Batch): 词图批量数据。
            bert_embeddings (torch.Tensor): 原始 BERT 嵌入，形状为 (B, bert_dim)。

        返回:
            分类 logits，形状为 (B, num_classes)。
        """
        # 字图 GCN 编码：获取所有节点的特征
        z_all_char = self.char_encoder(char_batch.x, char_batch.edge_index)
        # 词图 GCN 编码：获取所有节点的特征
        z_all_word = self.word_encoder(word_batch.x, word_batch.edge_index)

        # 从字图中提取文档节点特征 (B, hidden_channels)
        z_doc_char = self._extract_doc_features(char_batch, z_all_char)
        # 从词图中提取文档节点特征 (B, hidden_channels)
        z_doc_word = self._extract_doc_features(word_batch, z_all_word)

        # 将原始 BERT 嵌入投影到 GNN 隐藏维度，作为 Query
        h_proj = self.h_proj(bert_embeddings)

        # 根据 fusion_type 融合字图和词图特征
        if self.fusion_type == "cross_attention":
            z_out = self.cross_attention(h_proj, z_doc_char, z_doc_word)
        elif self.fusion_type == "concat":
            z_out = torch.cat([z_doc_char, z_doc_word], dim=-1)
        elif self.fusion_type == "add":
            z_out = z_doc_char + z_doc_word
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        # 最终分类层输出 logits
        logits = self.classifier(z_out)
        return logits
