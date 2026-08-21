"""
主训练流水线

论文: 基于自适应语义-字符多样性联合筛选与跨关系融合的不平衡中文分类

使用方法:
    1. 编辑 settings.py 配置参数
    2. 运行以下任意一种:
       # 方式 A: 文件夹模式
       python main.py --data_dir /path/to/your/dataset/

       # 方式 B: 单文件模式, 自动 7:3 划分 train/test，验证集与测试集通用
       python main.py --data_path /path/to/data.txt

       # 方式 C: 训练集 + 原始完整数据集模式
       python main.py --data_path /path/to/train.txt --original_data_path /path/to/full_data.txt

数据集文件夹结构 (--data_dir):
    your_dataset/
    ├── train.txt       (必需, 格式: label\ttext)
    ├── val.txt         (可选; 若 test.txt 缺失可作为测试集)
    └── test.txt        (可选; 验证集与测试集通用)

单个文件 (--data_path):
    data.txt            (格式: label\ttext)
    将自动划分为: 训练集 70% / 测试集(验证集) 30%，验证集与测试集通用
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch

from config import build_config, build_dataset_id
from utils import set_seed, setup_logger, compute_metrics, print_class_distribution, get_class_distribution
from data_loader import (
    load_dataset_folder, load_data, save_data, split_data, split_data_three_way,
    infer_num_classes, extract_val_test_from_full_data,
)
from data_augmentation import AdaptiveDataAugmenter
from model import BERTClassifier, BERTFeatureExtractor, DualGraphGNN
from bert_stage1 import run_stage1
from bert_stage2 import run_stage2
from graph_builder import InstanceGraphBuilder
from train import train_gnn
from test import run_test

import settings
import os
import tempfile
# 将 jieba 缓存放到用户目录下，避免 /tmp 权限问题
jieba_cache_dir = os.path.expanduser("~/.jieba_cache")
os.makedirs(jieba_cache_dir, exist_ok=True)
tempfile.tempdir = jieba_cache_dir
def parse_args():
    """
    解析命令行参数，要求必须提供 --data_dir 或 --data_path 之一。

    返回:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="Chinese Text Classification with Adaptive Augmentation and Dual-Graph GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data_dir ./data/my_dataset
  python main.py --data_path ./data/data.txt
  python main.py --data_path ./data/data.txt --test_path ./data/test.txt
  python main.py --data_path ./data/train_1percent.txt --original_data_path ./data/full_data.txt
        """,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="数据集文件夹路径，需包含 train.txt（可选 val.txt, test.txt）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="单个数据文件路径（格式: label\\ttext）。单独使用时自动按 7:3 划分 train/test，验证集与测试集通用；与 --original_data_path 联用时作为训练集",
    )
    parser.add_argument(
        "--original_data_path",
        type=str,
        default=None,
        help="原始完整数据集路径（格式: label\\ttext）。提供时从该数据集中按 30% 抽取验证集/测试集（与训练集互斥），验证集与测试集通用",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="独立测试集文件路径（可选），若提供则训练后会在测试集上评估",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="仅测试模式：跳过训练，直接加载已保存的模型对 --test_path 进行推理",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help="指定数据集标识符，用于在 logs/checkpoints/visualization 中定位对应目录。"
             "默认根据 --data_dir/--data_path/--test_path 自动推断。",
    )
    args = parser.parse_args()

    # 参数校验
    if args.test_only:
        # 仅测试模式：必须提供 --test_path，不需要训练数据
        if args.test_path is None:
            parser.error("--test_only 模式下必须提供 --test_path 指定测试文件。")
    else:
        # 训练模式：必须提供且只能提供一种数据源
        if args.data_dir is None and args.data_path is None:
            parser.error("必须提供 --data_dir 或 --data_path 之一。")
        if args.data_dir is not None and args.data_path is not None:
            parser.error("--data_dir 和 --data_path 只能提供其一，不能同时提供。")
        if args.original_data_path is not None and args.data_dir is not None:
            parser.error("--original_data_path 不能与 --data_dir 同时使用。")
        if args.original_data_path is not None and args.test_path is not None:
            parser.error("--original_data_path 不能与 --test_path 同时使用（会自动从原始数据集抽取测试集）。")
        if args.original_data_path is not None and args.data_path is None:
            parser.error("使用 --original_data_path 时必须同时提供 --data_path 作为训练集。")
    return args


def run_pipeline(data_dir: str = None, data_path: str = None, original_data_path: str = None, test_path: str = None):
    """
    运行完整的训练流水线，包括数据加载、BERT 微调、数据增强、图构建和 GNN 训练。

    参数:
        data_dir (str, optional): 数据集文件夹路径
        data_path (str, optional): 单个数据文件路径（作为完整数据集自动划分，或作为训练集与 --original_data_path 联用）
        original_data_path (str, optional): 原始完整数据集路径，从中抽取验证集/测试集
        test_path (str, optional): 独立测试集路径

    返回:
        Dict: 包含验证集和测试集评估结果的字典
    """
    # 根据数据集路径提前生成数据集标识符，用于初始化独立的日志目录
    dataset_id = build_dataset_id(data_dir, data_path, original_data_path)
    logger = setup_logger("main", os.path.join(settings.LOG_DIR, dataset_id, "training.log"))

    logger.info("=" * 80)
    logger.info("论文: 基于自适应语义-字符多样性联合筛选与跨关系融合的不平衡中文分类")
    logger.info("=" * 80)
    logger.info("数据集标识符: %s", dataset_id)

    # ============================================================
    # 0. 加载数据并自动推断类别数量
    # ============================================================
    test_texts, test_labels = [], []
    if data_dir is not None:
        # 从文件夹加载数据集（支持 train/val/test 自动划分）
        logger.info("从文件夹加载数据集: %s", data_dir)
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, num_classes = load_dataset_folder(
            data_dir,
            train_val_split=settings.TRAIN_VAL_SPLIT,
            random_seed=settings.RANDOM_SEED,
        )
        config = build_config(data_dir=data_dir, num_classes=num_classes)
    elif original_data_path is not None:
        # 训练集 + 原始完整数据集模式：从原始数据集中每类抽取固定数量作为 val/test
        logger.info("从训练集文件加载: %s", data_path)
        train_texts, train_labels = load_data(data_path)
        if not train_texts:
            raise ValueError(f"在 {data_path} 中未找到有效数据")
        num_classes = infer_num_classes(data_path)

        logger.info("从原始完整数据集加载: %s", original_data_path)
        full_texts, full_labels = load_data(original_data_path)
        if not full_texts:
            raise ValueError(f"在 {original_data_path} 中未找到有效数据")
        full_num_classes = infer_num_classes(original_data_path)
        if full_num_classes != num_classes:
            logger.warning(
                "训练集类别数 %d 与原始数据集类别数 %d 不一致",
                num_classes, full_num_classes,
            )

        # 保存/加载从原始数据集抽取的验证集和测试集
        # 使用训练集文件名作为前缀，避免不同数据集互相覆盖
        train_basename = os.path.splitext(os.path.basename(data_path))[0]
        auto_split_dir = os.path.join(os.path.dirname(data_path) or ".", "auto_split")
        os.makedirs(auto_split_dir, exist_ok=True)
        val_path = os.path.join(auto_split_dir, f"{train_basename}_val.txt")
        test_path_auto = os.path.join(auto_split_dir, f"{train_basename}_test.txt")

        if os.path.exists(val_path) and os.path.exists(test_path_auto):
            logger.info("发现已保存的验证集/测试集，直接加载: %s, %s", val_path, test_path_auto)
            val_texts, val_labels = load_data(val_path)
            test_texts, test_labels = load_data(test_path_auto)
        else:
            logger.info("从原始数据集中按 %.0f%% 抽取验证集/测试集", settings.TRAIN_VAL_SPLIT * 100)
            val_texts, val_labels, test_texts, test_labels = extract_val_test_from_full_data(
                train_texts, train_labels,
                full_texts, full_labels,
                test_ratio=1.0 - settings.TRAIN_VAL_SPLIT,
                random_seed=settings.RANDOM_SEED,
            )
            save_data(val_texts, val_labels, val_path)
            save_data(test_texts, test_labels, test_path_auto)
            logger.info("验证集/测试集已保存至: %s", auto_split_dir)

        config = build_config(data_path=data_path, original_data_path=original_data_path, num_classes=num_classes)
    else:
        # 从单个文件加载，并按 7:3 划分为训练集/测试集（验证集与测试集通用）
        logger.info("从文件加载数据集: %s", data_path)
        texts, labels = load_data(data_path)
        if not texts:
            raise ValueError(f"在 {data_path} 中未找到有效数据")
        num_classes = infer_num_classes(data_path)
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data_three_way(
            texts, labels,
            train_ratio=settings.TRAIN_VAL_SPLIT,
            test_ratio=1.0 - settings.TRAIN_VAL_SPLIT,
            random_seed=settings.RANDOM_SEED,
        )
        config = build_config(data_path=data_path, num_classes=num_classes)

        # 保存自动划分的数据集，方便复用和检查
        # 使用数据集文件名作为前缀，避免不同数据集互相覆盖
        basename = os.path.splitext(os.path.basename(data_path))[0]
        auto_split_dir = os.path.join(os.path.dirname(data_path) or ".", "auto_split")
        os.makedirs(auto_split_dir, exist_ok=True)
        save_data(train_texts, train_labels, os.path.join(auto_split_dir, f"{basename}_train.txt"))
        save_data(test_texts, test_labels, os.path.join(auto_split_dir, f"{basename}_test.txt"))
        logger.info("自动划分的数据集已保存至: %s (验证集与测试集通用，仅保存 test.txt)", auto_split_dir)

    # 若提供了独立测试集，则加载（此逻辑在 original_data_path 模式下已被参数校验禁止）
    if test_path is not None:
        test_texts, test_labels = load_data(test_path)
        logger.info("加载独立测试集: %d 条样本", len(test_texts))

    # 输出数据集基本信息
    logger.info("自动推断类别数量: %d", num_classes)
    logger.info("训练集: %d 条, 验证/测试集: %d 条", len(train_texts), len(val_texts))
    if test_texts:
        logger.info("测试集:  %d 条 (与验证集通用)", len(test_texts))
    print_class_distribution(get_class_distribution(train_labels, num_classes), logger)

    # 在 Stage-1 训练前绘制原始数据分布图
    from visualization.dataset_stats import plot_dataset_distributions, print_dataset_statistics
    plot_dataset_distributions(
        train_labels, None, test_labels if test_texts else None,
        num_classes=num_classes,
        save_path=os.path.join(config.plots_dir, "dataset_distribution_original.png"),
        logger=logger,
    )
    print_dataset_statistics(train_labels, None, test_labels if test_texts else None, num_classes=num_classes, logger=logger)

    # 输出配置信息
    logger.info("配置从 settings.py 加载")
    logger.info("设备: %s", config.bert.device)
    logger.info("IR 阈值: %.2f", config.data.ir_threshold)
    logger.info("Stage-1 epoch 数: %d, Stage-2 epoch 数: %d",
                config.bert.stage1_epochs, config.bert.stage2_epochs)
    logger.info("GNN epoch 数: %d, 隐藏层维度: %d", config.gnn.num_epochs, config.gnn.hidden_channels)

    # 设置随机种子并创建必要目录
    set_seed(settings.RANDOM_SEED)
    os.makedirs(config.plots_dir, exist_ok=True)

    # ============================================================
    # 1. Stage-1 BERT: 在原始数据 D 上微调
    # ============================================================
    # 检查点目录已按 dataset_id 隔离，使用固定文件名即可
    stage1_save_path = os.path.join(config.bert.save_dir, "stage1_best.pt")

    if os.path.exists(stage1_save_path):
        logger.info("\n" + "=" * 60)
        logger.info("[Stage-1 BERT] 发现已保存的 Stage-1 模型: %s，跳过训练", stage1_save_path)
        logger.info("=" * 60)
        stage1_model = BERTClassifier(
            model_name=config.bert.model_name,
            num_classes=num_classes,
            dropout=0.1,
        ).to(config.bert.device)
        checkpoint = torch.load(stage1_save_path, map_location=config.bert.device, weights_only=True)
        stage1_model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Stage-1 模型已加载")
    else:
        stage1_model = run_stage1(
            config,
            train_texts, train_labels,
            val_texts, val_labels,
            num_classes=num_classes,
            logger=logger,
            save_path=stage1_save_path,
        )

    # ============================================================
    # 2. 自适应数据增强 (DeepSeek + 语义-字符多样性筛选)
    # ============================================================
    # 数据增强在 Stage-1 BERT 之后执行，使用微调后的 BERT 评估语义一致性。
    # 确定增强后训练集保存目录
    # 使用数据集文件名作为前缀，避免不同数据集互相覆盖
    if data_dir is not None:
        aug_split_dir = os.path.join(data_dir, "auto_split", "augmented")
        aug_basename = os.path.basename(os.path.normpath(data_dir))
    else:
        aug_split_dir = os.path.join(os.path.dirname(data_path) or ".", "auto_split", "augmented")
        aug_basename = os.path.splitext(os.path.basename(data_path))[0]
    os.makedirs(aug_split_dir, exist_ok=True)
    train_aug_path = os.path.join(aug_split_dir, f"{aug_basename}_train_aug.txt")

    # 检查是否已经存在增强后的训练集
    # 只有在启用增强且需要 Stage-2 时才使用增强数据，否则跳过
    # 若缓存的增强文件标签与当前 num_classes 不一致（如数据集被修改后重新运行），则强制重新生成
    aug_cache_valid = False
    if os.path.exists(train_aug_path) and config.data.enable_augmentation and not config.bert.skip_stage2:
        _tmp_texts, _tmp_labels = load_data(train_aug_path)
        if _tmp_labels and all(0 <= lbl < num_classes for lbl in _tmp_labels):
            aug_cache_valid = True
        else:
            logger.warning(
                "[数据增强] 发现已增强的训练集，但标签范围与当前类别数 %d 不一致，将重新生成",
                num_classes,
            )

    if aug_cache_valid:
        logger.info("\n" + "=" * 60)
        logger.info("[数据增强] 发现已增强的训练集，跳过 API 调用")
        logger.info("=" * 60)
        aug_train_texts, aug_train_labels = load_data(train_aug_path)
        logger.info("增强后训练集: %d 条样本 (原 %d 条)", len(aug_train_texts), len(train_texts))
        print_class_distribution(get_class_distribution(aug_train_labels, num_classes), logger)
    elif config.data.enable_augmentation and config.aug.deepseek_api_key and not config.bert.skip_stage2:
        logger.info("\n" + "=" * 60)
        logger.info("[数据增强] DeepSeek 生成 + 语义-字符多样性联合筛选")
        logger.info("=" * 60)

        # 使用 Stage-1 微调后的 BERT 模型初始化增强器
        # 若 Stage-1 进行了 tokenizer 领域自适应，必须复用该 tokenizer，
        # 否则 token ID 与 BERT 嵌入层不匹配。
        from transformers import BertTokenizer
        adapted_tokenizer_dir = os.path.join(config.bert.save_dir, "tokenizer")
        if os.path.isdir(adapted_tokenizer_dir):
            stage1_tokenizer = BertTokenizer.from_pretrained(adapted_tokenizer_dir)
            logger.info("数据增强复用 Stage-1 领域自适应 tokenizer")
        else:
            stage1_tokenizer = None
        augmenter = AdaptiveDataAugmenter(
            config, logger,
            bert_model=stage1_model.bert,
            bert_tokenizer=stage1_tokenizer,
        )

        # 仅对训练集执行增强（验证集和测试集保持原始分布）
        aug_train_texts, aug_train_labels, adaptive_weights = augmenter.augment(
            train_texts, train_labels, num_classes=num_classes,
        )
        logger.info("增强后训练集: %d 条样本 (原 %d 条)", len(aug_train_texts), len(train_texts))
        print_class_distribution(get_class_distribution(aug_train_labels, num_classes), logger)

        # 保存增强后的训练集
        save_data(aug_train_texts, aug_train_labels, train_aug_path)
        logger.info("增强后的训练集已保存至: %s", train_aug_path)

        # 绘制少数类自适应增强权重 A_c / B_c 柱状图
        from visualization.plot_utils import plot_adaptive_weights
        plot_adaptive_weights(
            adaptive_weights,
            os.path.join(config.plots_dir, "adaptive_weights.png"),
        )
    else:
        if not config.aug.deepseek_api_key and config.data.enable_augmentation:
            logger.warning("settings.py 中未设置 DeepSeek API 密钥，跳过数据增强。")
        aug_train_texts, aug_train_labels = train_texts, train_labels
        adaptive_weights = {}

    # 验证集和测试集不增强
    aug_val_texts, aug_val_labels = val_texts, val_labels
    aug_test_texts, aug_test_labels = test_texts, test_labels

    # 增强后绘制数据集类别分布图（训练集已变化，val/test 保持原始）
    plot_dataset_distributions(
        aug_train_labels, None, aug_test_labels if aug_test_texts else None,
        num_classes=num_classes,
        save_path=os.path.join(config.plots_dir, "dataset_distribution.png"),
        logger=logger,
    )
    print_dataset_statistics(aug_train_labels, None, aug_test_labels if aug_test_texts else None, num_classes=num_classes, logger=logger)

    # 保存 Stage-2 使用的完整训练集（原始 + 增强）
    stage2_train_path = os.path.join(config.aug.save_dir, "stage2_train.txt")
    save_data(aug_train_texts, aug_train_labels, stage2_train_path)
    logger.info("Stage-2 训练数据已保存至 %s", stage2_train_path)

    # ============================================================
    # 3. Stage-2 BERT: 在 D_aug 上重新训练，取最优 Macro-F1，然后冻结
    # ============================================================
    # 检查点目录已按 dataset_id 隔离，使用固定文件名即可
    stage2_save_path = os.path.join(config.bert.save_dir, "stage2_best.pt")

    if config.bert.skip_stage2:
        # 消融实验：跳过 Stage-2，直接使用 Stage-1 模型作为 theta*
        logger.info("\n" + "=" * 60)
        logger.info("[消融实验] 跳过 Stage-2 BERT 微调，使用 Stage-1 模型作为 theta*")
        logger.info("=" * 60)
        os.makedirs(config.bert.save_dir, exist_ok=True)
        from utils import save_checkpoint
        save_checkpoint(stage1_model, None, 0, {"f1_macro": 0.0}, stage2_save_path)
        stage2_model = stage1_model
        stage2_model.eval()
        logger.info("Stage-1 模型已保存并作为 theta* 使用")
    elif os.path.exists(stage2_save_path):
        logger.info("\n" + "=" * 60)
        logger.info("[Stage-2 BERT] 发现已保存的 Stage-2 模型: %s，跳过训练", stage2_save_path)
        logger.info("=" * 60)
        stage2_model = BERTClassifier(
            model_name=config.bert.model_name,
            num_classes=num_classes,
            dropout=0.1,
        ).to(config.bert.device)
        checkpoint = torch.load(stage2_save_path, map_location=config.bert.device, weights_only=True)
        stage2_model.load_state_dict(checkpoint["model_state_dict"])
        # 冻结为 theta*
        for param in stage2_model.parameters():
            param.requires_grad = False
        stage2_model.eval()
        logger.info("Stage-2 模型已加载并冻结为 theta*")
    else:
        stage2_model = run_stage2(
            config,
            train_data_path=stage2_train_path,
            val_texts=aug_val_texts,
            val_labels=aug_val_labels,
            num_classes=num_classes,
            stage1_model=stage1_model,
            logger=logger,
            save_path=stage2_save_path,
        )

    # ============================================================
    # 4. 冻结 theta* 并提取均值池化特征
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("[特征提取] 冻结 theta*，进行均值池化特征提取")
    logger.info("=" * 60)

    # 初始化 BERT 特征提取器（复用 Stage-1 保存的领域自适应 tokenizer）
    adapted_tokenizer_dir = os.path.join(config.bert.save_dir, "tokenizer")
    tokenizer_path = adapted_tokenizer_dir if os.path.isdir(adapted_tokenizer_dir) else None
    if tokenizer_path:
        logger.info("特征提取加载领域自适应 tokenizer: %s", tokenizer_path)
    bert_extractor = BERTFeatureExtractor(
        config.bert.model_name,
        device=config.bert.device,
        tokenizer_path=tokenizer_path,
    )
    bert_extractor.load_finetuned_weights(stage2_save_path)
    bert_extractor.freeze()  # 冻结 BERT 参数，不再参与训练

    # 提取训练集和验证集的均值池化特征
    train_features = bert_extractor.extract_mean_pooled(aug_train_texts, max_length=config.bert.max_length)
    val_features = bert_extractor.extract_mean_pooled(aug_val_texts, max_length=config.bert.max_length)

    feature_info = f"训练特征: {train_features.shape}, 验证特征: {val_features.shape}"
    test_features = None
    if aug_test_texts:
        test_features = bert_extractor.extract_mean_pooled(aug_test_texts, max_length=config.bert.max_length)
        feature_info += f", 测试特征: {test_features.shape}"
    logger.info(feature_info)

    # 保存提取的特征到磁盘
    os.makedirs(config.bert.save_dir, exist_ok=True)
    np.savez(os.path.join(config.bert.save_dir, "train_features.npz"),
             features=train_features, labels=np.array(aug_train_labels))
    np.savez(os.path.join(config.bert.save_dir, "val_features.npz"),
             features=val_features, labels=np.array(aug_val_labels))
    if aug_test_texts:
        np.savez(os.path.join(config.bert.save_dir, "test_features.npz"),
                 features=test_features, labels=np.array(aug_test_labels))
    logger.info("特征已保存至 %s", config.bert.save_dir)

    # ============================================================
    # 5. 构建图构建器 (Mode 1: doc-char, doc-word)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("[图构建器] 实例级二分图 (文档-字符, 文档-词语)")
    logger.info("=" * 60)

    from transformers import BertTokenizer
    adapted_tokenizer_dir = os.path.join(config.bert.save_dir, "tokenizer")
    if os.path.isdir(adapted_tokenizer_dir):
        tokenizer = BertTokenizer.from_pretrained(adapted_tokenizer_dir)
        logger.info("图构建器加载领域自适应 tokenizer: %s", adapted_tokenizer_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.bert.model_name)
    # 提取 BERT 词表嵌入矩阵，用于初始化字/词节点，使图节点具有真实语义
    embedding_matrix = (
        bert_extractor.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
        if bert_extractor is not None else None
    )
    graph_builder = InstanceGraphBuilder(
        tokenizer=tokenizer,
        logger=logger,
        embedding_matrix=embedding_matrix,
    )

    logger.info("图构建器已初始化，图将在训练/评估时按 batch 按需构建")

    # ============================================================
    # 6. 使用跨关系注意力训练 GNN
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("[GNN 训练] 双图 + 跨关系注意力")
    logger.info("=" * 60)

    # 初始化 DualGraphGNN 模型
    gnn_model = DualGraphGNN(
        config,
        bert_dim=config.bert.hidden_size,
        num_classes=num_classes,
        fusion_type=config.gnn.fusion_type,
    ).to(config.gnn.device)

    logger.info("GNN 参数量: %s", f"{sum(p.numel() for p in gnn_model.parameters()):,}")

    # 训练 GNN
    trainer, best_epoch, best_f1, gnn_history = train_gnn(
        config, gnn_model,
        aug_train_texts, train_features, aug_train_labels,
        aug_val_texts, val_features, aug_val_labels,
        graph_builder,
        logger=logger,
    )

    # ============================================================
    # 7. 最终评估
    # ============================================================
    logger.info("\n" + "=" * 80)
    logger.info("最终评估 (最优 epoch: %d)", best_epoch)
    logger.info("=" * 80)

    # 预计算测试集 token 缓存（如果存在测试集）
    if aug_test_texts:
        graph_builder.precompute_tokens(aug_test_texts, cache_key="test")

    results = run_test(
        config, gnn_model,
        aug_val_texts, aug_val_labels, val_features,
        graph_builder,
        test_texts=aug_test_texts or None,
        test_labels=aug_test_labels or None,
        test_features=test_features,
        logger=logger,
        val_cache_key="val",
        test_cache_key="test" if aug_test_texts else None,
    )

    logger.info("=" * 80)
    return results


def run_test_only(test_path: str):
    """
    仅测试模式：加载已训练好的模型，对独立测试集进行推理评估。
    要求 checkpoints/<dataset_id>/bert/stage2_best.pt 和 checkpoints/<dataset_id>/gnn/best_gnn.pt 已存在。
    """
    # 加载测试数据
    test_texts, test_labels = load_data(test_path)
    if not test_texts:
        print(f"错误: {test_path} 中没有有效数据。")
        return

    # 从测试数据推断类别数
    num_classes = max(test_labels) + 1

    # 构建配置以获取 dataset_id，确保日志与模型路径按数据集隔离
    config = build_config(data_path=test_path, num_classes=num_classes)
    logger = setup_logger("test", os.path.join(config.log_dir, "test.log"))
    logger.info("=" * 80)
    logger.info("[Test Only Mode] Loading trained model and evaluating on test set")
    logger.info("=" * 80)

    logger.info("Loading test data from: %s", test_path)
    logger.info("Auto-inferred number of classes: %d", num_classes)
    logger.info("Test samples: %d", len(test_texts))

    set_seed(settings.RANDOM_SEED)

    # 检查模型文件是否存在
    stage2_path = os.path.join(config.bert.save_dir, "stage2_best.pt")
    best_gnn_path = os.path.join(config.gnn.save_dir, "best_gnn.pt")
    if not os.path.exists(stage2_path):
        logger.error("Stage-2 BERT model not found at %s. Please train first.", stage2_path)
        return
    if not os.path.exists(best_gnn_path):
        logger.error("Best GNN model not found at %s. Please train first.", best_gnn_path)
        return

    # 加载 BERT 特征提取器（复用 Stage-1 保存的领域自适应 tokenizer）
    logger.info("Loading BERT feature extractor from %s", config.bert.model_name)
    adapted_tokenizer_dir = os.path.join(config.bert.save_dir, "tokenizer")
    tokenizer_path = adapted_tokenizer_dir if os.path.isdir(adapted_tokenizer_dir) else None
    bert_extractor = BERTFeatureExtractor(
        config.bert.model_name,
        device=config.bert.device,
        tokenizer_path=tokenizer_path,
    )
    bert_extractor.load_finetuned_weights(stage2_path)
    bert_extractor.freeze()

    # 提取测试集特征
    logger.info("Extracting BERT features for test set...")
    test_features = bert_extractor.extract_mean_pooled(test_texts, max_length=config.bert.max_length)
    logger.info("Test features shape: %s", test_features.shape)

    # 初始化图构建器，图将在评估时按 batch 按需构建
    logger.info("Initializing graph builder for test set...")
    from transformers import BertTokenizer
    adapted_tokenizer_dir = os.path.join(config.bert.save_dir, "tokenizer")
    if os.path.isdir(adapted_tokenizer_dir):
        tokenizer = BertTokenizer.from_pretrained(adapted_tokenizer_dir)
        logger.info("图构建器加载领域自适应 tokenizer: %s", adapted_tokenizer_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.bert.model_name)
    embedding_matrix = (
        bert_extractor.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
        if bert_extractor is not None else None
    )
    graph_builder = InstanceGraphBuilder(
        tokenizer=tokenizer,
        logger=logger,
        embedding_matrix=embedding_matrix,
    )

    # 加载 GNN 模型
    logger.info("Loading GNN model from %s", best_gnn_path)
    gnn_model = DualGraphGNN(
        config,
        bert_dim=config.bert.hidden_size,
        num_classes=num_classes,
    ).to(config.gnn.device)
    gnn_model.load_state_dict(torch.load(best_gnn_path, map_location=config.gnn.device, weights_only=True))

    # 推理评估
    logger.info("Evaluating on test set...")
    from train import GNNTrainer
    trainer = GNNTrainer(config, gnn_model, logger)
    from test import evaluate_set
    metrics = evaluate_set(
        trainer, test_texts, test_labels, test_features,
        graph_builder,
        batch_size=config.gnn.batch_size,
        name="TEST ONLY",
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("Test Only Mode Complete")
    logger.info("=" * 80)
    return metrics


def main():
    """
    程序入口函数。
    初始化 jieba 分词器，解析命令行参数并启动训练或测试流水线。
    """
    import jieba
    jieba.initialize()  # 预加载 jieba 词典，避免运行时延迟

    args = parse_args()

    # 仅测试模式
    if args.test_only:
        if not os.path.isfile(args.test_path):
            print(f"错误: {args.test_path} 不是有效的文件。")
            sys.exit(1)
        run_test_only(args.test_path)
        return

    # 训练模式
    if args.data_dir is not None:
        data_dir = args.data_dir
        if not os.path.isdir(data_dir):
            print(f"错误: {data_dir} 不是有效的目录。")
            sys.exit(1)
        run_pipeline(data_dir=data_dir, test_path=args.test_path)
    else:
        data_path = args.data_path
        if not os.path.isfile(data_path):
            print(f"错误: {data_path} 不是有效的文件。")
            sys.exit(1)
        run_pipeline(
            data_path=data_path,
            original_data_path=args.original_data_path,
            test_path=args.test_path,
        )


if __name__ == "__main__":
    main()
