import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting to import required modules...")
import sys
sys.path.append('.')
logger.info("Python path updated")

try:
    import os
    import threading
    import os.path as osp
    import json
    import time
    import pickle
    import argparse
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from multiprocessing import Pool
    logger.info("Basic modules imported successfully")

    import torch
    logger.info("PyTorch imported successfully")
    import stark_qa
    logger.info("stark_qa imported successfully")
    logger.info("Starting to import avatar modules...")

    from avatar.models import get_model
    from avatar.models.causal_score import CausalScoreCalculator
    from stark_qa.tools.seed import set_seed
    from avatar.kb import Flickr30kEntities
    from avatar.qa_datasets import QADataset
    from scripts.args import parse_args_w_defaults
    logger.info("All modules imported successfully")
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    raise

if __name__ == '__main__':
    try:
        logger.info("Starting main execution...")
        
        # 进程信息记录
        logger.info(f"Process ID: {os.getpid()}")
        logger.info(f"Thread count: {threading.active_count()}")
        logger.info(f"Current thread: {threading.current_thread()}")
        logger.info(f"Python version: {sys.version}")
        
        # CUDA检查
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using device: {device}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            
        # 参数解析
        logger.info("Parsing arguments...")
        args = parse_args_w_defaults('config/default_args.json')
        set_seed(args.seed)
        logger.info(f"Arguments parsed: {args}")
        
        # 数据集处理
        logger.info(f"Processing dataset: {args.dataset}")
        if args.dataset in ['amazon', 'mag', 'prime']:
            logger.info("Setting up directories for document dataset...")
            emb_root = osp.join(args.emb_dir, args.dataset, args.emb_model)
            args.query_emb_dir = osp.join(emb_root, 'query')
            args.node_emb_dir = osp.join(emb_root, 'doc')
            args.chunk_emb_dir = osp.join(emb_root, 'chunk')
            os.makedirs(args.query_emb_dir, exist_ok=True)
            os.makedirs(args.node_emb_dir, exist_ok=True)
            os.makedirs(args.chunk_emb_dir, exist_ok=True)
            logger.info("Directories created successfully")

            logger.info("Loading document KB and QA dataset...")
            kb = stark_qa.load_skb(args.dataset)
            qa_dataset = stark_qa.load_qa(args.dataset)
            logger.info("Document dataset loaded successfully")

        elif args.dataset == 'flickr30k_entities':
            logger.info("Setting up directories for Flickr30k...")
            emb_root = osp.join(args.emb_dir, args.dataset, args.emb_model)
            args.chunk_emb_dir = None
            args.query_emb_dir = osp.join(emb_root, 'query')
            args.node_emb_dir = osp.join(emb_root, 'image')
            os.makedirs(args.query_emb_dir, exist_ok=True)
            os.makedirs(args.node_emb_dir, exist_ok=True)
            logger.info("Directories created successfully")

            logger.info("Loading Flickr30k KB...")
            kb = Flickr30kEntities(root=args.root_dir)
            logger.info("KB loaded successfully")
            
            logger.info("Loading QA dataset...")
            qa_dataset = QADataset(name=args.dataset, root=args.root_dir)
            logger.info("QA dataset loaded successfully")
        
        # 模型初始化
        logger.info("Initializing model...")
        model = get_model(args, kb)
        logger.info("Model initialized successfully")
        
        # 设置模型路径
        model.parent_pred_path = osp.join(args.output_dir, f'eval/{args.dataset}/VSS/{args.emb_model}/eval_results_test.csv')
        logger.info("Model path set successfully")
        
        # 初始化因果计算器
        logger.info("Initializing causal calculator...")
        causal_calculator = CausalScoreCalculator()
        logger.info("Causal calculator initialized")
        
        # 生成代码部分
        if args.dataset in ['amazon', 'mag', 'prime']:
            logger.info("Generating groups for document dataset...")
            logger.info("Generating train group...")
            group = model.generate_group(qa_dataset, batch_size=5, n_init_examples=200, split='train')
            logger.info("Generating validation group...")
            group = model.generate_group(qa_dataset, batch_size=5, split='val')
            logger.info("Generating test group...")
            group = model.generate_group(qa_dataset, batch_size=5, split='test')
            logger.info("Groups generated successfully")
        
        # 设置指标
        logger.info("Setting up metrics...")
        metrics = [
            'mrr', 'map', 'rprecision',
            'recall@5', 'recall@10', 'recall@20', 'recall@50', 'recall@100',
            'hit@1', 'hit@3', 'hit@5', 'hit@10', 'hit@20', 'hit@50'
        ]
        logger.info("Metrics set up successfully")
        
        # 计算因果指标
        logger.info("Calculating causal metrics...")
        causal_metrics = {
            'causal_score': causal_calculator.calculate_overall_score(),
            'direct_effect': causal_calculator.get_direct_effect(),
            'indirect_effect': causal_calculator.get_indirect_effect()
        }
        logger.info("Causal metrics calculated")
        
        # 开始优化
        logger.info("Starting optimization process...")
        model.optimize_actions(
            qa_dataset=qa_dataset,
            seed=args.seed,
            group_idx=args.group_idx,
            use_group=args.use_group,
            n_eval=args.n_eval,
            n_examples=args.n_examples,
            n_total_steps=args.n_total_steps,
            topk_eval=args.topk_eval,
            topk_test=args.topk_test,
            batch_size=args.batch_size,
            metrics=metrics,
            causal_metrics=causal_metrics
        )
        logger.info("Optimization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise