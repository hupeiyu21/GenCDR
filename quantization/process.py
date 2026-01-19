#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from logging import getLogger
from sklearn.decomposition import PCA
from decoder.utils import get_config, init_seed, init_logger, init_device, get_dataset
from decoder.models.Gen.tokenizer import GenTokenizer
from accelerate import Accelerator


class DataProcessor:
    
    def __init__(self, category: str, config_dict: dict = None):
        self.category = category
        
        self.config = get_config(
            model_name='Gen',
            dataset_name='AmazonReviews2014',
            config_dict=config_dict or {'category': category}
        )
        
        self.config['device'], self.config['use_ddp'] = init_device()
        self.accelerator = Accelerator()
        self.config['accelerator'] = self.accelerator
        
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        
        self.cache_dir = os.path.join("cache", "AmazonReviews2014", category)
        self.processed_dir = os.path.join(self.cache_dir, "processed")
        self.final_pca_path = os.path.join(self.processed_dir, "final_pca_embeddings.npy")
        
        self.log(f'[QUANTIZATION] Data Processor initialized for category: {category}')
        self.log(f'[QUANTIZATION] Device: {self.config["device"]}')
        self.log(f'[QUANTIZATION] Cache directory: {self.cache_dir}')
    
    def log(self, message, level='info'):
        if self.accelerator.is_main_process:
            if level == 'info':
                self.logger.info(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'warning':
                self.logger.warning(message)
    
    def check_file_status(self):
        status = {
            'final_pca_embeddings': os.path.exists(self.final_pca_path),
            'raw_embeddings': False,
            'processed_data': False,
            'raw_data': False
        }
        
        if os.path.exists(self.processed_dir):
            sent_emb_pattern = f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
            raw_emb_path = os.path.join(self.processed_dir, sent_emb_pattern)
            status['raw_embeddings'] = os.path.exists(raw_emb_path)
            status['raw_embeddings_path'] = raw_emb_path if status['raw_embeddings'] else None
        
        processed_files = [
            'all_item_seqs.json',
            'id_mapping.json',
            f'metadata.{self.config["metadata"]}.json'
        ]
        status['processed_data'] = all(
            os.path.exists(os.path.join(self.processed_dir, f)) for f in processed_files
        )
        
        raw_data_files = [
            f'reviews_{self.category}_5.json.gz',
            f'meta_{self.category}.json.gz'
        ]
        status['raw_data'] = all(
            os.path.exists(os.path.join(self.cache_dir, f)) for f in raw_data_files
        )
        
        return status
    
    def apply_pca_to_embeddings(self, raw_emb_path):
        self.log("[QUANTIZATION] Applying PCA to raw embeddings...")
        
        embeddings = np.fromfile(raw_emb_path, dtype=np.float32)
        embeddings = embeddings.reshape(-1, self.config['sent_emb_dim'])
        
        self.log(f"[QUANTIZATION] Loaded raw embeddings: shape={embeddings.shape}")
        
        pca_dim = self.config.get('sent_emb_pca', 0)
        if pca_dim > 0 and pca_dim < embeddings.shape[1]:
            self.log(f"[QUANTIZATION] Applying PCA: {embeddings.shape[1]} -> {pca_dim}")
            pca = PCA(n_components=pca_dim)
            embeddings = pca.fit_transform(embeddings)
            self.log(f"[QUANTIZATION] PCA applied: explained variance ratio = {pca.explained_variance_ratio_.sum():.4f}")
        else:
            self.log("[QUANTIZATION] No PCA applied (pca_dim <= 0 or >= original_dim)")
        
        os.makedirs(self.processed_dir, exist_ok=True)
        
        np.save(self.final_pca_path, embeddings.astype(np.float32))
        self.log(f"[QUANTIZATION] Final embeddings saved to: {self.final_pca_path}")
        self.log(f"[QUANTIZATION] Final embedding shape: {embeddings.shape}")
    
    def download_and_process_data(self):
        self.log("[QUANTIZATION] Starting data download and processing...")
        
        dataset = get_dataset('AmazonReviews2014')(self.config)
        
        self.log(f"[QUANTIZATION] Dataset processed successfully:")
        self.log(f"[QUANTIZATION]   - Number of users: {dataset.n_users}")
        self.log(f"[QUANTIZATION]   - Number of items: {dataset.n_items}")
        self.log(f"[QUANTIZATION]   - Cache directory: {dataset.cache_dir}")
        
        return dataset
    
    def generate_embeddings(self, dataset):
        self.log("[QUANTIZATION] Starting embedding generation...")
        
        tokenizer = GenTokenizer(self.config, dataset)
        
        processed_dir = os.path.join(dataset.cache_dir, 'processed')
        sent_emb_file = os.path.join(processed_dir, f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb')
        sem_ids_file = os.path.join(processed_dir, f'{os.path.basename(self.config["sent_emb_model"])}_{tokenizer.index_factory}.sem_ids')
        
        self.log(f"[QUANTIZATION] Embedding generation completed:")
        self.log(f"[QUANTIZATION]   - Sentence embeddings: {sent_emb_file}")
        self.log(f"[QUANTIZATION]   - Semantic IDs: {sem_ids_file}")
        self.log(f"[QUANTIZATION]   - Vocabulary size: {tokenizer.vocab_size}")
        
        return sent_emb_file, sem_ids_file
    
    def run_smart_pipeline(self):
        self.log("[QUANTIZATION] " + "=" * 60)
        self.log("[QUANTIZATION] Starting SMART data processing pipeline")
        self.log("[QUANTIZATION] " + "=" * 60)
        
        status = self.check_file_status()
        self.log("[QUANTIZATION] File status check:")
        self.log(f"[QUANTIZATION]   - Final PCA embeddings: {'✓' if status['final_pca_embeddings'] else '✗'}")
        self.log(f"[QUANTIZATION]   - Raw embeddings: {'✓' if status['raw_embeddings'] else '✗'}")
        self.log(f"[QUANTIZATION]   - Processed data: {'✓' if status['processed_data'] else '✗'}")
        self.log(f"[QUANTIZATION]   - Raw data: {'✓' if status['raw_data'] else '✗'}")
        
        dataset = None
        
        if status['final_pca_embeddings']:
            self.log("[QUANTIZATION] Final PCA embeddings found - pipeline complete!")
            if status['processed_data']:
                dataset = get_dataset('AmazonReviews2014')(self.config)
        
        elif status['raw_embeddings']:
            self.log("[QUANTIZATION] Raw embeddings found - applying PCA...")
            self.apply_pca_to_embeddings(status['raw_embeddings_path'])
            if status['processed_data']:
                dataset = get_dataset('AmazonReviews2014')(self.config)
        
        elif status['processed_data']:
            self.log("[QUANTIZATION] Processed data found - generating embeddings...")
            dataset = get_dataset('AmazonReviews2014')(self.config)
            sent_emb_file, sem_ids_file = self.generate_embeddings(dataset)
            if self.config.get('sent_emb_pca', 0) > 0:
                self.apply_pca_to_embeddings(sent_emb_file)
        
        else:
            self.log("[QUANTIZATION] Starting from scratch - downloading and processing data...")
            dataset = self.download_and_process_data()
            sent_emb_file, sem_ids_file = self.generate_embeddings(dataset)
            if self.config.get('sent_emb_pca', 0) > 0:
                self.apply_pca_to_embeddings(sent_emb_file)
        
        if os.path.exists(self.final_pca_path):
            final_embeddings = np.load(self.final_pca_path)
            self.log(f"[QUANTIZATION] Final embeddings ready: {final_embeddings.shape}")
        else:
            self.log("[QUANTIZATION] Final embeddings not found - something went wrong!")
            return None
        
        if dataset is None and status['processed_data']:
            dataset = get_dataset('AmazonReviews2014')(self.config)
        
        summary = {
            'category': self.category,
            'cache_dir': self.cache_dir,
            'final_embeddings_path': self.final_pca_path,
            'final_embeddings_shape': final_embeddings.shape,
            'pipeline_status': 'completed'
        }
        
        if dataset:
            summary.update({
                'num_users': dataset.n_users,
                'num_items': dataset.n_items,
            })
        
        self.log("[QUANTIZATION] " + "=" * 60)
        self.log("[QUANTIZATION] Smart pipeline completed successfully!")
        self.log("[QUANTIZATION] " + "=" * 60)
        self.log(f"[QUANTIZATION] Summary:")
        self.log(f"[QUANTIZATION]   - Category: {summary['category']}")
        if 'num_users' in summary:
            self.log(f"[QUANTIZATION]   - Users: {summary['num_users']}")
            self.log(f"[QUANTIZATION]   - Items: {summary['num_items']}")
        self.log(f"[QUANTIZATION]   - Final embeddings: {summary['final_embeddings_path']}")
        self.log(f"[QUANTIZATION]   - Embedding shape: {summary['final_embeddings_shape']}")
        
        return summary

    def run_full_pipeline(self):
        self.log("[QUANTIZATION] " + "=" * 50)
        self.log("[QUANTIZATION] Starting full data processing pipeline")
        self.log("[QUANTIZATION] " + "=" * 50)
        
        dataset = self.download_and_process_data()
        sent_emb_file, sem_ids_file = self.generate_embeddings(dataset)
        emb_info = self.get_embedding_info(dataset)
        
        summary = {
            'category': self.category,
            'cache_dir': dataset.cache_dir,
            'num_users': dataset.n_users,
            'num_items': dataset.n_items,
            'embedding_info': emb_info,
            'files': {
                'sentence_embeddings': sent_emb_file,
                'semantic_ids': sem_ids_file,
                'processed_reviews': os.path.join(dataset.cache_dir, 'processed', 'all_item_seqs.json'),
                'id_mapping': os.path.join(dataset.cache_dir, 'processed', 'id_mapping.json'),
                'metadata': os.path.join(dataset.cache_dir, 'processed', f'metadata.{self.config["metadata"]}.json')
            }
        }
        
        self.log("[QUANTIZATION] " + "=" * 50)
        self.log("[QUANTIZATION] Data processing pipeline completed successfully!")
        self.log("[QUANTIZATION] " + "=" * 50)
        self.log(f"[QUANTIZATION] Summary:")
        self.log(f"[QUANTIZATION]   - Category: {summary['category']}")
        self.log(f"[QUANTIZATION]   - Users: {summary['num_users']}")
        self.log(f"[QUANTIZATION]   - Items: {summary['num_items']}")
        self.log(f"[QUANTIZATION]   - Cache directory: {summary['cache_dir']}")
        if emb_info:
            self.log(f"[QUANTIZATION]   - Embedding shape: {emb_info['final_shape']}")
            self.log(f"[QUANTIZATION]   - Embedding dimension: {emb_info['embedding_dim']}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Process Amazon Reviews data and generate embeddings')
    parser.add_argument('--category', type=str, required=True,
                       choices=['Sports_and_Outdoors', 'Beauty', 'Toys_and_Games', 'CDs_and_Vinyl'],
                       help='Amazon category to process')
    parser.add_argument('--sent_emb_model', type=str, default=None,
                       help='Sentence embedding model to use')
    parser.add_argument('--sent_emb_pca', type=int, default=None,
                       help='PCA dimension for sentence embeddings')
    parser.add_argument('--openai_api_key', type=str, default=None,
                       help='OpenAI API key for text-embedding-3 models')
    parser.add_argument('--mode', type=str, default='smart', choices=['smart', 'full'],
                       help='Pipeline mode: smart (check files and start from appropriate step) or full (start from scratch)')
    
    args = parser.parse_args()
    
    config_dict = {'category': args.category}
    if args.sent_emb_model:
        config_dict['sent_emb_model'] = args.sent_emb_model
    if args.sent_emb_pca:
        config_dict['sent_emb_pca'] = args.sent_emb_pca
    if args.openai_api_key:
        config_dict['openai_api_key'] = args.openai_api_key
    
    processor = DataProcessor(args.category, config_dict)
    
    if args.mode == 'smart':
        summary = processor.run_smart_pipeline()
    else:
        summary = processor.run_full_pipeline()
    
    if summary:
        summary_file = os.path.join(summary['cache_dir'], 'processing_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n[QUANTIZATION] Processing summary saved to: {summary_file}")
    else:
        print("\n[QUANTIZATION] Pipeline failed!")


if __name__ == '__main__':
    main()