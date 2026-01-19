# amazon_data_processor.py

import os
import gzip
import json
import math
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
import requests
from urllib.parse import urlparse
import yaml


class AmazonDataProcessor:
    def __init__(self, category: str, cache_dir: str = "cache", config_path: str = None, config: dict = None):
        self.category = category
        self.cache_dir = os.path.join(cache_dir, 'AmazonReviews2014', category)
        self.raw_dir = os.path.join(self.cache_dir, 'raw')
        self.processed_dir = os.path.join(self.cache_dir, 'processed')
        
        self.default_config = {
            'metadata': 'sentence',
            'sent_emb_model': 'text-embedding-3-large',
            'sent_emb_dim': 3072,
            'sent_emb_batch_size': 100,
            'sent_emb_pca': 0,
            'n_codebook': 32,
            'codebook_size': 256,
            'faiss_omp_num_threads': 16,
            'opq_use_gpu': False,
            'opq_gpu_id': 0,
            'openai_api_key': None,
        }
        
        self.config = self._load_config(config_path, config)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.all_item_seqs = {}
        self.id_mapping = {
            'user2id': {},
            'item2id': {},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = {}
        
    def _check_available_category(self):
        available_categories = [
            'Books', 'Electronics', 'Movies_and_TV', 'CDs_and_Vinyl',
            'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen', 'Kindle_Store',
            'Sports_and_Outdoors', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games', 'Video_Games',
            'Tools_and_Home_Improvement', 'Beauty', 'Apps_for_Android',
            'Office_Products', 'Pet_Supplies', 'Automotive',
            'Grocery_and_Gourmet_Food', 'Patio_Lawn_and_Garden', 'Baby',
            'Digital_Music', 'Musical_Instruments', 'Amazon_Instant_Video'
        ]
        assert self.category in available_categories, f'Category "{self.category}" not available. Available categories: {available_categories}'
    
    def download_file(self, url: str, local_path: str):
        if os.path.exists(local_path):
            print(f"File already exists: {local_path}")
            return
            
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(local_path, 'wb') as f, tqdm(
            desc=os.path.basename(local_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def _download_raw(self, data_type: str = 'reviews') -> str:
        url = f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{data_type}_{self.category}{"_5" if data_type == "reviews" else ""}.json.gz'
        base_name = os.path.basename(url)
        local_filepath = os.path.join(self.raw_dir, base_name)
        
        if not os.path.exists(local_filepath):
            self.download_file(url, local_filepath)
        return local_filepath
    
    def _parse_gz(self, path: str):
        with gzip.open(path, 'r') as g:
            for line in g:
                line = line.replace(b'true', b'True').replace(b'false', b'False')
                yield eval(line)
    
    def _load_reviews(self, path: str) -> List[Tuple]:
        print('[DATASET] Loading reviews...')
        reviews = []
        for inter in self._parse_gz(path):
            user = inter['reviewerID']
            item = inter['asin']
            time = inter['unixReviewTime']
            reviews.append((user, item, int(time)))
        return reviews
    
    def _get_item_seqs(self, reviews: List[Tuple]) -> Dict:
        item_seqs = defaultdict(list)
        for user, item, time in reviews:
            item_seqs[user].append((item, time))
        
        for user, item_time in item_seqs.items():
            item_time.sort(key=lambda x: x[1])
            item_seqs[user] = [item for item, _ in item_time]
        return item_seqs
    
    def _remap_ids(self, item_seqs: Dict) -> Tuple[Dict, Dict]:
        print('[DATASET] Remapping user and item IDs...')
        for user, items in item_seqs.items():
            if user not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user)
            
            iids = []
            for item in items:
                if item not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item)
                iids.append(item)
            self.all_item_seqs[user] = iids
        
        return self.all_item_seqs, self.id_mapping
    
    def _process_reviews(self, input_path: str) -> Tuple[Dict, Dict]:
        seq_file = os.path.join(self.processed_dir, 'all_item_seqs.json')
        id_mapping_file = os.path.join(self.processed_dir, 'id_mapping.json')
        
        if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
            print('[DATASET] Reviews have been processed...')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_item_seqs, id_mapping
        
        print('[DATASET] Processing reviews...')
        reviews = self._load_reviews(input_path)
        item_seqs = self._get_item_seqs(reviews)
        all_item_seqs, id_mapping = self._remap_ids(item_seqs)
        
        print('[DATASET] Saving mapping data...')
        with open(seq_file, 'w') as f:
            json.dump(all_item_seqs, f)
        with open(id_mapping_file, 'w') as f:
            json.dump(id_mapping, f)
        
        return all_item_seqs, id_mapping
    
    def _load_metadata(self, path: str, item2id: Dict) -> Dict:
        print('[DATASET] Loading metadata...')
        data = {}
        item_asins = set(item2id.keys())
        for info in tqdm(self._parse_gz(path)):
            if info['asin'] not in item_asins:
                continue
            data[info['asin']] = info
        return data
    
    def clean_text(self, raw_text: str) -> str:
        import re
        import html
        
        if isinstance(raw_text, list):
            raw_text = ' '.join(str(item) for item in raw_text)
        
        text = str(raw_text)
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def _sent_process(self, raw) -> str:
        sentence = ""
        if isinstance(raw, float):
            sentence += str(raw) + '.'
        elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            for v1 in raw:
                for v in v1:
                    sentence += self.clean_text(str(v))[:-1] + ', '
            sentence = sentence[:-2] + '.'
        elif isinstance(raw, list):
            for v1 in raw:
                sentence += self.clean_text(str(v1))
        else:
            sentence = self.clean_text(str(raw))
        return sentence + ' '
    
    def _extract_meta_sentences(self, metadata: Dict) -> Dict:
        print('[DATASET] Extracting meta sentences...')
        item2meta = {}
        for item, meta in tqdm(metadata.items()):
            meta_sentence = ''
            keys = set(meta.keys())
            features_needed = ['title', 'price', 'brand', 'feature', 'categories', 'description']
            for feature in features_needed:
                if feature in keys:
                    meta_sentence += self._sent_process(meta[feature])
            item2meta[item] = meta_sentence
        return item2meta
    
    def _process_meta(self, input_path: str) -> Optional[Dict]:
        process_mode = self.config['metadata']
        meta_file = os.path.join(self.processed_dir, f'metadata.{process_mode}.json')
        
        if os.path.exists(meta_file):
            print('[DATASET] Metadata has been processed...')
            with open(meta_file, 'r') as f:
                return json.load(f)
        
        print(f'[DATASET] Processing metadata, mode: {process_mode}')
        
        if process_mode == 'none':
            return None
        
        item2meta = self._load_metadata(path=input_path, item2id=self.id_mapping['item2id'])
        
        if process_mode == 'sentence':
            item2meta = self._extract_meta_sentences(metadata=item2meta)
        
        with open(meta_file, 'w') as f:
            json.dump(item2meta, f)
        
        return item2meta
    
    def _encode_sent_emb(self, output_path: str) -> np.ndarray:
        print('[TOKENIZER] Encoding sentence embeddings...')
        
        meta_sentences = []
        for i in range(1, len(self.id_mapping['id2item'])):
            item = self.id_mapping['id2item'][i]
            meta_sentences.append(self.item2meta[item])
        
        if 'sentence-transformers' in self.config['sent_emb_model']:
            try:
                from sentence_transformers import SentenceTransformer
                device = self.config.get('device', 'cpu')
                sent_emb_model = SentenceTransformer(self.config['sent_emb_model']).to(device)
                
                sent_embs = sent_emb_model.encode(
                    meta_sentences,
                    convert_to_numpy=True,
                    batch_size=self.config['sent_emb_batch_size'],
                    show_progress_bar=True,
                    device=device
                )
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        elif 'text-embedding-3' in self.config['sent_emb_model']:
            if not self.config['openai_api_key']:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            
            try:
                from openai import OpenAI
                
                client_kwargs = {'api_key': self.config['openai_api_key']}
                if 'openai_base_url' in self.config and self.config['openai_base_url']:
                    client_kwargs['base_url'] = self.config['openai_base_url']
                
                client = OpenAI(**client_kwargs)
                
                sent_embs = []
                for i in tqdm(range(0, len(meta_sentences), self.config['sent_emb_batch_size']), desc='Encoding'):
                    batch = meta_sentences[i:i + self.config['sent_emb_batch_size']]
                    try:
                        responses = client.embeddings.create(
                            input=batch,
                            model=self.config['sent_emb_model']
                        )
                        for response in responses.data:
                            sent_embs.append(response.embedding)
                    except Exception as e:
                        print(f'Encoding failed {i} - {i + self.config["sent_emb_batch_size"]}: {e}')
                        
                        try:
                            new_batch = []
                            for sent in batch:
                                if len(sent) > 8000:
                                    new_batch.append(sent[:8000])
                                else:
                                    new_batch.append(sent)
                            
                            print(f'[TOKENIZER] Retrying batch {i} - {i + self.config["sent_emb_batch_size"]}')
                            import time
                            time.sleep(2)
                            
                            responses = client.embeddings.create(
                                input=new_batch,
                                model=self.config['sent_emb_model']
                            )
                            for response in responses.data:
                                sent_embs.append(response.embedding)
                        except Exception as retry_e:
                            print(f'Retry also failed: {retry_e}')
                            raise retry_e
                    
                sent_embs = np.array(sent_embs, dtype=np.float32)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        else:
            raise ValueError(f"Unsupported embedding model: {self.config['sent_emb_model']}")
        
        sent_embs.tofile(output_path)
        print(f'[TOKENIZER] Sentence embeddings saved to: {output_path}')
        return sent_embs
    
    def _get_items_for_training(self) -> np.ndarray:
        mask = np.ones(len(self.id_mapping['id2item']) - 1, dtype=bool)
        print(f'[TOKENIZER] Training items count: {mask.sum()} / {len(self.id_mapping["id2item"]) - 1}')
        return mask
    
    def _get_codebook_bits(self, n_codebook: int) -> int:
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)
    
    def generate_embeddings(self):
        if self.config['metadata'] != 'sentence':
            print('[TOKENIZER] Skipping embedding generation, metadata is not in sentence mode')
            return
        
        sem_ids_path = os.path.join(
            self.processed_dir,
            f'{os.path.basename(self.config["sent_emb_model"])}_OPQ{self.config["n_codebook"]},IVF1,PQ{self.config["n_codebook"]}x{self._get_codebook_bits(self.config["codebook_size"])}.sem_ids'
        )
        
        if os.path.exists(sem_ids_path):
            print(f'[TOKENIZER] Semantic IDs already exist: {sem_ids_path}')
            return
        
        sent_emb_path = os.path.join(
            self.processed_dir,
            f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
        )
        
        if os.path.exists(sent_emb_path):
            print(f'[TOKENIZER] Loading sentence embeddings: {sent_emb_path}...')
            sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
        else:
            print('[TOKENIZER] Encoding sentence embeddings...')
            sent_embs = self._encode_sent_emb(sent_emb_path)
        
        if self.config['sent_emb_pca'] > 0:
            print(f'[TOKENIZER] Applying PCA to sentence embeddings...')
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
                
                pca_emb_path = os.path.join(self.processed_dir, 'final_pca_embeddings.npy')
                np.save(pca_emb_path, sent_embs)
                print(f'[TOKENIZER] PCA embeddings saved to: {pca_emb_path}')
            except ImportError:
                raise ImportError("Please install scikit-learn: pip install scikit-learn")
        
        print(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')
    
    def run_full_pipeline(self, joint_categories=None, create_joint_dataset=False):
        print(f"Starting Amazon Reviews 2014 dataset processing - Category: {self.category}")
        
        self._check_available_category()
        
        print("\n=== Step 1: Download raw data ===")
        reviews_path = self._download_raw('reviews')
        meta_path = self._download_raw('meta')
        
        print("\n=== Step 2: Process reviews ===")
        self.all_item_seqs, self.id_mapping = self._process_reviews(reviews_path)
        
        print("\n=== Step 3: Process metadata ===")
        self.item2meta = self._process_meta(meta_path)
        
        if self.item2meta:
            print("\n=== Step 4: Generate embeddings and semantic IDs ===")
            self.generate_embeddings()
        
        if create_joint_dataset and joint_categories:
            print("\n=== Step 5: Create joint dataset ===")
            self._create_joint_dataset_step(joint_categories)
        
        print(f"\n=== Processing completed ===")
        print(f"Data saved in: {self.cache_dir}")
        print(f"Raw data: {self.raw_dir}")
        print(f"Processed data: {self.processed_dir}")
        
        print("\nGenerated files:")
        for root, dirs, files in os.walk(self.cache_dir):
            level = root.replace(self.cache_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    
    def _create_joint_dataset_step(self, joint_categories):
        """Create joint dataset step in pipeline"""
        try:
            from .create_joint_dataset import create_joint_dataset
            
            print(f"[JOINT] Creating joint dataset for categories: {joint_categories}")
            
            base_cache_dir = os.path.dirname(os.path.dirname(self.cache_dir))
            missing_categories = []
            
            for category in joint_categories:
                category_processed_dir = os.path.join(base_cache_dir, 'AmazonReviews2014', category, 'processed')
                required_files = ['all_item_seqs.json', 'id_mapping.json']
                
                for required_file in required_files:
                    file_path = os.path.join(category_processed_dir, required_file)
                    if not os.path.exists(file_path):
                        missing_categories.append(f"{category}/{required_file}")
            
            if missing_categories:
                print(f"[JOINT] Warning: Missing required files for joint dataset creation:")
                for missing in missing_categories:
                    print(f"[JOINT]   - {missing}")
                print(f"[JOINT] Please process all individual categories first.")
                return
            
            output_path = create_joint_dataset(joint_categories, base_cache_dir)
            print(f"[JOINT] Joint dataset created successfully: {output_path}")
            
        except ImportError as e:
            print(f"[JOINT] Error: Cannot import create_joint_dataset module: {e}")
            print(f"[JOINT] Please ensure create_joint_dataset.py is available in the dataloader directory.")
        except Exception as e:
            print(f"[JOINT] Error creating joint dataset: {e}")

    def _load_config(self, config_path: str = None, config_dict: dict = None) -> dict:
        final_config = self.default_config.copy()
        
        category_config_path = os.path.join(
            os.path.dirname(__file__), 
            'AmazonReviews2014', 
            'config.yaml'
        )
        if os.path.exists(category_config_path):
            try:
                with open(category_config_path, 'r', encoding='utf-8') as f:
                    category_config = yaml.safe_load(f)
                    if category_config:
                        print(f"[CONFIG] Loading from category config: {category_config_path}")
                        final_config.update(category_config)
            except Exception as e:
                print(f"[CONFIG] Warning: Cannot load category config: {e}")
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    external_config = yaml.safe_load(f)
                    if external_config:
                        print(f"[CONFIG] Loading from external config: {config_path}")
                        final_config.update(external_config)
            except Exception as e:
                print(f"[CONFIG] Warning: Cannot load external config: {e}")
        
        if config_dict:
            print("[CONFIG] Loading from provided config dict")
            final_config.update(config_dict)
        
        return final_config


def main():
    parser = argparse.ArgumentParser(description='Amazon Reviews 2014 Data Processor')
    parser.add_argument('--category', type=str, required=True, help='Amazon category to process')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--metadata', type=str, choices=['none', 'raw', 'sentence'], default='sentence', help='Metadata processing mode')
    
    parser.add_argument('--joint_categories', type=str, nargs='+', help='Categories for joint dataset creation')
    parser.add_argument('--create_joint', action='store_true', help='Create joint dataset after processing')
    
    args = parser.parse_args()
    
    config_override = {
        'metadata': args.metadata
    }
    
    processor = AmazonDataProcessor(
        category=args.category,
        cache_dir=args.cache_dir,
        config_path=args.config,
        config=config_override
    )
    
    processor.run_full_pipeline(
        joint_categories=args.joint_categories,
        create_joint_dataset=args.create_joint
    )


def generate_unified_embeddings_for_joint_dataset(categories: List[str], base_cache_dir: str = "cache", config: dict = None):
    """Generate unified embeddings based on joint dataset processed files"""
    joint_category = "_".join(sorted(categories))
    joint_processed_dir = os.path.join(base_cache_dir, 'AmazonReviews2014', joint_category, 'processed')
    
    print(f"[UNIFIED_EMB] Generating unified embeddings for joint dataset: {joint_category}")
    
    required_files = [
        os.path.join(joint_processed_dir, 'all_item_seqs.json'),
        os.path.join(joint_processed_dir, 'id_mapping.json'),
        os.path.join(joint_processed_dir, 'metadata.sentence.json')
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Joint dataset file not found: {file_path}")
    
    metadata_path = os.path.join(joint_processed_dir, 'metadata.sentence.json')
    with open(metadata_path, 'r') as f:
        metadata_sentences = json.load(f)
    
    id_mapping_path = os.path.join(joint_processed_dir, 'id_mapping.json')
    with open(id_mapping_path, 'r') as f:
        id_mapping = json.load(f)
    
    print(f"[UNIFIED_EMB] Loaded metadata for {len(metadata_sentences)} items")
    
    sentences_list = []
    valid_indices = []
    
    valid_ids = [int(k) for k in metadata_sentences.keys() if k.isdigit() and int(k) > 0]
    valid_ids.sort()
    
    print(f"[UNIFIED_EMB] Found {len(valid_ids)} valid IDs (excluding pad token)")
    print(f"[UNIFIED_EMB] ID range: {min(valid_ids)} - {max(valid_ids)}")
    
    for item_id in valid_ids:
        item_id_str = str(item_id)
        if item_id_str in metadata_sentences:
            sentence = metadata_sentences[item_id_str]
            if sentence and sentence.strip():
                sentences_list.append(sentence)
                valid_indices.append(item_id)
            else:
                print(f"[UNIFIED_EMB] Warning: Empty sentence for ID {item_id}")
        else:
            print(f"[UNIFIED_EMB] Warning: No metadata for ID {item_id}")
    
    print(f"[UNIFIED_EMB] Final valid sentences count: {len(sentences_list)}")
    
    if not sentences_list:
        raise ValueError("No valid sentences found for embedding generation")
    
    if config is None:
        config = {
            'sent_emb_model': 'text-embedding-3-large',
            'sent_emb_dim': 3072,
            'sent_emb_batch_size': 100,
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'openai_base_url': os.getenv('OPENAI_BASE_URL')
        }
    
    sent_emb_path = os.path.join(joint_processed_dir, f'{os.path.basename(config["sent_emb_model"])}.sent_emb')
    sent_emb_npy_path = sent_emb_path + '.npy'
    
    if os.path.exists(sent_emb_npy_path):
        print(f"[UNIFIED_EMB] Embedding file already exists, skipping generation: {sent_emb_npy_path}")
        return sent_emb_npy_path
    
    if 'text-embedding-3' in config['sent_emb_model']:
        if not config.get('openai_api_key'):
            raise ValueError("OpenAI API key required for OpenAI embeddings")
        
        try:
            from openai import OpenAI
            from tqdm import tqdm
            
            client_kwargs = {'api_key': config['openai_api_key']}
            if config.get('openai_base_url'):
                client_kwargs['base_url'] = config['openai_base_url']
            
            client = OpenAI(**client_kwargs)
            
            sent_embs = []
            batch_size = config.get('sent_emb_batch_size', 100)
            
            for i in tqdm(range(0, len(sentences_list), batch_size), desc='Generating unified embeddings'):
                batch = sentences_list[i:i + batch_size]
                valid_batch = [sent for sent in batch if sent and sent.strip()]
                
                if not valid_batch:
                    print(f"[UNIFIED_EMB] Warning: Empty batch at index {i}, skipping")
                    continue
                
                try:
                    responses = client.embeddings.create(
                        input=valid_batch,
                        model=config['sent_emb_model']
                    )
                    for response in responses.data:
                        sent_embs.append(response.embedding)
                except Exception as e:
                    print(f'Embedding generation failed {i} - {i + batch_size}: {e}')
                    try:
                        new_batch = []
                        for sent in valid_batch:
                            if len(sent) > 8000:
                                new_batch.append(sent[:8000])
                            else:
                                new_batch.append(sent)
                        
                        import time
                        time.sleep(2)
                        responses = client.embeddings.create(
                            input=new_batch,
                            model=config['sent_emb_model']
                        )
                        for response in responses.data:
                            sent_embs.append(response.embedding)
                    except Exception as retry_e:
                        print(f'Retry also failed: {retry_e}')
                        raise retry_e
            
            sent_embs = np.array(sent_embs, dtype=np.float32)
            
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    else:
        raise ValueError(f"Unsupported embedding model: {config['sent_emb_model']}")
    
    print(f"[UNIFIED_EMB] Unified embedding generation completed")
    print(f"[UNIFIED_EMB] Generated embeddings shape: {sent_embs.shape}")
    
    np.save(sent_emb_npy_path, sent_embs)
    print(f"[UNIFIED_EMB] Unified embeddings saved to: {sent_emb_npy_path}")
    
    index_mapping = {str(valid_indices[i]): i for i in range(len(valid_indices))}
    index_mapping_path = os.path.join(joint_processed_dir, 'embedding_index_mapping.json')
    with open(index_mapping_path, 'w') as f:
        json.dump(index_mapping, f, indent=2)
    print(f"[UNIFIED_EMB] Index mapping saved to: {index_mapping_path}")
    
    if os.path.exists(sent_emb_npy_path):
        file_size = os.path.getsize(sent_emb_npy_path)
        print(f"[UNIFIED_EMB] File save verification successful, size: {file_size} bytes")
    else:
        raise FileNotFoundError(f"Embedding file save failed: {sent_emb_npy_path}")
    
    return sent_emb_npy_path


if __name__ == '__main__':
    main()