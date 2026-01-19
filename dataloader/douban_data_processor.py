# douban_data_processor.py

import os
import json
import math
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
import yaml
import pandas as pd
from datetime import datetime


class DoubanDataProcessor:
    def __init__(self, category: str, cache_dir: str = "cache", config_path: str = None, config: dict = None):
        self.category = category
        self.cache_dir = cache_dir
        self.raw_dir = os.path.join(cache_dir, 'DoubanDataset', category, 'raw')
        self.processed_dir = os.path.join(cache_dir, 'DoubanDataset', category, 'processed')
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.all_item_seqs = {}
        self.id_mapping = {
            'user2id': {},
            'id2user': [],
            'item2id': {},
            'id2item': []
        }
        self.item2meta = {}
        
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
            'min_interactions': 5  
        }
        
        self.config = self._load_config(config_path, config)
    
    def _check_available_category(self):
        available_categories = ['books', 'movies', 'music']
        assert self.category in available_categories, f'Category "{self.category}" not available. Available categories: {available_categories}'
    
    def _get_file_paths(self):
        base_path = "cache/douban_dataset(text information)"
        
        if self.category == 'books':
            reviews_file = os.path.join(base_path, 'bookreviews_cleaned.txt')
            items_file = os.path.join(base_path, 'books_cleaned.txt')
            users_file = os.path.join(base_path, 'users_cleaned.txt')
        elif self.category == 'movies':
            reviews_file = os.path.join(base_path, 'moviereviews_cleaned.txt')
            items_file = os.path.join(base_path, 'movies_cleaned.txt')
            users_file = os.path.join(base_path, 'users_cleaned.txt')
        elif self.category == 'music':
            reviews_file = os.path.join(base_path, 'musicreviews_cleaned.txt')
            items_file = os.path.join(base_path, 'music_cleaned.txt')
            users_file = os.path.join(base_path, 'users_cleaned.txt')
        else:
            raise ValueError(f"Unsupported category: {self.category}")
        
        return reviews_file, items_file, users_file
    
    def _load_reviews(self, reviews_file: str) -> List[Tuple]:
        print('[DATASET] Loading reviews...')
        reviews = []
        
        df = pd.read_csv(reviews_file, sep='\t', quotechar='"')
        
        if self.category == 'books':
            user_col, item_col, rating_col, comment_col, time_col = 'user_id', 'book_id', 'rating', 'comment', 'time'
        elif self.category == 'movies':
            user_col, item_col, rating_col, comment_col, time_col = 'user_id', 'movie_id', 'rating', 'comment', 'time'
        elif self.category == 'music':
            user_col, item_col, rating_col, comment_col, time_col = 'user_id', 'music_id', 'rating', 'comment', 'time'
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
            user = str(row[user_col])
            item = str(row[item_col])
            
            try:
                time_str = str(row[time_col])
                if time_str and time_str != 'nan':
                    time_obj = datetime.strptime(time_str, '%Y-%m-%d')
                    time = int(time_obj.timestamp())
                else:
                    time = 0
            except:
                time = 0
            
            reviews.append((user, item, time))
        
        print(f'[DATASET] Loaded {len(reviews)} reviews')
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
    
    def _filter_by_min_interactions(self, all_item_seqs: Dict, min_interactions: int = 5) -> Dict:
        """Filter out users with less than min_interactions"""
        print(f'[DATASET] Filtering users with less than {min_interactions} interactions...')
        original_count = len(all_item_seqs)
        filtered_seqs = {}
        
        for user_id, item_seq in all_item_seqs.items():
            if len(item_seq) >= min_interactions:
                filtered_seqs[user_id] = item_seq
        
        filtered_count = len(filtered_seqs)
        removed_count = original_count - filtered_count
        
        print(f'[DATASET] Original users: {original_count}')
        print(f'[DATASET] Filtered users: {filtered_count}')
        print(f'[DATASET] Removed users: {removed_count} ({removed_count/original_count*100:.2f}%)')
        
        return filtered_seqs

    def _update_id_mapping_after_filter(self, id_mapping: Dict, filtered_seqs: Dict) -> Dict:
        """Update id_mapping to only include used users and items after filtering"""
        print('[DATASET] Updating ID mapping after filtering...')
        
        used_users = set(filtered_seqs.keys())
        used_items = set()
        for item_seq in filtered_seqs.values():
            used_items.update(item_seq)
        
        # Update user mapping
        new_user2id = {}
        new_id2user = {}
        for user_id in used_users:
            if user_id in id_mapping['user2id']:
                new_id = id_mapping['user2id'][user_id]
                new_user2id[user_id] = new_id
                new_id2user[new_id] = user_id
        
        # Update item mapping
        new_item2id = {}
        new_id2item = {}
        for item_id in used_items:
            if item_id in id_mapping['item2id']:
                new_id = id_mapping['item2id'][item_id]
                new_item2id[item_id] = new_id
                new_id2item[new_id] = item_id
        
        updated_mapping = {
            'user2id': new_user2id,
            'id2user': new_id2user,
            'item2id': new_item2id,
            'id2item': new_id2item
        }
        
        print(f'[DATASET] Updated user mapping: {len(new_user2id)} users')
        print(f'[DATASET] Updated item mapping: {len(new_item2id)} items')
        
        return updated_mapping

    def _process_reviews(self, reviews_file: str) -> Tuple[Dict, Dict]:
        seq_file = os.path.join(self.processed_dir, 'all_item_seqs.json')
        id_mapping_file = os.path.join(self.processed_dir, 'id_mapping.json')
        
        filtered_seq_file = os.path.join(self.processed_dir, 'all_item_seqs_filtered.json')
        filtered_id_mapping_file = os.path.join(self.processed_dir, 'id_mapping_filtered.json')
        
        if os.path.exists(filtered_seq_file) and os.path.exists(filtered_id_mapping_file):
            print('[DATASET] Using filtered data...')
            with open(filtered_seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(filtered_id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_item_seqs, id_mapping
        elif os.path.exists(seq_file) and os.path.exists(id_mapping_file):
            print('[DATASET] Reviews have been processed, applying 5-core filtering...')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            
            min_interactions = self.config.get('min_interactions', 5)
            filtered_seqs = self._filter_by_min_interactions(all_item_seqs, min_interactions)
            updated_mapping = self._update_id_mapping_after_filter(id_mapping, filtered_seqs)
            
            with open(filtered_seq_file, 'w') as f:
                json.dump(filtered_seqs, f, indent=2, ensure_ascii=False)
            with open(filtered_id_mapping_file, 'w') as f:
                json.dump(updated_mapping, f, indent=2, ensure_ascii=False)
            
            print('[DATASET] Filtered data saved')
            return filtered_seqs, updated_mapping
        else:
            print('[DATASET] Processing reviews...')
            reviews = self._load_reviews(reviews_file)
            item_seqs = self._get_item_seqs(reviews)
            all_item_seqs, id_mapping = self._remap_ids(item_seqs)
            
            print('[DATASET] Saving original mapping data...')
            with open(seq_file, 'w') as f:
                json.dump(all_item_seqs, f)
            with open(id_mapping_file, 'w') as f:
                json.dump(id_mapping, f)
            
            min_interactions = self.config.get('min_interactions', 5)
            filtered_seqs = self._filter_by_min_interactions(all_item_seqs, min_interactions)
            updated_mapping = self._update_id_mapping_after_filter(id_mapping, filtered_seqs)
            
            with open(filtered_seq_file, 'w') as f:
                json.dump(filtered_seqs, f, indent=2, ensure_ascii=False)
            with open(filtered_id_mapping_file, 'w') as f:
                json.dump(updated_mapping, f, indent=2, ensure_ascii=False)
            
            print('[DATASET] Original and filtered data saved')
            return filtered_seqs, updated_mapping
    
    def _load_metadata(self, items_file: str, reviews_file: str) -> Dict:
        print('[DATASET] Loading metadata...')
        
        if self.category == 'movies':
            return self._load_movies_metadata(items_file, reviews_file)
        else:
            return self._load_reviews_metadata(reviews_file)
    
    def _load_movies_metadata(self, items_file: str, reviews_file: str) -> Dict:
        print('[DATASET] Loading movies metadata from items file...')
        
        movies_df = pd.read_csv(items_file, sep='\t', quotechar='"')
        
        reviews_df = pd.read_csv(reviews_file, sep='\t', quotechar='"')
        
        movie_labels = defaultdict(set)
        for _, row in reviews_df.iterrows():
            movie_id = str(row['movie_id'])
            labels = str(row['labels']) if pd.notna(row['labels']) else ""
            
            if labels and labels != 'nan':
                label_list = labels.split('|')
                movie_labels[movie_id].update(label_list)
        
        metadata = {}
        
        for _, row in movies_df.iterrows():
            movie_id = str(row['movie_id'])
            
            name = str(row['name']) if pd.notna(row['name']) else ""
            director = str(row['director']) if pd.notna(row['director']) else ""
            summary = str(row['summary']) if pd.notna(row['summary']) else ""
            writer = str(row['writer']) if pd.notna(row['writer']) else ""
            country = str(row['country']) if pd.notna(row['country']) else ""
            pubdate = str(row['pubdate']) if pd.notna(row['pubdate']) else ""
            language = str(row['language']) if pd.notna(row['language']) else ""
            rating = str(row['rating']) if pd.notna(row['rating']) else ""
            
            labels = sorted(list(movie_labels[movie_id])) if movie_id in movie_labels else []
            
            meta_info = {
                'name': name,
                'director': director,
                'summary': summary,
                'writer': writer,
                'country': country,
                'pubdate': pubdate,
                'language': language,
                'rating': rating,
                'labels': labels
            }
            
            metadata[movie_id] = meta_info
        
        return metadata
    
    def _load_reviews_metadata(self, reviews_file: str) -> Dict:
        print('[DATASET] Loading reviews metadata...')
        
        reviews_df = pd.read_csv(reviews_file, sep='\t', quotechar='"')
        
        if self.category == 'books':
            review_item_col, labels_col = 'book_id', 'labels'
        elif self.category == 'music':
            review_item_col, labels_col = 'music_id', 'labels'
        else:
            raise ValueError(f"Unsupported category for reviews metadata: {self.category}")
        
        item_labels = defaultdict(set)
        
        for _, row in reviews_df.iterrows():
            item_id = str(row[review_item_col])
            labels = str(row[labels_col]) if pd.notna(row[labels_col]) else ""
            
            if labels and labels != 'nan':  
                label_list = labels.split('|')
                item_labels[item_id].update(label_list)
        
        metadata = {}
        
        for item_id, labels_set in item_labels.items():
            labels_list = sorted(list(labels_set))
            metadata[item_id] = labels_list
        
        all_items_in_seqs = set()
        if hasattr(self, 'all_item_seqs'):
            for user_id, item_seq in self.all_item_seqs.items():
                for item_id in item_seq:
                    all_items_in_seqs.add(item_id)
        
        missing_items = all_items_in_seqs - set(metadata.keys())
        if missing_items:
            print(f'[DATASET] Found {len(missing_items)} items without labels, adding default metadata')
            for item_id in missing_items:
                metadata[item_id] = ['未知分类']
        
        return metadata
    
    def _load_movies_metadata_with_seqs(self, items_file: str, reviews_file: str, all_items_in_seqs: set) -> Dict:
        print('[DATASET] Loading movies metadata with sequence coverage...')
        
        movies_df = pd.read_csv(items_file, sep='\t', quotechar='"')
        
        reviews_df = pd.read_csv(reviews_file, sep='\t', quotechar='"')
        
        movie_labels = defaultdict(set)
        for _, row in reviews_df.iterrows():
            movie_id = str(row['movie_id'])
            labels = str(row['labels']) if pd.notna(row['labels']) else ""
            
            if labels and labels != 'nan':
                label_list = labels.split('|')
                movie_labels[movie_id].update(label_list)
        
        metadata = {}
        
        for _, row in movies_df.iterrows():
            movie_id = str(row['movie_id'])
            
            name = str(row['name']) if pd.notna(row['name']) else ""
            director = str(row['director']) if pd.notna(row['director']) else ""
            summary = str(row['summary']) if pd.notna(row['summary']) else ""
            writer = str(row['writer']) if pd.notna(row['writer']) else ""
            country = str(row['country']) if pd.notna(row['country']) else ""
            pubdate = str(row['pubdate']) if pd.notna(row['pubdate']) else ""
            language = str(row['language']) if pd.notna(row['language']) else ""
            rating = str(row['rating']) if pd.notna(row['rating']) else ""
            
            labels = sorted(list(movie_labels[movie_id])) if movie_id in movie_labels else []
            
            meta_info = {
                'name': name,
                'director': director,
                'summary': summary,
                'writer': writer,
                'country': country,
                'pubdate': pubdate,
                'language': language,
                'rating': rating,
                'labels': labels
            }
            
            metadata[movie_id] = meta_info
        
        missing_items = all_items_in_seqs - set(metadata.keys())
        if missing_items:
            print(f'[DATASET] Found {len(missing_items)} movies without metadata, adding default metadata')
            for item_id in missing_items:
                metadata[item_id] = {
                    'name': f'未知电影_{item_id}',
                    'director': '',
                    'summary': '',
                    'writer': '',
                    'country': '',
                    'pubdate': '',
                    'language': '',
                    'rating': '',
                    'labels': ['未知分类']
                }
        
        return metadata
    
    def _load_reviews_metadata_with_seqs(self, reviews_file: str, all_items_in_seqs: set) -> Dict:
        print('[DATASET] Loading reviews metadata with sequence coverage...')
        
        reviews_df = pd.read_csv(reviews_file, sep='\t', quotechar='"')

        if self.category == 'books':
            review_item_col, labels_col = 'book_id', 'labels'
        elif self.category == 'music':
            review_item_col, labels_col = 'music_id', 'labels'
        else:
            raise ValueError(f"Unsupported category for reviews metadata: {self.category}")
        
        item_labels = defaultdict(set)
        
        for _, row in reviews_df.iterrows():
            item_id = str(row[review_item_col])
            labels = str(row[labels_col]) if pd.notna(row[labels_col]) else ""
            
            if labels and labels != 'nan':
                label_list = labels.split('|')
                item_labels[item_id].update(label_list)
        
        metadata = {}
        
        for item_id, labels_set in item_labels.items():
            labels_list = sorted(list(labels_set))
            metadata[item_id] = labels_list
        
        missing_items = all_items_in_seqs - set(metadata.keys())
        if missing_items:
            print(f'[DATASET] Found {len(missing_items)} items without labels, adding default metadata')
            for item_id in missing_items:
                metadata[item_id] = ['未知分类']
        
        for item_id, meta in metadata.items():
            if not meta or (isinstance(meta, list) and len(meta) == 0):
                metadata[item_id] = ['未知分类']
        
        return metadata
    
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
            if self.category == 'movies':
                meta_sentence = self._build_movie_sentence(meta)
            else:
                if isinstance(meta, list):
                    if meta and len(meta) > 0:
                        meta_sentence = ', '.join(meta)
                    else:
                        meta_sentence = "未知书籍，暂无详细信息"
                else:
                    meta_sentence = "未知书籍，暂无详细信息"
            
            item2meta[item] = meta_sentence.strip()
        
        empty_count = 0
        for item, sentence in item2meta.items():
            if not sentence or sentence.strip() == "":
                item2meta[item] = "未知书籍，暂无详细信息"
                empty_count += 1
        
        if empty_count > 0:
            print(f'[DATASET] 修复了 {empty_count} 个空的sentence')
        
        return item2meta
    
    def _build_movie_sentence(self, meta: Dict) -> str:
        parts = []
        
        if meta.get('name'):
            parts.append(f"电影: {meta['name']}")
        
        if meta.get('director'):
            director = meta['director'].replace('|', ', ').strip(', ')
            if director:
                parts.append(f"导演: {director}")
        
        if meta.get('writer'):
            writer = meta['writer'].replace('|', ', ').strip(', ')
            if writer:
                parts.append(f"编剧: {writer}")
        
        if meta.get('country'):
            parts.append(f"国家: {meta['country']}")
        
        if meta.get('language'):
            parts.append(f"语言: {meta['language']}")
        
        if meta.get('pubdate'):
            parts.append(f"上映: {meta['pubdate']}")
        
        if meta.get('rating'):
            parts.append(f"评分: {meta['rating']}")
        
        if meta.get('labels'):
            labels_text = ', '.join(meta['labels'])
            if labels_text:
                parts.append(f"标签: {labels_text}")
        
        if meta.get('summary'):
            summary = meta['summary'][:100] + "..." if len(meta['summary']) > 100 else meta['summary']
            parts.append(f"简介: {summary}")
        
        return '. '.join(parts)
    
    def _process_meta_with_seqs(self, items_file: str, reviews_file: str) -> Optional[Dict]:
        """Process metadata only for items that appear in sequences"""
        print('[DATASET] Processing metadata for items in sequences...')
        
        filtered_id_mapping_file = os.path.join(self.processed_dir, 'id_mapping_filtered.json')
        if os.path.exists(filtered_id_mapping_file):
            with open(filtered_id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
        else:
            id_mapping_file = os.path.join(self.processed_dir, 'id_mapping.json')
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
        
        all_items_in_seqs = set()
        filtered_seq_file = os.path.join(self.processed_dir, 'all_item_seqs_filtered.json')
        if os.path.exists(filtered_seq_file):
            with open(filtered_seq_file, 'r') as f:
                all_item_seqs = json.load(f)
        else:
            seq_file = os.path.join(self.processed_dir, 'all_item_seqs.json')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
        
        for item_seq in all_item_seqs.values():
            all_items_in_seqs.update(item_seq)
        
        print(f'[DATASET] Processing metadata for {len(all_items_in_seqs)} items')
        
        if self.category == 'movies':
            return self._load_movies_metadata_with_seqs(items_file, reviews_file, all_items_in_seqs)
        else:
            return self._load_reviews_metadata_with_seqs(reviews_file, all_items_in_seqs)
    
    def _process_meta(self, items_file: str, reviews_file: str) -> Optional[Dict]:
        process_mode = self.config['metadata']
        meta_file = os.path.join(self.processed_dir, f'metadata.{process_mode}.json')
        
        if os.path.exists(meta_file):
            print('[DATASET] Metadata has been processed...')
            with open(meta_file, 'r') as f:
                return json.load(f)
        
        print(f'[DATASET] Processing metadata, mode: {process_mode}')
        
        if process_mode == 'none':
            return None
        
        item2meta = self._load_metadata(items_file=items_file, reviews_file=reviews_file)
        
        if process_mode == 'sentence':
            item2meta = self._extract_meta_sentences(metadata=item2meta)
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(item2meta, f, ensure_ascii=False, indent=2)
        
        return item2meta
    
    def _encode_sent_emb(self, output_path: str) -> np.ndarray:
        print('[TOKENIZER] Encoding sentence embeddings...')
        
        meta_sentences = []
        for i in range(1, len(self.id_mapping['id2item'])):
            item = self.id_mapping['id2item'][i]
            if item in self.item2meta:
                meta_sentences.append(self.item2meta[item])
            else:
                meta_sentences.append("")
        
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
    
    def _get_codebook_bits(self, n_codebook: int) -> int:
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)
    
    def run_full_pipeline(self):
        print(f"Starting Douban dataset processing - Category: {self.category}")
        
        self._check_available_category()
        
        print("\n=== Step 1: Get file paths ===")
        reviews_file, items_file, users_file = self._get_file_paths()
        
        print(f"Reviews file: {reviews_file}")
        print(f"Items file: {items_file}")
        print(f"Users file: {users_file}")
        
        print("\n=== Step 2: Process reviews ===")
        self.all_item_seqs, self.id_mapping = self._process_reviews(reviews_file)
        
        print("\n=== Step 3: Process metadata ===")
        self.item2meta = self._process_meta_with_seqs(items_file, reviews_file)
        
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
        
        if self.item2meta and self.config.get('openai_api_key'):
            print("\n=== Step 4: Generate embeddings and semantic IDs ===")
            self.generate_embeddings()
        elif self.item2meta and not self.config.get('openai_api_key'):
            print("\n=== Step 4: Skipping embeddings (no API key) ===")
            print("To generate embeddings, set OPENAI_API_KEY environment variable or provide it in config")
    
    def _load_config(self, config_path: str = None, config_dict: dict = None) -> dict:
        final_config = self.default_config.copy()
        
        category_config_path = os.path.join(
            os.path.dirname(__file__), 
            'DoubanDataset', 
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
    parser = argparse.ArgumentParser(description='Douban Dataset Processor')
    parser.add_argument('--category', type=str, required=True, choices=['books', 'movies', 'music'], help='Douban category to process')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--metadata', type=str, choices=['none', 'raw', 'sentence'], default='sentence', help='Metadata processing mode')
    
    args = parser.parse_args()
    
    config_override = {
        'metadata': args.metadata
    }
    
    processor = DoubanDataProcessor(
        category=args.category,
        cache_dir=args.cache_dir,
        config_path=args.config,
        config=config_override
    )
    
    processor.run_full_pipeline()


if __name__ == '__main__':
    main()