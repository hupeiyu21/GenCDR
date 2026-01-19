# douban_joint_dataset.py

import os
import json
from typing import Dict, List, Set

def create_douban_cross_domain_mapping(categories: List[str], base_cache_dir: str = "cache"):
    """Create cross-domain mapping table for Douban datasets"""
    joint_category = "_".join(sorted(categories))
    joint_dir = os.path.join(base_cache_dir, 'DoubanDataset', joint_category)
    os.makedirs(joint_dir, exist_ok=True)
    
    dataset_info = {}
    global_item_offset = 1
    
    for category in categories:
        print(f"[INFO] Analyzing Douban category: {category}")
        
        category_dir = os.path.join(base_cache_dir, 'DoubanDataset', category)
        
        id_mapping_path = os.path.join(category_dir, 'processed', 'id_mapping_filtered.json')
        if not os.path.exists(id_mapping_path):
            id_mapping_path = os.path.join(category_dir, 'processed', 'id_mapping.json')
        
        with open(id_mapping_path, 'r') as f:
            category_id_mapping = json.load(f)
        
        local_ids = [local_id for local_id in category_id_mapping.get('item2id', {}).values() if local_id > 0]
        if local_ids:
            max_local_id = max(local_ids)
            local_item_count = max_local_id
        else:
            local_item_count = 0
        
        dataset_info[category] = {
            'start_idx': global_item_offset,
            'end_idx': global_item_offset + local_item_count - 1,
            'local_item_count': local_item_count,
            'id_mapping': category_id_mapping
        }
        
        global_item_offset += local_item_count
        
        print(f"  - Items: {local_item_count}")
        print(f"  - Global ID range: {dataset_info[category]['start_idx']} - {dataset_info[category]['end_idx']}")
    
    cross_domain_mapping = {
        'dataset_info': dataset_info,
        'total_items': global_item_offset - 1,
        'datasets': categories
    }
    
    cross_domain_mapping_path = os.path.join(joint_dir, 'cross_domain_mapping.json')
    with open(cross_domain_mapping_path, 'w') as f:
        json.dump(cross_domain_mapping, f, indent=2)
    
    print(f"[INFO] Douban cross-domain mapping created: {cross_domain_mapping_path}")
    
    return cross_domain_mapping

def merge_douban_all_item_seqs(categories: List[str], base_cache_dir: str, dataset_info: Dict) -> Dict:
    """Merge all_item_seqs.json files from all Douban categories"""
    merged_all_item_seqs = {}
    
    for category in categories:
        print(f"[JOINT] Processing Douban category: {category}")
        
        category_dir = os.path.join(base_cache_dir, 'DoubanDataset', category)
        seqs_path = os.path.join(category_dir, 'processed', 'all_item_seqs.json')
        
        if not os.path.exists(seqs_path):
            print(f"[ERROR] File not found: {seqs_path}")
            continue
            
        with open(seqs_path, 'r') as f:
            category_seqs = json.load(f)
        
        id_mapping_path = os.path.join(category_dir, 'processed', 'id_mapping.json')
        with open(id_mapping_path, 'r') as f:
            category_id_mapping = json.load(f)
        
        category_info = dataset_info[category]
        category_user2id = category_info['id_mapping']['user2id']
        
        for user_id, item_seq in category_seqs.items():
            if user_id in category_user2id:
                unique_key = f"{category}_{user_id}"
                merged_all_item_seqs[unique_key] = item_seq
        
        print(f"[JOINT] {category} processed, users: {len([u for u in category_seqs.keys() if u in category_user2id])}")
    
    return merged_all_item_seqs

def create_douban_joint_id_mapping_data(categories: List[str], base_cache_dir: str, cross_domain_mapping: Dict) -> Dict:
    """Create joint dataset id_mapping.json data for Douban"""
    dataset_info = cross_domain_mapping['dataset_info']
    
    global_item2id = {"<pad>": 0}
    global_id2item = {"0": "<pad>"}
    
    all_item_ids = set()
    category_items = {}
    
    for category in categories:
        category_dir = os.path.join(base_cache_dir, 'DoubanDataset', category)
        id_mapping_path = os.path.join(category_dir, 'processed', 'id_mapping.json')
        
        with open(id_mapping_path, 'r') as f:
            category_id_mapping = json.load(f)
        
        category_items[category] = set(category_id_mapping['item2id'].keys())
        all_item_ids.update(category_items[category])
    
    item_to_categories = {}
    for item_id in all_item_ids:
        item_to_categories[item_id] = []
        for category in categories:
            if item_id in category_items[category]:
                item_to_categories[item_id].append(category)
    
    duplicate_items = {item_id: cats for item_id, cats in item_to_categories.items() if len(cats) > 1}
    if duplicate_items:
        print(f"[JOINT] Found {len(duplicate_items)} duplicate items across categories")
    
    global_id_counter = 1
    
    for category in categories:
        category_dir = os.path.join(base_cache_dir, 'DoubanDataset', category)
        id_mapping_path = os.path.join(category_dir, 'processed', 'id_mapping.json')
        
        with open(id_mapping_path, 'r') as f:
            category_id_mapping = json.load(f)
        
        for item_id, local_id in category_id_mapping['item2id'].items():
            if local_id > 0:
                if item_id not in global_item2id:
                    global_item2id[item_id] = global_id_counter
                    global_id2item[str(global_id_counter)] = item_id
                    global_id_counter += 1
    
    global_user2id = {}
    global_id2user = {}
    global_user_counter = 1
    
    for category in categories:
        category_info = dataset_info[category]
        category_user2id = category_info['id_mapping']['user2id']
        
        for user_id, local_user_idx in category_user2id.items():
            global_user_key = f"{category}_{user_id}"
            if global_user_key not in global_user2id:
                global_user2id[global_user_key] = global_user_counter
                global_id2user[str(global_user_counter)] = global_user_key
                global_user_counter += 1
    
    print(f"[JOINT] Fixed ID mapping: {len(global_item2id) - 1} items (excluding pad)")
    print(f"[JOINT] ID range: 1 - {global_id_counter - 1}")
    
    return {
        'user2id': global_user2id,
        'id2user': global_id2user,
        'item2id': global_item2id,
        'id2item': global_id2item
    }

def merge_douban_metadata_sentences(categories: List[str], base_cache_dir: str, joint_id_mapping: Dict) -> Dict:
    """Merge metadata.sentence.json files from all Douban categories"""
    merged_metadata = {}
    
    processed_items = set()
    
    for category in categories:
        print(f"[JOINT] Merging {category} metadata")
        
        category_dir = os.path.join(base_cache_dir, 'DoubanDataset', category)
        metadata_path = os.path.join(category_dir, 'processed', 'metadata.sentence.json')
        
        if not os.path.exists(metadata_path):
            print(f"[WARNING] Metadata file not found: {metadata_path}")
            continue
            
        with open(metadata_path, 'r') as f:
            category_metadata = json.load(f)
        
        merged_count = 0
        skipped_count = 0
        
        for item_id, sentence in category_metadata.items():
            if item_id in joint_id_mapping['item2id']:
                global_id = joint_id_mapping['item2id'][item_id]
                global_id_str = str(global_id)
                
                if item_id not in processed_items:
                    merged_metadata[global_id_str] = sentence
                    processed_items.add(item_id)
                    merged_count += 1
                else:
                    skipped_count += 1
        
        print(f"[JOINT] {category} metadata merged, entries: {merged_count}/{len(category_metadata)} (skipped: {skipped_count})")
    
    import random
    pad_sentences = [
        "This is a placeholder item for padding purposes in the recommendation system."
    ]
    merged_metadata["0"] = random.choice(pad_sentences)
    print(f"[JOINT] Added pad metadata entry for ID 0")
    
    return merged_metadata

def create_douban_joint_dataset(categories: List[str], base_cache_dir: str = "cache"):
    """Create joint dataset for Douban with three core files: all_item_seqs.json, id_mapping.json, metadata.sentence.json"""
    joint_category = "_".join(sorted(categories))
    joint_dir = os.path.join(base_cache_dir, 'DoubanDataset', joint_category)
    joint_processed_dir = os.path.join(joint_dir, 'processed')
    
    os.makedirs(joint_processed_dir, exist_ok=True)
    os.makedirs(joint_dir, exist_ok=True)
    
    print(f"[JOINT] Creating Douban joint dataset: {joint_category}")
    print(f"[JOINT] Output directory: {joint_processed_dir}")
    
    cross_domain_mapping_path = os.path.join(joint_dir, 'cross_domain_mapping.json')
    
    if not os.path.exists(cross_domain_mapping_path):
        print(f"[JOINT] Creating cross-domain mapping: {cross_domain_mapping_path}")
        cross_domain_mapping = create_douban_cross_domain_mapping(categories, base_cache_dir)
    else:
        print(f"[JOINT] Loading cross-domain mapping: {cross_domain_mapping_path}")
        with open(cross_domain_mapping_path, 'r') as f:
            cross_domain_mapping = json.load(f)
    
    dataset_info = cross_domain_mapping['dataset_info']
    
    print(f"[JOINT] Step 1: Merging all_item_seqs.json")
    merged_all_item_seqs = merge_douban_all_item_seqs(categories, base_cache_dir, dataset_info)
    
    all_item_seqs_path = os.path.join(joint_processed_dir, 'all_item_seqs.json')
    with open(all_item_seqs_path, 'w') as f:
        json.dump(merged_all_item_seqs, f, indent=2)
    print(f"[JOINT] all_item_seqs.json saved: {all_item_seqs_path}")
    print(f"[JOINT] Total user sequences: {len(merged_all_item_seqs)}")
    
    print(f"[JOINT] Step 2: Creating joint id_mapping.json")
    joint_id_mapping = create_douban_joint_id_mapping_data(categories, base_cache_dir, cross_domain_mapping)
    
    id_mapping_path = os.path.join(joint_processed_dir, 'id_mapping.json')
    with open(id_mapping_path, 'w') as f:
        json.dump(joint_id_mapping, f, indent=2)
    print(f"[JOINT] id_mapping.json saved: {id_mapping_path}")
    print(f"[JOINT] Global users: {len(joint_id_mapping['user2id'])}")
    print(f"[JOINT] Global items: {len(joint_id_mapping['item2id']) - 1}")
    
    print(f"[JOINT] Step 3: Merging metadata.sentence.json")
    merged_metadata = merge_douban_metadata_sentences(categories, base_cache_dir, joint_id_mapping)
    
    metadata_path = os.path.join(joint_processed_dir, 'metadata.sentence.json')
    with open(metadata_path, 'w') as f:
        json.dump(merged_metadata, f, indent=2)
    print(f"[JOINT] metadata.sentence.json saved: {metadata_path}")
    print(f"[JOINT] Metadata entries: {len(merged_metadata)}")
    
    print(f"[JOINT] Douban joint dataset core files created successfully!")
    print(f"[JOINT] Ready for unified embedding generation")
    
    return all_item_seqs_path

def generate_douban_unified_embeddings(categories: List[str], base_cache_dir: str = "cache", config: dict = None):
    """Generate unified embeddings for Douban joint dataset"""
    print(f"[JOINT] Generating unified embeddings for Douban datasets: {categories}")
    
    joint_category = "_".join(sorted(categories))
    joint_dir = os.path.join(base_cache_dir, 'DoubanDataset', joint_category)
    joint_processed_dir = os.path.join(joint_dir, 'processed')
    
    # Load joint dataset files
    id_mapping_path = os.path.join(joint_processed_dir, 'id_mapping.json')
    metadata_path = os.path.join(joint_processed_dir, 'metadata.sentence.json')
    
    with open(id_mapping_path, 'r') as f:
        id_mapping = json.load(f)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Prepare sentences for embedding
    meta_sentences = []
    for i in range(1, len(id_mapping['id2item'])):
        item_id = id_mapping['id2item'][str(i)]
        if str(i) in metadata:
            meta_sentences.append(metadata[str(i)])
        else:
            meta_sentences.append("Unknown item, no details available")
    
    # Generate embeddings
    sent_emb_model = config.get('sent_emb_model', 'text-embedding-3-large')
    sent_emb_batch_size = config.get('sent_emb_batch_size', 100)
    
    if 'text-embedding-3' in sent_emb_model:
        if not config.get('openai_api_key'):
            raise ValueError("OpenAI API key required for OpenAI embeddings")
        
        try:
            from openai import OpenAI
            from tqdm import tqdm
            import numpy as np
            
            client_kwargs = {'api_key': config['openai_api_key']}
            if 'openai_base_url' in config and config['openai_base_url']:
                client_kwargs['base_url'] = config['openai_base_url']
            
            client = OpenAI(**client_kwargs)
            
            sent_embs = []
            for i in tqdm(range(0, len(meta_sentences), sent_emb_batch_size), desc='Encoding'):
                batch = meta_sentences[i:i + sent_emb_batch_size]
                try:
                    responses = client.embeddings.create(
                        input=batch,
                        model=sent_emb_model
                    )
                    for response in responses.data:
                        sent_embs.append(response.embedding)
                except Exception as e:
                    print(f'Encoding failed {i} - {i + sent_emb_batch_size}: {e}')
                    raise e
                    
            sent_embs = np.array(sent_embs, dtype=np.float32)
            
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    else:
        raise ValueError(f"Unsupported embedding model: {sent_emb_model}")
    
    # Save embeddings
    emb_path = os.path.join(joint_processed_dir, f'{os.path.basename(sent_emb_model)}.sent_emb.npy')
    np.save(emb_path, sent_embs)
    print(f"[JOINT] Unified embeddings saved to: {emb_path}")
    print(f"[JOINT] Embedding shape: {sent_embs.shape}")
    
    return emb_path

def create_douban_joint_dataset_with_unified_embeddings(categories: List[str], base_cache_dir: str = "cache", config: dict = None):
    """Create Douban joint dataset with unified embeddings"""
    print(f"[JOINT] Creating Douban joint dataset with unified embeddings: {categories}")
    
    print(f"[JOINT] Step 1: Creating joint dataset core files")
    joint_dataset_path = create_douban_joint_dataset(categories, base_cache_dir)
    
    print(f"[JOINT] Step 2: Generating unified embeddings")
    emb_path = generate_douban_unified_embeddings(categories, base_cache_dir, config)
    
    print(f"[JOINT] Douban joint dataset with unified embeddings created successfully!")
    print(f"[JOINT] - Joint dataset files: {joint_dataset_path}")
    print(f"[JOINT] - Unified embedding file: {emb_path}")
    return joint_dataset_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Douban joint dataset')
    parser.add_argument('--categories', type=str, nargs='+', required=True,
                       help='List of Douban categories to merge (e.g., movies books)')
    parser.add_argument('--cache_dir', type=str, default='cache',
                       help='Cache directory path')
    
    args = parser.parse_args()
    
    print(f"[INFO] Creating Douban joint dataset: {args.categories}")
    output_path = create_douban_joint_dataset(args.categories, args.cache_dir)
    print(f"[INFO] Douban joint dataset created: {output_path}") 