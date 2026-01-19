# create_joint_dataset.py

import os
import json
import logging
from typing import Dict, List
import numpy as np
from tqdm import tqdm


def create_unified_id_mapping(categories: List[str], base_cache_dir: str) -> Dict:
    global_item2id = {"<pad>": 0}
    global_id2item = {"0": "<pad>"}
    global_item_counter = 1
    global_user2id = {}
    global_id2user = {}
    global_user_counter = 0

    logging.info("[JOINT-ID] Generating unified item and user mappings...")
    for category in categories:
        id_mapping_path = os.path.join(base_cache_dir, 'AmazonReviews2014', category, 'processed', 'id_mapping.json')
        if not os.path.exists(id_mapping_path): continue
        with open(id_mapping_path, 'r') as f:
            category_id_mapping = json.load(f)
        for item_asin in category_id_mapping['item2id']:
            if item_asin != "<pad>" and item_asin not in global_item2id:
                global_item2id[item_asin] = global_item_counter
                global_id2item[str(global_item_counter)] = item_asin
                global_item_counter += 1
        for local_user_id in category_id_mapping['user2id']:
            global_user_key = f"{category}_{local_user_id}"
            if global_user_key not in global_user2id:
                global_user2id[global_user_key] = global_user_counter
                global_id2user[str(global_user_counter)] = global_user_key
                global_user_counter += 1
    return {'user2id': global_user2id, 'id2user': global_id2user, 'item2id': global_item2id, 'id2item': global_id2item}

def merge_and_remap_all_item_seqs(categories: List[str], base_cache_dir: str, joint_id_mapping: Dict) -> Dict:

    logging.info("Merging user sequences (ASINs) from all domains...")
    merged_all_item_seqs = {}
    
    for category in categories:
        seqs_path = os.path.join(base_cache_dir, 'AmazonReviews2014', category, 'processed', 'all_item_seqs.json')
        if not os.path.exists(seqs_path):
            logging.warning(f"Sequence file not found for {category}, skipping.")
            continue
            
        with open(seqs_path, 'r') as f:
            category_seqs = json.load(f)
        
        for local_user_id, item_seq in category_seqs.items():
            global_user_key = f"{category}_{local_user_id}"
            merged_all_item_seqs[global_user_key] = item_seq
            
    logging.info(f"Total merged user sequences: {len(merged_all_item_seqs)}")
    return merged_all_item_seqs

def merge_metadata_sentences(categories: List[str], base_cache_dir: str, joint_id_mapping: Dict) -> Dict:
    merged_metadata, processed_asins = {}, set()
    global_item2id = joint_id_mapping['item2id']
    for category in categories:
        metadata_path = os.path.join(base_cache_dir, 'AmazonReviews2014', category, 'processed', 'metadata.sentence.json')
        if not os.path.exists(metadata_path): continue
        with open(metadata_path, 'r') as f: category_metadata = json.load(f)
        for item_asin, sentence in category_metadata.items():
            if item_asin not in processed_asins and item_asin in global_item2id:
                merged_metadata[str(global_item2id[item_asin])] = sentence
                processed_asins.add(item_asin)
    merged_metadata["0"] = "This is a placeholder item."
    return merged_metadata

def create_joint_embeddings(categories: List[str], base_cache_dir: str, config: dict):

    joint_category = "_".join(sorted(categories))
    joint_processed_dir = os.path.join(base_cache_dir, 'AmazonReviews2014', joint_category, 'processed')
    
    with open(os.path.join(joint_processed_dir, 'id_mapping.json'), 'r') as f:
        joint_id_mapping = json.load(f)
    
    global_id2item = joint_id_mapping['id2item']
    total_items = len(global_id2item) - 1

    data_config = config 
    
    use_pca = data_config.get('sent_emb_pca', 0) > 0
    if use_pca:
        any_pca_path = os.path.join(base_cache_dir, 'AmazonReviews2014', categories[0], 'processed', 'final_pca_embeddings.npy')
        if not os.path.exists(any_pca_path):
            raise FileNotFoundError(f"Required to use PCA, but source PCA file not found: {any_pca_path}")
        emb_dim = np.load(any_pca_path).shape[1]
        output_filename = 'final_pca_embeddings.npy'
    else:
        emb_dim = data_config.get('sent_emb_dim', 3072)
        model_name = os.path.basename(data_config.get('sent_emb_model', 'text-embedding-3-large'))
        output_filename = f'{model_name}.sent_emb'

    logging.info(f"Assembling joint embedding file with dim={emb_dim} (PCA: {use_pca})")

    joint_embeddings = np.zeros((total_items, emb_dim), dtype=np.float32)
    loaded_embeddings_cache = {}
    
    for global_id_str, asin in tqdm(global_id2item.items(), desc="Assembling Embeddings"):
        global_id = int(global_id_str)
        if global_id == 0: continue

        found = False
        for category in categories:
            local_id_map_path = os.path.join(base_cache_dir, 'AmazonReviews2014', category, 'processed', 'id_mapping.json')
            with open(local_id_map_path, 'r') as f: local_id_map = json.load(f)
            
            if asin in local_id_map['item2id']:
                local_id = local_id_map['item2id'][asin]
                if category not in loaded_embeddings_cache:
                    emb_input_filename = 'final_pca_embeddings.npy' if use_pca else 'text-embedding-3-large.sent_emb'
                    emb_path = os.path.join(base_cache_dir, 'AmazonReviews2014', category, 'processed', emb_input_filename)
                    if use_pca:
                        loaded_embeddings_cache[category] = np.load(emb_path)
                    else:
                        loaded_embeddings_cache[category] = np.fromfile(emb_path, dtype=np.float32).reshape(-1, emb_dim)
                
                joint_embeddings[global_id - 1] = loaded_embeddings_cache[category][local_id - 1]
                found = True
                break
        
        if not found:
            logging.warning(f"ASIN {asin} (Global ID: {global_id}) not found in any individual dataset's embeddings!")

    output_path = os.path.join(joint_processed_dir, output_filename)
    if use_pca:
        np.save(output_path, joint_embeddings)
    else:
        joint_embeddings.tofile(output_path)
        
    logging.info(f"Joint embedding file saved to: {output_path}")

def create_joint_dataset(categories: List[str], base_cache_dir: str, config: dict):

    joint_category = "_".join(sorted(categories))
    joint_dir = os.path.join(base_cache_dir, 'AmazonReviews2014', joint_category)
    joint_processed_dir = os.path.join(joint_dir, 'processed')
    os.makedirs(joint_processed_dir, exist_ok=True)
    
    logging.info(f"[JOINT] Creating joint dataset: {joint_category}")
    
    logging.info("\n[JOINT] Step 1: Creating unified ID mapping")
    joint_id_mapping = create_unified_id_mapping(categories, base_cache_dir)
    with open(os.path.join(joint_processed_dir, 'id_mapping.json'), 'w') as f: json.dump(joint_id_mapping, f, indent=2)

    logging.info("\n[JOINT] Step 2: Merging all_item_seqs.json")
    merged_all_item_seqs = merge_and_remap_all_item_seqs(categories, base_cache_dir, joint_id_mapping)
    with open(os.path.join(joint_processed_dir, 'all_item_seqs.json'), 'w') as f: json.dump(merged_all_item_seqs, f, indent=2)

    logging.info("\n[JOINT] Step 3: Merging metadata.sentence.json")
    merged_metadata = merge_metadata_sentences(categories, base_cache_dir, joint_id_mapping)
    with open(os.path.join(joint_processed_dir, 'metadata.sentence.json'), 'w') as f: json.dump(merged_metadata, f, indent=2)
    
    logging.info("\n[JOINT] Step 4: Assembling joint embeddings file...")
    create_joint_embeddings(categories, base_cache_dir, config)
    
    logging.info(f"\n[JOINT] All joint dataset files created successfully!")
    return joint_processed_dir