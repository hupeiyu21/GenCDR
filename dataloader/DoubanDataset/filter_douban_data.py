#!/usr/bin/env python3
"""
Script to filter out users with less than 5 interactions from Douban dataset
"""

import json
import os
import logging
from collections import Counter
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def filter_all_item_seqs(all_item_seqs: Dict, min_interactions: int = 5) -> Dict:
    """Filter out users with less than min_interactions"""
    original_count = len(all_item_seqs)
    filtered_seqs = {}
    
    for user_id, item_seq in all_item_seqs.items():
        if len(item_seq) >= min_interactions:
            filtered_seqs[user_id] = item_seq
    
    filtered_count = len(filtered_seqs)
    removed_count = original_count - filtered_count
    
    logger.info(f"Original users: {original_count}")
    logger.info(f"Filtered users: {filtered_count}")
    logger.info(f"Removed users: {removed_count} ({removed_count/original_count*100:.2f}%)")
    
    return filtered_seqs

def update_id_mapping(id_mapping: Dict, filtered_seqs: Dict) -> Dict:
    """Update id_mapping to only include used users and items"""
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
    
    logger.info(f"Updated user mapping: {len(new_user2id)} users")
    logger.info(f"Updated item mapping: {len(new_item2id)} items")
    
    return updated_mapping

def update_metadata(metadata: Dict, used_items: set) -> Dict:
    """Update metadata to only include used items"""
    filtered_metadata = {}
    for item_id, item_data in metadata.items():
        if item_id in used_items:
            filtered_metadata[item_id] = item_data
    
    logger.info(f"Updated metadata: {len(filtered_metadata)} items")
    return filtered_metadata

def process_category(category_path: str, min_interactions: int = 5):
    """Process single category data"""
    logger.info(f"Processing category: {category_path}")
    
    # File paths
    all_item_seqs_path = os.path.join(category_path, 'processed', 'all_item_seqs.json')
    id_mapping_path = os.path.join(category_path, 'processed', 'id_mapping.json')
    metadata_path = os.path.join(category_path, 'processed', 'metadata.sentence.json')
    
    if not os.path.exists(all_item_seqs_path):
        logger.warning(f"all_item_seqs.json not found in {category_path}")
        return
    
    # Load original data
    logger.info("Loading original data...")
    with open(all_item_seqs_path, 'r', encoding='utf-8') as f:
        all_item_seqs = json.load(f)
    
    # Filter user sequences
    logger.info("Filtering user sequences...")
    filtered_seqs = filter_all_item_seqs(all_item_seqs, min_interactions)
    
    # Save filtered sequences
    filtered_seqs_path = os.path.join(category_path, 'processed', 'all_item_seqs_filtered.json')
    with open(filtered_seqs_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_seqs, f, indent=2, ensure_ascii=False)
    logger.info(f"Filtered sequences saved to: {filtered_seqs_path}")
    
    # Update id_mapping
    if os.path.exists(id_mapping_path):
        logger.info("Updating id_mapping...")
        with open(id_mapping_path, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
        
        updated_mapping = update_id_mapping(id_mapping, filtered_seqs)
        
        filtered_mapping_path = os.path.join(category_path, 'processed', 'id_mapping_filtered.json')
        with open(filtered_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(updated_mapping, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated mapping saved to: {filtered_mapping_path}")
    
    # Update metadata
    if os.path.exists(metadata_path):
        logger.info("Updating metadata...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        used_items = set()
        for item_seq in filtered_seqs.values():
            used_items.update(item_seq)
        
        filtered_metadata = update_metadata(metadata, used_items)
        
        filtered_metadata_path = os.path.join(category_path, 'processed', 'metadata.sentence_filtered.json')
        with open(filtered_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated metadata saved to: {filtered_metadata_path}")

def main():
    """Main function"""
    base_path = "../../cache/DoubanDataset"  # Adjusted path for dataloader/DoubanDataset location
    categories = ["books", "movies", "books_movies"]
    min_interactions = 5
    
    logger.info(f"Starting to filter Douban dataset with min_interactions={min_interactions}")
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        if os.path.exists(category_path):
            try:
                process_category(category_path, min_interactions)
                logger.info(f"Successfully processed {category}")
            except Exception as e:
                logger.error(f"Error processing {category}: {e}")
        else:
            logger.warning(f"Category path not found: {category_path}")
    
    logger.info("Filtering completed!")

if __name__ == "__main__":
    main()