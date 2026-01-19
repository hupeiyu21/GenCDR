# quantization.py

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml
import argparse
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from quantization.utils import set_weight_decay, calc_cos_sim
from quantization.rqvae.rqvae import RQVAE
from dataloader.amazon_data_processor import AmazonDataProcessor
from dataloader.create_joint_dataset import create_joint_dataset
from dataloader.douban_joint_dataset import create_douban_joint_dataset_with_unified_embeddings

def check_and_prepare_data_single(config, dataset_name):
    cache_dir = config.get('cache_dir', 'cache')
    category = dataset_name
    
    processed_dir = os.path.join(cache_dir, 'AmazonReviews2014', category, 'processed')
    
    required_files = [
        os.path.join(processed_dir, 'all_item_seqs.json'),
        os.path.join(processed_dir, 'id_mapping.json'),
        os.path.join(processed_dir, 'metadata.sentence.json')
    ]
    
    sent_emb_model = config.get('sent_emb_model', 'text-embedding-3-large')
    sent_emb_path = os.path.join(
        processed_dir,
        f'{os.path.basename(sent_emb_model)}.sent_emb'
    )
    required_files.append(sent_emb_path)
    
    pca_emb_path = None
    if config.get('sent_emb_pca', 0) > 0:
        pca_emb_path = os.path.join(processed_dir, 'final_pca_embeddings.npy')
        required_files.append(pca_emb_path)
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logging.info(f"[TRAINING] [{dataset_name}] Missing files detected, starting data processing pipeline...")
        logging.info(f"[TRAINING] [{dataset_name}] Missing files: {missing_files}")
        
        try:
            processor = AmazonDataProcessor(
                category=category,
                cache_dir=cache_dir,
                config=config
            )
            processor.run_full_pipeline()
            logging.info(f"[TRAINING] [{dataset_name}] Data processing pipeline completed")
            
            still_missing = [f for f in required_files if not os.path.exists(f)]
            if still_missing:
                logging.error(f"[ERROR] [{dataset_name}] Files still missing after processing: {still_missing}")
                return False
                
        except Exception as e:
            logging.error(f"[ERROR] [{dataset_name}] Error during data processing: {e}", exc_info=True)
            return False
    else:
        logging.info(f"[TRAINING] [{dataset_name}] All required data files exist, skipping data processing")
        if pca_emb_path and os.path.exists(pca_emb_path):
            logging.info(f"[TRAINING] [{dataset_name}] Found PCA embedding file: {pca_emb_path}")
    
    return True


def check_and_prepare_data_multi(config, dataset_names):
    logging.info(f"[TRAINING] Starting parallel data preparation for datasets: {dataset_names}")
    
    with ThreadPoolExecutor(max_workers=len(dataset_names)) as executor:
        future_to_dataset = {
            executor.submit(check_and_prepare_data_single, config, dataset_name): dataset_name 
            for dataset_name in dataset_names
        }
        
        results = {}
        for future in as_completed(future_to_dataset):
            dataset_name = future_to_dataset[future]
            try:
                result = future.result()
                results[dataset_name] = result
                logging.info(f"[TRAINING] [{dataset_name}] Data preparation {'completed' if result else 'failed'}")
            except Exception as e:
                logging.error(f"[ERROR] [{dataset_name}] Exception during data preparation: {e}", exc_info=True)
                results[dataset_name] = False
    
    failed_datasets = [name for name, success in results.items() if not success]
    if failed_datasets:
        logging.error(f"[ERROR] Failed to prepare data for datasets: {failed_datasets}")
        return False
    
    logging.info(f"[TRAINING] All datasets prepared successfully: {list(results.keys())}")
    return True

def check_and_create_joint_dataset(config, dataset_names):
    if len(dataset_names) <= 1:
        logging.info("[TRAINING] Single dataset mode, skipping joint dataset creation")
        return True
    
    data_config = config
    use_pca = data_config.get('sent_emb_pca', 0) > 0
    if use_pca:
        embedding_filename = 'final_pca_embeddings.npy'
    else:
        model_name = os.path.basename(data_config.get('sent_emb_model', 'text-embedding-3-large'))
        embedding_filename = f'{model_name}.sent_emb'

    cache_dir = data_config.get('cache_dir', 'cache')
    joint_dataset_name = "_".join(sorted(dataset_names))
    joint_processed_dir = os.path.join(cache_dir, 'AmazonReviews2014', joint_dataset_name, 'processed')
    
    required_joint_files = [
        os.path.join(joint_processed_dir, 'all_item_seqs.json'),
        os.path.join(joint_processed_dir, 'id_mapping.json'),
        os.path.join(joint_processed_dir, 'metadata.sentence.json'),
        os.path.join(joint_processed_dir, embedding_filename)
    ]
    
    missing_joint_files = [f for f in required_joint_files if not os.path.exists(f)]
    
    if missing_joint_files:
        logging.info(f"[TRAINING] One or more joint dataset files are missing. Recreating ALL joint files.")
        logging.info(f"Missing files: {missing_joint_files}")
        try:
            create_joint_dataset(categories=dataset_names, base_cache_dir=cache_dir, config=config)
            
            still_missing = [f for f in required_joint_files if not os.path.exists(f)]
            if still_missing:
                logging.error(f"[ERROR] Joint dataset creation failed, still missing: {still_missing}")
                return False
        except Exception as e:
            logging.error(f"[ERROR] Failed to create joint dataset: {e}", exc_info=True)
            return False
    else:
        logging.info(f"[TRAINING] All required joint dataset files (including embeddings) already exist. Skipping creation.")
    
    return True

def check_and_prepare_douban_data(config):
    cache_dir = config.get('cache_dir', 'cache')
    douban_categories = ['movies', 'books']
    logging.info(f"Preparing Douban datasets: {douban_categories}")
    joint_category = "_".join(sorted(douban_categories))
    joint_processed_dir = os.path.join(cache_dir, 'DoubanDataset', joint_category, 'processed')
    required_joint_files = [
        os.path.join(joint_processed_dir, 'all_item_seqs.json'),
        os.path.join(joint_processed_dir, 'id_mapping.json'),
        os.path.join(joint_processed_dir, 'metadata.sentence.json'),
        os.path.join(joint_processed_dir, f"{os.path.basename(config.get('sent_emb_model', 'text-embedding-3-large'))}.sent_emb.npy")
    ]
    if any(not os.path.exists(f) for f in required_joint_files):
        logging.info("Douban joint dataset files missing, creating...")
        try:
            create_douban_joint_dataset_with_unified_embeddings(categories=douban_categories, base_cache_dir=cache_dir, config=config)
            if any(not os.path.exists(f) for f in required_joint_files):
                logging.error("Douban joint dataset creation failed, files still missing.")
                return False
        except Exception as e:
            logging.error(f"Failed to create Douban joint dataset: {e}", exc_info=True)
            return False
    logging.info("All required Douban joint dataset files exist.")
    return True

def load_douban_embeddings(base_cache_dir, config, device):
    joint_processed_dir = os.path.join(base_cache_dir, 'DoubanDataset', 'movies_books', 'processed')
    id_mapping_path = os.path.join(joint_processed_dir, 'id_mapping.json')
    emb_path = os.path.join(joint_processed_dir, f"{os.path.basename(config.get('data_processing', {}).get('sent_emb_model', 'text-embedding-3-large'))}.sent_emb.npy")
    
    with open(id_mapping_path, 'r') as f: id_mapping = json.load(f)
    embeddings = np.load(emb_path)

    if embeddings.shape[0] != len(id_mapping['item2id']) - 1:
        raise ValueError("Mismatch between Douban embedding count and item count.")

    final_embeddings = np.concatenate([np.zeros((1, embeddings.shape[1]), dtype=np.float32), embeddings], axis=0)
    dataset_info = build_dataset_info(['movies', 'books'], base_cache_dir, id_mapping, 'DoubanDataset')
    return torch.tensor(final_embeddings, dtype=torch.float32), dataset_info, id_mapping

# -----------------------------------------------------------------------------

def load_data_and_embeddings(dataset_names, base_cache_dir, config, device):

    data_config = config.get('data_processing', {})
    sent_emb_model_name = os.path.basename(data_config.get('sent_emb_model', 'text-embedding-3-large'))
    
    if len(dataset_names) > 1:
        joint_dataset_name = "_".join(sorted(dataset_names))
        processed_dir = os.path.join(base_cache_dir, 'AmazonReviews2014', joint_dataset_name, 'processed')
    else:
        processed_dir = os.path.join(base_cache_dir, 'AmazonReviews2014', dataset_names[0], 'processed')

    id_mapping_path = os.path.join(processed_dir, 'id_mapping.json')
    pca_emb_path = os.path.join(processed_dir, 'final_pca_embeddings.npy')
    raw_emb_path = os.path.join(processed_dir, f'{sent_emb_model_name}.sent_emb')

    if not os.path.exists(id_mapping_path):
        raise FileNotFoundError(f"Critical file id_mapping.json not found at: {id_mapping_path}")

    if data_config.get('sent_emb_pca', 0) > 0 and os.path.exists(pca_emb_path):
        logging.info(f"Loading PCA embeddings from: {pca_emb_path}")
        embeddings = np.load(pca_emb_path)
    elif os.path.exists(raw_emb_path):
        logging.info(f"Loading raw embeddings from: {raw_emb_path}")
        emb_dim = data_config.get('sent_emb_dim', 3072)
        embeddings = np.fromfile(raw_emb_path, dtype=np.float32).reshape(-1, emb_dim)
    else:
        raise FileNotFoundError(f"No embedding file found. Checked for '{pca_emb_path}' and '{raw_emb_path}'")

    with open(id_mapping_path, 'r') as f:
        id_mapping = json.load(f)

    num_items_in_map = len(id_mapping['item2id'])    
    num_items_no_pad = num_items_in_map - 1          
    num_embeddings_in_file = embeddings.shape[0]    

    final_embeddings = None
    if num_embeddings_in_file == num_items_no_pad:
        logging.info(f"Embeddings file count ({num_embeddings_in_file}) matches item count ({num_items_no_pad}). Adding pad vector manually.")
        pad_embedding = np.zeros((1, embeddings.shape[1]), dtype=np.float32)
        final_embeddings = np.concatenate([pad_embedding, embeddings], axis=0)

    elif num_embeddings_in_file == num_items_in_map:
        logging.info(f"Embeddings file count ({num_embeddings_in_file}) matches item count including <pad> ({num_items_in_map}). Assuming file is complete.")
        final_embeddings = embeddings
        
    else:
        raise ValueError(
            f"Fatal mismatch: Number of embeddings in file ({num_embeddings_in_file}) "
            f"does not match expected item count with pad ({num_items_in_map}) "
            f"or without pad ({num_items_no_pad})."
        )

    dataset_info = build_dataset_info(dataset_names, base_cache_dir, id_mapping)

    return torch.tensor(final_embeddings, dtype=torch.float32), dataset_info, id_mapping

def build_dataset_info(dataset_names, base_cache_dir, global_id_mapping):
    global_item_asin_to_id = global_id_mapping['item2id']
    dataset_info = {}

    for name in dataset_names:
        local_id_mapping_path = os.path.join(base_cache_dir, 'AmazonReviews2014', name, 'processed', 'id_mapping.json')
        with open(local_id_mapping_path, 'r') as f:
            local_id_mapping = json.load(f)
        
        local_id2item = local_id_mapping['id2item']
        local_to_global_map = {}

        iterator = None
        if isinstance(local_id2item, dict):
            iterator = local_id2item.items()
        elif isinstance(local_id2item, list):
            iterator = enumerate(local_id2item)
        
        if iterator:
            for local_id, item_asin in iterator:
                local_id_str = str(local_id)
                
                if item_asin != '<pad>' and item_asin in global_item_asin_to_id:
                    global_id = global_item_asin_to_id[item_asin]
                    local_to_global_map[local_id_str] = global_id

        dataset_info[name] = {
            'local_item_count': len(local_to_global_map),
            'local_to_global_id_map': local_to_global_map
        }
    return dataset_info

def build_dedup_layer(base_codes_np: np.ndarray, vocab_size: int) -> np.ndarray:
    
    logging.info("Building deduplication layer...")
    N = base_codes_np.shape[0]
    groups = defaultdict(list)

    # Group items that share the same quantized code sequence
    for idx, key in enumerate(map(tuple, base_codes_np)):
        groups[key].append(idx)

    dedup_layer = np.zeros((N, 1), dtype=np.int64)
    max_dup, overflow_count = 0, 0

    # Assign local indices within each duplicate cluster
    for idx_list in groups.values():
        k = len(idx_list)
        max_dup = max(max_dup, k)
        if k > vocab_size:
            logging.warning(
                f"Duplicate cluster size {k} exceeds codebook size {vocab_size}. "
                f"Local IDs will wrap around modulo vocab size, which may cause collisions."
            )
            local_ids = np.arange(k, dtype=np.int64) % vocab_size
            overflow_count += 1
        else:
            local_ids = np.arange(k, dtype=np.int64)
        dedup_layer[np.array(idx_list), 0] = local_ids

    logging.info(
        f"Deduplication layer built. "
        f"Maximum duplicates in a cluster: {max_dup}. "
        f"Clusters with modulo overflow: {overflow_count}."
    )
    return dedup_layer

# -----------------------------------------------------------------------------

def generate_unified_codebook(model, merged_embeddings, id_mapping, dataset_info, config, device):
    logging.info("[CODEBOOK] Generating unified codebook...")
    model.to(device)
    model.eval()

    model_config = config["RQ-VAE"]
    cache_dir = config.get('data_processing', {}).get('cache_dir', 'cache')

    embeddings_no_pad = merged_embeddings[1:]
    eval_dataset = TensorDataset(embeddings_no_pad)
    eval_dataloader = DataLoader(eval_dataset, batch_size=model_config["batch_size"], shuffle=False)

    all_codes_list = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="[RQ-VAE] Generating codes"):
            x_batch = batch[0].to(device)
            codes = model.get_codes(x_batch).cpu().numpy()
            all_codes_list.append(codes)
    all_codes_np = np.vstack(all_codes_list)
    
    logging.info(f"[CODEBOOK] Base codes generated, shape: {all_codes_np.shape}. Building dedup layer...")
    code_book_size = model_config["code_book_size"]
    dedup_layer = build_dedup_layer(all_codes_np, code_book_size)
    
    final_codes_np = np.hstack([all_codes_np, dedup_layer])
    logging.info(f"[CODEBOOK] Dedup layer built. Final codes shape: {final_codes_np.shape}")

    is_multi_domain = len(dataset_info) > 1
    dataset_keys = sorted(dataset_info.keys())
    save_dir_name = '_'.join(dataset_keys) if is_multi_domain else dataset_keys[0]

    global_codebook_dir = os.path.join(cache_dir, "AmazonReviews2014", save_dir_name, "codebook")
    os.makedirs(global_codebook_dir, exist_ok=True)
    global_codebook_path = os.path.join(global_codebook_dir, "joint_codebook.json")
    with open(global_codebook_path, 'w') as f:
        json.dump({str(i + 1): code.tolist() for i, code in enumerate(final_codes_np)}, f)
    logging.info(f"[CODEBOOK] Global codebook saved to: {global_codebook_path}")

    if is_multi_domain:
        logging.info("[CODEBOOK] Generating individual codebooks for each dataset...")
        for dataset_name, info in dataset_info.items():
            local_codebook = {}
            local_to_global_map = info['local_to_global_id_map']
            
            for local_id_str, global_id in local_to_global_map.items():
                code_index = global_id - 1
                if 0 <= code_index < len(final_codes_np):
                    local_codebook[local_id_str] = final_codes_np[code_index].tolist()
            
            local_codebook_dir = os.path.join(cache_dir, "AmazonReviews2014", dataset_name, "codebook")
            os.makedirs(local_codebook_dir, exist_ok=True)
            local_codebook_path = os.path.join(local_codebook_dir, f"{dataset_name}_codebook.json")
            with open(local_codebook_path, 'w') as f:
                json.dump(local_codebook, f)
            logging.info(f"[CODEBOOK] [{dataset_name}] Individual codebook ({len(local_codebook)} items) saved.")

    logging.info("[CODEBOOK] Codebook generation finished.")
    return global_codebook_path

def train_epoch(model, dataloader, optimizer, config, flag_eval=False):
    model.eval() if flag_eval else model.train()
    beta = config["beta"]
    total_loss, total_rec_loss, total_commit_loss = 0.0, 0.0, 0.0

    for i_batch, batch in enumerate(dataloader):
        x_batch = batch[0]
        if not flag_eval:
            optimizer.zero_grad()

        with torch.set_grad_enabled(not flag_eval):
            recon_x, commitment_loss, _ = model(x_batch)
            reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction="mean")
            loss = reconstruction_mse_loss + beta * commitment_loss

        if not flag_eval:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_rec_loss += reconstruction_mse_loss.item()
        total_commit_loss += commitment_loss.item()

    return total_loss / len(dataloader), total_rec_loss / len(dataloader), total_commit_loss / len(dataloader)

def train_rqvae(model, x, device, config):
    model.to(device)
    model_config = config["RQ-VAE"]
    batch_size = model_config["batch_size"]
    num_epochs = model_config["epochs"]
    lr = model_config["lr"]
    n_eval_interval = model_config.get("eval_interval", 100)

    optimizer = getattr(torch.optim, model_config["optimizer"])(model.parameters(), lr=lr)
    if "weight_decay" in optimizer.param_groups[0]:
        set_weight_decay(optimizer, model_config["weight_decay"])

    trainset, validationset = train_test_split(x, test_size=0.05, random_state=42)
    trainset = torch.Tensor(trainset).to(device)
    validationset = torch.Tensor(validationset).to(device)
    train_dataset = TensorDataset(trainset)
    val_dataset = TensorDataset(validationset)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in tqdm(range(num_epochs), desc="Training RQ-VAE"):
        train_loss, train_rec_loss, train_commit_loss = train_epoch(model, dataloader, optimizer, model_config)
        
        if (epoch + 1) % 10 == 0:
             logging.info(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Recon: {train_rec_loss:.4f} | Commit: {train_commit_loss:.4f}")
        
        if (epoch + 1) % n_eval_interval == 0 and len(validationset) > 0:
            val_loss, val_rec_loss, val_commit_loss = train_epoch(model, val_dataloader, None, model_config, flag_eval=True)
            
            try:
                cos_sim_array = calc_cos_sim(model, validationset, model_config)
                logging.info(f"--- Validation @ Epoch {epoch+1} ---")
                logging.info(f"    Validation Recon Loss: {val_rec_loss:.4f} | Commit Loss: {val_commit_loss:.4f}")
                for i in range(model_config["num_layers"]):
                    logging.info(f"    Eval Cosine Sim @L{i+1}: {cos_sim_array[i]:.4f}")
                logging.info("-" * (len(f"--- Validation @ Epoch {epoch+1} ---")))
            except NameError:
                logging.warning("`calc_cos_sim` function not found, skipping cosine similarity calculation.")
            except Exception as e:
                logging.error(f"Error during validation: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train RQ-VAE with cross-domain support")
    parser.add_argument('--config', type=str, default='quantization/rqvae_config.yaml', help='Path to config file')
    parser.add_argument('--datasets', type=str, nargs='+', help='List of datasets (e.g., Clothing_Shoes_and_Jewelry Sports_and_Outdoors)')
    parser.add_argument('--douban', action='store_true', help='Use Douban books+movies joint dataset')

    parser.add_argument('--hidden_dim', type=int, nargs='+', help='Hidden dimensions for RQ-VAE')
    parser.add_argument('--latent_dim', type=int, help='Latent dimension for RQ-VAE')
    parser.add_argument('--num_layers', type=int, help='Number of layers for RQ-VAE')
    parser.add_argument('--code_book_size', type=int, help='Codebook size for RQ-VAE')
    parser.add_argument('--dropout', type=float, help='Dropout rate for RQ-VAE')
    parser.add_argument('--beta', type=float, help='Beta parameter for RQ-VAE')
    
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--optimizer', type=str, help='Optimizer name')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    
    args = parser.parse_args()

    if args.douban:
        args.datasets = ['DoubanBook', 'DoubanMovie']
    if not args.datasets:
        parser.error("You must specify --datasets or --douban")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.hidden_dim:
        config["RQ-VAE"]["hidden_dim"] = args.hidden_dim
    if args.latent_dim:
        config["RQ-VAE"]["latent_dim"] = args.latent_dim
    if args.num_layers:
        config["RQ-VAE"]["num_layers"] = args.num_layers
    if args.code_book_size:
        config["RQ-VAE"]["code_book_size"] = args.code_book_size
    if args.dropout is not None:
        config["RQ-VAE"]["dropout"] = args.dropout
    if args.beta is not None:
        config["RQ-VAE"]["beta"] = args.beta
    
    if args.epochs:
        config["RQ-VAE"]["epochs"] = args.epochs
    if args.batch_size:
        config["RQ-VAE"]["batch_size"] = args.batch_size
    if args.lr:
        config["RQ-VAE"]["lr"] = args.lr
    if args.optimizer:
        config["RQ-VAE"]["optimizer"] = args.optimizer
    if args.weight_decay is not None:
        config["RQ-VAE"]["weight_decay"] = args.weight_decay

    if args.douban:
        dataset_type, dataset_names = 'douban', ['movies', 'books']
        if args.datasets: logging.warning("--datasets argument is ignored when --douban is used.")
    elif args.datasets:
        dataset_type, dataset_names = 'amazon', args.datasets
    else:
        raise ValueError("Please specify either --datasets (for Amazon) or --douban flag.")

    log_dataset_name = "_".join(sorted(dataset_names)) if len(dataset_names) > 1 else dataset_names[0]
    
    log_dir = os.path.join('logs', 'rqvae', log_dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

    device = torch.device(config.get('training', {}).get('device', 'cuda:0'))
    data_config = config.get('data_processing', {})
    cache_dir = data_config.get('cache_dir', 'cache')

    logging.info(f"--- Running pipeline for {dataset_type.upper()} dataset ---")
    if dataset_type == 'douban':
        logging.info("STEP 1&2: Checking and preparing Douban joint dataset...")
        if not check_and_prepare_douban_data(data_config): return
    else: # amazon
        logging.info("STEP 1: Checking and preparing individual Amazon datasets...")
        if not check_and_prepare_data_multi(data_config, dataset_names): return
        if len(dataset_names) > 1:
            logging.info("STEP 2: Checking and creating joint Amazon dataset...")
            if not check_and_create_joint_dataset(data_config, dataset_names): return
    
    logging.info("STEP 3: Loading embeddings and mappings...")
    try:
        if dataset_type == 'douban':
            merged_embeddings, dataset_info, id_mapping = load_douban_embeddings(cache_dir, config, device)
        else: # amazon
            merged_embeddings, dataset_info, id_mapping = load_data_and_embeddings(dataset_names, cache_dir, config, device)
        logging.info(f"Successfully loaded embeddings. Shape: {merged_embeddings.shape}")
        
        info_save_dir = os.path.join(cache_dir, 'AmazonReviews2014' if dataset_type == 'amazon' else 'DoubanDataset', log_dataset_name, 'processed')
        os.makedirs(info_save_dir, exist_ok=True)
        info_save_path = os.path.join(info_save_dir, 'dataset_info.json')
        with open(info_save_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        logging.info(f"Dataset info saved to: {info_save_path}")

    except Exception as e:
        logging.error(f"Failed to load data: {e}", exc_info=True)
        return

    logging.info("STEP 4: Initializing or loading RQ-VAE model...")
    model_config = config["RQ-VAE"]
    input_size = merged_embeddings.shape[1]

    rqvae = RQVAE(
        input_size=input_size,
        hidden_sizes=model_config["hidden_dim"],
        latent_size=model_config["latent_dim"],
        num_levels=model_config["num_layers"],
        codebook_size=model_config["code_book_size"],
        dropout=model_config["dropout"],
        latent_loss_weight=model_config["beta"]
    )
    
    save_dir = os.path.join("ckpt", "rqvae", log_dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"rqvae-{log_dataset_name}.pth")

    embeddings_for_training = merged_embeddings[1:]

    if os.path.exists(save_path):
        logging.info(f"Found existing model at: {save_path}. Skipping training.")
        rqvae.load_state_dict(torch.load(save_path, map_location=device))
    else:
        logging.info("No existing model found. Starting training...")
        train_rqvae(rqvae, embeddings_for_training.numpy(), device, config)

        logging.info("Training complete. Starting final collision detection...")
        rqvae.to(device)
        rqvae.eval()

        all_codes_list = []
        eval_dataset = TensorDataset(embeddings_for_training.to(device))
        eval_dataloader = DataLoader(eval_dataset, batch_size=model_config["batch_size"], shuffle=False)

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Generating all codes for collision check"):
                x_batch = batch[0]
                codes = rqvae.get_codes(x_batch).cpu().numpy()
                all_codes_list.append(codes)

        all_codes_np = np.vstack(all_codes_list)
        all_codes_str = ["-".join(map(str, row)) for row in all_codes_np]

        total_items = len(all_codes_str)
        unique_items = len(set(all_codes_str))
        num_duplicates = total_items - unique_items
        collision_rate = num_duplicates / total_items if total_items > 0 else 0

        logging.info("--- [COLLISION] Final Collision Detection Results ---")
        logging.info(f"    Total Items: {total_items}")
        logging.info(f"    Unique Codes: {unique_items}")
        logging.info(f"    Duplicated Items (Collisions): {num_duplicates}")
        logging.info(f"    Final Collision Rate: {collision_rate:.4%}")
        logging.info("-----------------------------------------------------")

        torch.save(rqvae.state_dict(), save_path)
        logging.info(f"Training complete! Model saved to: {save_path}")

    logging.info("STEP 5: Generating codebooks...")
    rqvae.to(device)
    rqvae.eval()
    unified_codebook_path = generate_unified_codebook(
        rqvae, merged_embeddings, id_mapping, dataset_info, config, device
    )

    logging.info("--- ALL DONE! ---")
    logging.info(f"Final Model Path: {save_path}")
    logging.info(f"Final Codebook Path: {unified_codebook_path}")

if __name__ == '__main__':
    main()