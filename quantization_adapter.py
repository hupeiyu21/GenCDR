# quantization_adapter.py

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

from quantization.rqvae.rqvae import RQVAE
from quantization.utils import set_weight_decay, calc_cos_sim
from dataloader.amazon_data_processor import AmazonDataProcessor

def check_and_prepare_data_single(config, dataset_name):
    logging.info(f"[{dataset_name}] Checking data integrity...")
    cache_dir = config.get('cache_dir', 'cache')
    processed_dir = os.path.join(cache_dir, 'AmazonReviews2014', dataset_name, 'processed')
    
    sent_emb_model = config.get('sent_emb_model', 'text-embedding-3-large')
    use_pca = config.get('sent_emb_pca', 0) > 0
    if use_pca:
        emb_filename = 'final_pca_embeddings.npy'
    else:
        emb_filename = f'{os.path.basename(sent_emb_model)}.sent_emb'

    required_files = [
        os.path.join(processed_dir, 'id_mapping.json'),
        os.path.join(processed_dir, emb_filename)
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logging.info(f"[{dataset_name}] Missing files detected, starting data processing pipeline...")
        try:
            processor = AmazonDataProcessor(category=dataset_name, cache_dir=cache_dir, config=config)
            processor.run_full_pipeline()
            logging.info(f"[{dataset_name}] Data processing pipeline completed.")
        except Exception as e:
            logging.error(f"[{dataset_name}] Error during data processing: {e}", exc_info=True)
            return False
    else:
        logging.info(f"[{dataset_name}] All required data files exist.")
    
    return True

def load_embeddings_for_domain(dataset_name, config):
    data_config = config['data_processing']
    cache_dir = data_config.get('cache_dir', 'cache')
    processed_dir = os.path.join(cache_dir, 'AmazonReviews2014', dataset_name, 'processed')

    sent_emb_model_name = os.path.basename(data_config.get('sent_emb_model', 'text-embedding-3-large'))
    use_pca = data_config.get('sent_emb_pca', 0) > 0
    if use_pca:
        emb_path = os.path.join(processed_dir, 'final_pca_embeddings.npy')
        logging.info(f"[{dataset_name}] Loading PCA embeddings from: {emb_path}")
        embeddings = np.load(emb_path)
    else:
        emb_path = os.path.join(processed_dir, f'{sent_emb_model_name}.sent_emb')
        logging.info(f"[{dataset_name}] Loading raw embeddings from: {emb_path}")
        emb_dim = data_config.get('sent_emb_dim', 3072)
        embeddings = np.fromfile(emb_path, dtype=np.float32).reshape(-1, emb_dim)

    id_mapping_path = os.path.join(processed_dir, 'id_mapping.json')
    with open(id_mapping_path, 'r') as f:
        id_mapping = json.load(f)

    num_items_in_map = len(id_mapping['item2id'])
    if embeddings.shape[0] == num_items_in_map - 1:
        pad_embedding = np.zeros((1, embeddings.shape[1]), dtype=np.float32)
        final_embeddings = np.concatenate([pad_embedding, embeddings], axis=0)
    elif embeddings.shape[0] == num_items_in_map:
        final_embeddings = embeddings
    else:
        raise ValueError(f"[{dataset_name}] Mismatch in embedding count ({embeddings.shape[0]}) and item count ({num_items_in_map-1}).")

    return torch.tensor(final_embeddings, dtype=torch.float32), id_mapping

def generate_codebook_for_domain(model, embeddings_tensor, id_mapping, domain_name, config):
    logging.info(f"[{domain_name}] Generating codebook...")
    model.eval()
    device = next(model.parameters()).device
    model.to(device)

    embeddings_no_pad = embeddings_tensor[1:].to(device)
    
    batch_size = config['RQ-VAE']['batch_size']
    eval_dataloader = DataLoader(TensorDataset(embeddings_no_pad), batch_size=batch_size, shuffle=False)
    
    all_codes_list = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"[{domain_name}] Generating codes"):
            x_batch = batch[0]
            codes = model.get_codes(x_batch).cpu().numpy()
            all_codes_list.append(codes)
    
    all_codes_np = np.vstack(all_codes_list)
    
    all_codes_str = ["-".join(map(str, row)) for row in all_codes_np]
    total_items = len(all_codes_str)
    unique_codes = len(set(all_codes_str))
    collisions = total_items - unique_codes
    collision_rate = collisions / total_items if total_items > 0 else 0
    logging.info(f"[{domain_name}] Final Collision Rate: {collision_rate:.2%}")
    
    codebook_dict = {str(i + 1): code.tolist() for i, code in enumerate(all_codes_np)}

    cache_dir = config['data_processing']['cache_dir']
    codebook_dir = os.path.join(cache_dir, "AmazonReviews2014", domain_name, "codebook")
    os.makedirs(codebook_dir, exist_ok=True)
    codebook_path = os.path.join(codebook_dir, f"{domain_name}_codebook.json")
    
    with open(codebook_path, 'w') as f:
        json.dump(codebook_dict, f, indent=2)
    
    logging.info(f"[{domain_name}] Codebook with {len(codebook_dict)} items saved to: {codebook_path}")
    return codebook_path

def train_epoch(model, dataloader, optimizer, config, device, flag_eval=False):
    model.eval() if flag_eval else model.train()
    beta = config["beta"]
    total_loss, total_rec_loss, total_commit_loss = 0.0, 0.0, 0.0

    for i_batch, batch in enumerate(dataloader):
        x_batch = batch[0].to(device)
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


def run_training_for_domain(domain_name, config, device):

    logging.info(f"\n{'='*25} Starting Pipeline for Domain: {domain_name.upper()} {'='*25}")
    
    if not check_and_prepare_data_single(config['data_processing'], domain_name):
        logging.error(f"[{domain_name}] Data preparation failed. Skipping this domain.")
        return

    try:
        embeddings, id_mapping = load_embeddings_for_domain(domain_name, config)
        embeddings_for_training = embeddings[1:]
        input_dim = embeddings_for_training.shape[1]
        logging.info(f"[{domain_name}] Loaded embeddings with shape: {embeddings.shape}")
    except Exception as e:
        logging.error(f"[{domain_name}] Failed to load embeddings: {e}", exc_info=True)
        return

    model_config = config['RQ-VAE']
    model = RQVAE(
        input_size=input_dim,
        hidden_sizes=model_config["hidden_dim"],
        latent_size=model_config["latent_dim"],
        num_levels=model_config["num_layers"],
        codebook_size=model_config["code_book_size"],
        dropout=model_config["dropout"],
        latent_loss_weight=model_config["beta"]
    ).to(device)

    logging.info(f"[{domain_name}] Starting new model training...")
    optimizer = getattr(torch.optim, model_config["optimizer"])(model.parameters(), lr=model_config["lr"])
    if "weight_decay" in optimizer.param_groups[0] and model_config.get("weight_decay") is not None:
        set_weight_decay(optimizer, model_config["weight_decay"])
    
    trainset_np, _ = train_test_split(embeddings_for_training.cpu().numpy(), test_size=0.05, random_state=42)
    train_dataset = TensorDataset(torch.Tensor(trainset_np))
    train_dataloader = DataLoader(train_dataset, batch_size=model_config["batch_size"], shuffle=True)
    
    for epoch in tqdm(range(model_config["epochs"]), desc=f"[{domain_name}] Training"):
        model.train()
        train_loss, train_rec_loss, train_commit_loss = train_epoch(model, train_dataloader, optimizer, model_config, device)
        if (epoch + 1) % 10 == 0:
            logging.info(f"[{domain_name}] Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Recon: {train_rec_loss:.4f} | Commit: {train_commit_loss:.4f}")

    logging.info(f"[{domain_name}] Training finished.")
    
    save_dir = os.path.join("ckpt", "rqvae_independent", domain_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"rqvae-{domain_name}.pth")
    torch.save(model.state_dict(), save_path)
    logging.info(f"[{domain_name}] Model saved to: {save_path}")

    generate_codebook_for_domain(model, embeddings, id_mapping, domain_name, config)

    logging.info(f"{'='*25} Pipeline for Domain: {domain_name.upper()} Finished {'='*25}\n")

# ==========================================================================================
# Main Function
# ==========================================================================================
def main():
    parser = argparse.ArgumentParser(description="Train an independent RQ-VAE for each specified domain and generate codebook.")
    parser.add_argument('--config', type=str, default='quantization/rqvae_config.yaml', help='Path to config file')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to train on, e.g., Clothing_Shoes_and_Jewelry Sports_and_Outdoors')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    log_dir = os.path.join('logs', 'rqvae_independent_runs')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

    device = torch.device(config.get('training', {}).get('device', 'cuda:0'))

    logging.info(f"Starting independent training for domains: {args.datasets}")
    for domain_name in args.datasets:
        try:
            run_training_for_domain(domain_name, config, device)
        except Exception as e:
            logging.error(f"!!!!!! An uncaught error occurred while processing domain: {domain_name} !!!!!!")
            logging.error(f"Error details: {e}", exc_info=True)
    
    logging.info("All specified domains have been processed.")

if __name__ == '__main__':
    main()