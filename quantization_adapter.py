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

from peft import LoraConfig, get_peft_model

from quantization.rqvae.rqvae import RQVAE
from quantization.utils import set_weight_decay
from dataloader.amazon_data_processor import AmazonDataProcessor


def check_and_prepare_data_single(config, dataset_name):
    cache_dir = config.get('cache_dir', 'cache')
    processed_dir = os.path.join(cache_dir, 'AmazonReviews2014', dataset_name, 'processed')

    sent_emb_model = config.get('sent_emb_model', 'text-embedding-3-large')
    use_pca = config.get('sent_emb_pca', 0) > 0
    emb_filename = 'final_pca_embeddings.npy' if use_pca else f'{os.path.basename(sent_emb_model)}.sent_emb'

    required_files = [
        os.path.join(processed_dir, 'id_mapping.json'),
        os.path.join(processed_dir, emb_filename)
    ]

    if any(not os.path.exists(f) for f in required_files):
        processor = AmazonDataProcessor(
            category=dataset_name,
            cache_dir=cache_dir,
            config=config
        )
        processor.run_full_pipeline()

    return True


def load_embeddings_for_domain(dataset_name, config):
    data_config = config['data_processing']
    cache_dir = data_config.get('cache_dir', 'cache')
    processed_dir = os.path.join(cache_dir, 'AmazonReviews2014', dataset_name, 'processed')

    sent_emb_model_name = os.path.basename(data_config.get('sent_emb_model', 'text-embedding-3-large'))
    use_pca = data_config.get('sent_emb_pca', 0) > 0

    if use_pca:
        embeddings = np.load(os.path.join(processed_dir, 'final_pca_embeddings.npy'))
    else:
        emb_dim = data_config.get('sent_emb_dim', 3072)
        embeddings = np.fromfile(
            os.path.join(processed_dir, f'{sent_emb_model_name}.sent_emb'),
            dtype=np.float32
        ).reshape(-1, emb_dim)

    with open(os.path.join(processed_dir, 'id_mapping.json'), 'r') as f:
        id_mapping = json.load(f)

    pad = np.zeros((1, embeddings.shape[1]), dtype=np.float32)
    embeddings = np.concatenate([pad, embeddings], axis=0)

    return torch.tensor(embeddings, dtype=torch.float32), id_mapping


def generate_codebook_for_domain(model, embeddings, domain_name, config):
    model.eval()
    model.set_adapter(domain_name)

    embeddings = embeddings[1:].to(next(model.parameters()).device)
    loader = DataLoader(
        TensorDataset(embeddings),
        batch_size=config['RQ-VAE']['batch_size'],
        shuffle=False
    )

    codes = []
    with torch.no_grad():
        for batch in loader:
            codes.append(model.get_codes(batch[0]).cpu().numpy())

    codes = np.vstack(codes)

    cache_dir = config['data_processing']['cache_dir']
    save_dir = os.path.join(cache_dir, 'AmazonReviews2014', domain_name, 'codebook')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f'{domain_name}_codebook.json'), 'w') as f:
        json.dump({str(i + 1): c.tolist() for i, c in enumerate(codes)}, f, indent=2)


def train_epoch(model, dataloader, optimizer, beta, device):
    model.train()
    total = 0.0

    for batch in dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()

        recon, commit_loss, _ = model(x)
        loss = F.mse_loss(recon, x) + beta * commit_loss

        loss.backward()
        optimizer.step()
        total += loss.item()

    return total / len(dataloader)


def run_training_for_domain(model, domain_name, config, device):
    check_and_prepare_data_single(config['data_processing'], domain_name)
    embeddings, _ = load_embeddings_for_domain(domain_name, config)
    train_data = embeddings[1:]

    model.set_adapter(domain_name)

    optimizer = getattr(torch.optim, config['RQ-VAE']['optimizer'])(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['RQ-VAE']['lr']
    )

    train_np, _ = train_test_split(train_data.numpy(), test_size=0.05, random_state=42)
    loader = DataLoader(
        TensorDataset(torch.tensor(train_np)),
        batch_size=config['RQ-VAE']['batch_size'],
        shuffle=True
    )

    for _ in range(config['RQ-VAE']['epochs']):
        train_epoch(
            model,
            loader,
            optimizer,
            config['RQ-VAE']['beta'],
            device
        )

    save_dir = os.path.join('ckpt', 'rqvae_independent', domain_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)

    generate_codebook_for_domain(model, embeddings, domain_name, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='quantization/rqvae_config.yaml')
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('training', {}).get('device', 'cuda:0'))

    model_cfg = config['RQ-VAE']
    backbone = RQVAE(
        input_size=model_cfg['input_dim'],
        hidden_sizes=model_cfg['hidden_dim'],
        latent_size=model_cfg['latent_dim'],
        num_levels=model_cfg['num_layers'],
        codebook_size=model_cfg['code_book_size'],
        dropout=model_cfg['dropout'],
        latent_loss_weight=model_cfg['beta']
    )

    lora_cfg = LoraConfig(
        r=model_cfg.get('lora_r', 8),
        lora_alpha=model_cfg.get('lora_alpha', 16),
        lora_dropout=model_cfg.get('lora_dropout', 0.1),
        bias='none',
        target_modules=['linear']
    )

    model = get_peft_model(backbone, lora_cfg)
    model.to(device)

    for domain in args.datasets:
        model.add_adapter(domain, lora_cfg)
        model.set_adapter(domain)
        run_training_for_domain(model, domain, config, device)


if __name__ == '__main__':
    main()
