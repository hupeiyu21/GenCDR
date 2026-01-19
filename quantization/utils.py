import json
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import trange
import sys

def calc_cos_sim(model, data, config):
    if len(data.shape) > 2:
        data = data[:, 0, :]
    ids = model.get_codes(data).cpu().numpy()
    max_item_calculate = 1000
    cos_sim_array = np.zeros(config["num_layers"])

    for n_prefix in range(1, config["num_layers"] + 1):
        unique_prefix = np.unique(ids[:, :n_prefix], axis=0)
        this_level_cos_sim_within_cluster = []

        for this_level_prefix in unique_prefix:
            mask = (ids[:, :n_prefix] == this_level_prefix).all(axis=1)
            this_cluster = data[mask].cpu()
            this_cluster_num = this_cluster.shape[0]

            if this_cluster_num > 1:
                indice = torch.randperm(this_cluster_num)[:max_item_calculate]
                cos_sim = F.cosine_similarity(
                    this_cluster[indice, :, None],
                    this_cluster.t()[None, :, indice]
                )
                cos_sim_sum = torch.tril(cos_sim, diagonal=-1).sum()
                normalization_factor = (this_cluster_num - 1) * this_cluster_num / 2
                this_level_cos_sim_within_cluster.append(
                    cos_sim_sum.item() / normalization_factor
                )

        if this_level_cos_sim_within_cluster:
            cos_sim_array[n_prefix - 1] = np.mean(this_level_cos_sim_within_cluster)

    return cos_sim_array


def process_embeddings(config, device, id2meta_file=None, embedding_save_path=None):
    category = config["dataset"]["name"]
    type = config["dataset"]["type"]
    final_output_path = os.path.join("cache", type, category, "processed", "final_pca_embeddings.npy")

    if not os.path.exists(final_output_path):
        raise FileNotFoundError(f"Embedding file not found: {final_output_path}")

    np_array = np.load(final_output_path)
    tensor = torch.from_numpy(np_array).to(device, dtype=torch.float32)
    print(f"[QUANTIZATION] Loaded embeddings from '{final_output_path}', shape={tensor.shape}, dtype={tensor.dtype}")
    return tensor


def set_weight_decay(optimizer, weight_decay):
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = weight_decay