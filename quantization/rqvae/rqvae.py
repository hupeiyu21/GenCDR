import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from .layers import MLP


class RQVAE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        latent_size,
        num_levels,
        codebook_size,
        dropout,
        decay=0.99,
        loss_type="mse",
        latent_loss_weight=0.25,
        bottleneck_type="rq",
        **kwargs
    ):
        super().__init__()

        assert loss_type in ["mse", "l1"]

        self.encoder = MLP(input_size, hidden_sizes, latent_size, dropout=dropout)
        hidden_sizes.reverse()
        self.decoder = MLP(latent_size, hidden_sizes, input_size, dropout=dropout)

        if bottleneck_type == "rq":
            code_shape = [codebook_size] * num_levels
            self.quantizer = RQBottleneck(
                latent_shape=latent_size,
                code_shape=code_shape,
                decay=decay,
                shared_codebook=False,
                restart_unused_codes=True,
            )
            self.code_shape = code_shape
        else:
            raise ValueError("invalid 'bottleneck_type' (must be 'rq')")

        self.loss_type = loss_type
        self.latent_loss_weight = latent_loss_weight
        self.num_levels = num_levels
        self.codebook_size = codebook_size

    def forward(self, xs):
        z_e = self.encode(xs)
        z_q, quant_loss, code = self.quantizer(z_e)
        out = self.decode(z_q)
        return out, quant_loss, code

    def encode(self, x):
        z_e = self.encoder(x)
        return z_e

    def decode(self, z_q):
        out = self.decoder(z_q)
        return out

    @torch.no_grad()
    def get_codes(self, xs):
        z_e = self.encode(xs)
        _, _, code = self.quantizer(z_e)
        return code

    def compute_loss(self, out, quant_loss, code, xs=None, valid=False):

        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("incompatible loss type")

        loss_latent = quant_loss

        if valid:
            loss_recon = loss_recon * xs.shape[0] * xs.shape[1]
            loss_latent = loss_latent * xs.shape[0]

        loss_total = loss_recon + self.latent_loss_weight * loss_latent

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_latent": loss_latent,
            "codes": [code],
        }
    
    @staticmethod
    def add_lora_adapter(model, lora_config):

        for param in model.parameters():
            param.requires_grad = False

        config = LoraConfig(
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("lora_alpha", 16),
            lora_dropout=lora_config.get("lora_dropout", 0.1),
            bias="none",
            target_modules=lora_config.get("target_modules", ["Linear"]),
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        lora_model = get_peft_model(model, config)
        
        print("LoRA Adapter added! Trainable parameters:")
        lora_model.print_trainable_parameters()
        
        return lora_model



class VQEmbedding(nn.Embedding):
    """VQ embedding module with ema update."""

    def __init__(
        self,
        n_embed,
        embed_dim,
        ema=True,
        decay=0.99,
        restart_unused_codes=True,
        eps=1e-5,
    ):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            # padding index is not updated by EMA
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(
            *inputs_shape[:-1], -1
        )  # [B, h, w, n_embed or n_embed+1]

        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)  # [B, h, w, n_embed or n_embed+1]
        embed_idxs = distances.argmin(dim=-1)  # use padding index or not

        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):

        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)

        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(
            dim=0, index=idxs.unsqueeze(0), src=vectors.new_ones(1, n_vectors)
        )

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(
            vectors_sum_per_cluster, alpha=1 - self.decay
        )

        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][
                :n_embed
            ]

            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(
                torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1)
            )

    @torch.no_grad()
    def _update_embedding(self):

        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)

        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds


class RQBottleneck(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.

    Arguments:
        latent_shape (Tuple[int, int, int]): the shape of latents, denoted (H, W, D)
        code_shape (Tuple[int, int, int]): the shape of codes, denoted (h, w, d)
        n_embed (int, List, or Tuple): the number of embeddings (i.e., the size of codebook)
            If isinstance(n_embed, int), the sizes of all codebooks are same.
        shared_codebook (bool): If True, codebooks are shared in all location. If False,
            uses separate codebooks along the ``depth'' dimension. (default: False)
        restart_unused_codes (bool): If True, it randomly assigns a feature vector in the curruent batch
            as the new embedding of unused codes in training. (default: True)
    """

    def __init__(
        self,
        latent_shape,
        code_shape,
        decay=0.99,
        shared_codebook=False,
        restart_unused_codes=True,
        commitment_loss="cumsum",
    ):
        super().__init__()

        self.latent_shape = latent_shape
        self.code_shape = code_shape
        self.shared_codebook = shared_codebook
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = [n_embed for n_embed in code_shape]
        self.decay = [decay for _ in range(len(self.code_shape))]

        if self.shared_codebook:
            codebook0 = VQEmbedding(
                self.n_embed[0],
                self.latent_shape,
                decay=self.decay[0],
                restart_unused_codes=restart_unused_codes,
            )
            self.codebooks = nn.ModuleList(
                [codebook0 for _ in range(self.code_shape[-1])]
            )
        else:
            codebooks = [
                VQEmbedding(
                    self.n_embed[idx],
                    latent_shape,
                    decay=self.decay[idx],
                    restart_unused_codes=restart_unused_codes,
                )
                for idx in range(len(self.code_shape))
            ]
            self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss

    def quantize(self, x):
        r"""
        Return list of quantized features and the selected codewords by the residual quantization.
        The code is selected by the residuals between x and quantized features by the previous codebooks.

        Arguments:
            x (Tensor): bottleneck feature maps to quantize.

        Returns:
            quant_list (list): list of sequentially aggregated and quantized feature maps by codebooks.
            codes (LongTensor): codewords index, corresponding to quants.

        Shape:
            - x: (B, h, w, embed_dim)
            - quant_list[i]: (B, h, w, embed_dim)
            - codes: (B, h, w, d)
        """
        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(len(self.code_shape)):
            quant, code = self.codebooks[i](residual_feature)

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))

        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def forward(self, x):
        quant_list, codes = self.quantize(x)

        commitment_loss = self.compute_commitment_loss(x, quant_list)
        quants_trunc = quant_list[-1]
        quants_trunc = x + (quants_trunc - x).detach()

        return quants_trunc, commitment_loss, codes

    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []

        for idx, quant in enumerate(quant_list):
            partial_loss = (x - quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)

        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss

    @torch.no_grad()
    def embed_code(self, code):
        assert code.shape[1:] == self.code_shape

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [
                self.codebooks[0].embed(code_slice)
                for i, code_slice in enumerate(code_slices)
            ]
        else:
            embeds = [
                self.codebooks[i].embed(code_slice)
                for i, code_slice in enumerate(code_slices)
            ]

        embeds = torch.cat(embeds, dim=-2).sum(-2)

        return embeds

class RQVAEForPEFT(RQVAE):
    """
    A wrapper class that inherits from the original RQVAE, specifically designed for PEFT library.
    Its sole purpose is to provide a forward method signature compatible with PEFT.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x=None, input_ids=None, **kwargs):
        """
        This new forward method can accept the 'input_ids' keyword argument passed by PEFT.
        """
        # If PEFT passes input_ids, we treat it as x
        if input_ids is not None:
            x = input_ids
        
        # If x is still None, it means there's an issue with the call, raise an error
        if x is None:
            raise ValueError("RQVAEForPEFT.forward() needs 'x' or 'input_ids' argument.")

        # Call the parent class's forward method (i.e., the original RQVAE) with the correct x parameter
        return super().forward(x)
