import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import SaeConfig
from .utils import decoder_impl


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    l0_loss: Tensor
    """Number of nonzero latents after ReLU"""

    l2_loss: Tensor

    auxk_loss: Tensor
    """AuxK loss, if applicable."""

    multi_topk_fvu: Tensor
    """Multi-TopK FVU, if applicable."""

    explained_variance: Tensor


class Sae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.encoder = nn.Linear(
            d_in,
            self.num_latents,
            device=device,
            dtype=dtype,
        )
        self.encoder.bias.data.zero_()
        self.W_dec = (
            nn.Parameter(self.encoder.weight.data[:, :d_in].clone())
            if decoder
            else None
        )
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "Sae"]:
        """Load SAEs for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: Sae.load_from_disk(
                    repo_path / layer, device=device, decoder=decoder
                )
                for layer in natsorted(layers)
            }
        files = [
            f
            for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.name: Sae.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{hookpoint}/*" if hookpoint is not None else None,
            )
        )
        if hookpoint is not None:
            repo_path = repo_path / hookpoint

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return Sae.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SaeConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = Sae(d_in, cfg, device=device, decoder=decoder)
        load_model(
            model=sae,
            filename=str(path / "sae.safetensors"),
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
            strict=decoder,
        )
        sae.cfg.batch_topk = False
        sae.cfg.sample_topk = False
        return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        # Remove decoder bias as per Anthropic
        sae_in = x.to(self.dtype) - self.b_dec
        out = self.encoder(sae_in)

        return nn.functional.relu(out)

    def pre_acts_without_relu(self, x: Tensor) -> Tensor:
        # Remove decoder bias as per Anthropic
        sae_in = x.to(self.dtype) - self.b_dec
        out = self.encoder(sae_in)

        return out

    def select_topk(self, latents: Tensor, k=None, batch_size=None) -> EncoderOutput:
        """Select the top-k latents."""
        # latents shape: [bs * sample_size, num_latents]
        if k is None:
            k = self.cfg.k
        if self.cfg.batch_topk:
            # BatchTopK: Select k*bs latents per all patches in a batch
            # Total activated latents in batch: bs*sample_size*k
            flatten_latents = latents.flatten()
            total_k = k * latents.shape[0]
            top_acts_flatten, top_indices_flatten = flatten_latents.topk(
                total_k, sorted=False
            )
            top_acts = (
                torch.zeros_like(flatten_latents, device=latents.device)
                .scatter(-1, top_indices_flatten, top_acts_flatten)
                .reshape(latents.shape)
            )
            top_indices_flatten = top_indices_flatten % self.num_latents
            top_indices = top_indices_flatten.reshape(
                latents.shape[0], k
            )  # NOTE: This will not be proper assignment of latents per sample, but per batch it will be correct
            return EncoderOutput(
                top_acts=top_acts,
                top_indices=top_indices,
            )
        elif self.cfg.sample_topk:
            # SampleTopK: Select k latents per sample
            # Total activated latents in batch: bs*k
            # [bs * sample_size, num_latents] -> [bs, sample_size * num_latents]
            sample_latents = latents.view(batch_size, -1)
            top_acts_sample, top_indices_sample = sample_latents.topk(
                k, sorted=False
            )  # Take top k along num_latents dimension [bs, k]
            top_acts = (
                torch.zeros_like(sample_latents)
                .scatter_(-1, top_indices_sample, top_acts_sample)
                .reshape(latents.shape)
            )
            top_indices = top_indices_sample % self.num_latents
            return EncoderOutput(
                top_acts=top_acts,
                top_indices=top_indices,
            )

        # PatchTopK: Select k latents per patch
        # Total activated latents in batch: bs*sample_size*k
        return EncoderOutput(*latents.topk(k, sorted=False))

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        batch_size, sample_size, emb_size = x.shape

        x = x.reshape(batch_size * sample_size, emb_size)

        return self.select_topk(self.pre_acts(x))

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        if self.cfg.batch_topk or self.cfg.sample_topk:
            # if batch TopK top_acts are already scattered
            y = top_acts.to(self.dtype) @ self.W_dec
        else:
            y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    def preprocess_input(self, x):
        batch_size, sample_size, emb_size = x.shape
        x = x.reshape(batch_size * sample_size, emb_size)
        if self.cfg.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.input_unit_norm:
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    def forward(self, x: Tensor, dead_mask: Tensor | None = None) -> ForwardOutput:
        # x shape: [bs, sample_size, d_in]
        batch_size, sample_size, emb_size = x.shape
        x, x_mean, x_std = self.preprocess_input(x)
        pre_acts = self.pre_acts(x)  # [bs * sample_size, num_latents]
        # Decode and compute residual
        top_acts, top_indices = self.select_topk(
            pre_acts, batch_size=batch_size, k=self.cfg.k
        )
        sae_out = self.decode(top_acts, top_indices)
        e = (sae_out - x).float()
        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (x - x.mean(0)).float().pow(2).sum()
        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = self.select_topk(
                auxk_latents, k=k_aux, batch_size=batch_size
            )

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).float().pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l0_loss = (pre_acts > 0).float().sum(-1).mean()
        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        # metrics for explained variance
        per_token_l2_loss = (sae_out - x).pow(2).sum(dim=-1).squeeze()
        per_token_total_variance = (x - x.mean(0)).pow(2).sum(-1)
        explained_variance = 1 - per_token_l2_loss / per_token_total_variance

        if self.cfg.multi_topk:
            top_acts, top_indices = self.select_topk(
                pre_acts, k=4 * self.cfg.k, batch_size=batch_size
            )
            sae_out = self.decode(top_acts, top_indices)

            multi_topk_fvu = (sae_out - x).float().pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)

        sae_out = self.postprocess_output(sae_out, x_mean, x_std)

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            l0_loss,
            l2_loss / (batch_size * sample_size * emb_size),
            auxk_loss,
            multi_topk_fvu,
            explained_variance,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
