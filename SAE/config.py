from dataclasses import dataclass

import torch
from simple_parsing import Serializable, list_field


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    batch_topk: bool = False
    """Train Batch-TopK SAEs"""

    sample_topk: bool = False
    """Take TopK latents per whole generated sample, not only per patch of the feature map"""

    input_unit_norm: bool = False

    multi_topk: bool = False
    """Use Multi-TopK loss."""


@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig
    dataset_path: list[str] = list_field()

    effective_batch_size: int = 4096
    """Number of activation vectors in a batch."""

    num_workers: int = 1

    persistent_workers: bool = True
    prefetch_factor: int = 2

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for SAE training."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_scheduler: str = "constant"
    """
        Scheduler types:
       - "linear" = get_linear_schedule_with_warmup
       - "cosine" = get_cosine_schedule_with_warmup
       - "cosine_with_restarts" = get_cosine_with_hard_restarts_schedule_with_warmup
       - "polynomial" = get_polynomial_decay_schedule_with_warmup
       - "constant" =  get_constant_schedule
       - "constant_with_warmup" = get_constant_schedule_with_warmup
       - "inverse_sqrt" = get_inverse_sqrt_schedule
       - "reduce_lr_on_plateau" = get_reduce_on_plateau_schedule
       - "cosine_with_min_lr" = get_cosine_with_min_lr_schedule_with_warmup
       - "warmup_stable_decay" = get_wsd_schedule
    """

    lr_warmup_steps: int = 1000

    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    feature_sampling_window: int = 100

    hookpoints: list[str] = list_field()
    """List of hookpoints to train SAEs on."""

    distribute_modules: bool = False
    """Store a single copy of each SAE, instead of copying them across devices."""

    save_every: int = 5000
    """Save SAEs every `save_every` steps."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1
    wandb_project: str = "sae_stable-diffusion-v1-4"

    def __post_init__(self):
        if self.run_name is None:
            variant = "patch_topk"
            if self.sae.batch_topk:
                variant = "batch_topk"
            elif self.sae.sample_topk:
                variant = "sample_topk"
            self.run_name = f"{variant}_expansion_factor{self.sae.expansion_factor}_k{self.sae.k}_multi_topk{self.sae.multi_topk}_auxk_alpha{self.auxk_alpha}"


@dataclass
class CacheActivationsRunnerConfig:
    hook_names: list[str] | None = None
    new_cached_activations_path: str | None = None
    dataset_name: str = "guangyil/laion-coco-aesthetic"
    split: str = "train"
    column: str = "caption"
    device: torch.device | str = "cuda"
    model_name: str = "CompVis/stable-diffusion-v1-4"
    dtype: torch.dtype = torch.float16
    num_inference_steps: int = 50
    seed: int = 42
    batch_size: int = 100
    num_workers: int = 8
    output_or_diff: str = "output"
    max_num_examples: int | None = None
    cache_every_n_timesteps: int = 1
    guidance_scale: float = 9.0
    class_start: int = 0
    class_end: int = 20

    hf_repo_id: str | None = None
    hf_num_shards: int | None = None
    hf_revision: str = "main"
    hf_is_private_repo: bool = False

    def __post_init__(self):
        if self.new_cached_activations_path is None:
            self.new_cached_activations_path = f"activations/{self.dataset_name.split('/')[-1]}/{self.model_name.split('/')[-1]}/{self.output_or_diff}/"
        if isinstance(self.hook_names, str):
            self.hook_names = [self.hook_names]
