from collections import defaultdict
from dataclasses import asdict
from time import time

import psutil
import torch
import torch.distributed as dist
from datasets import Dataset
from matplotlib import pyplot as plt
from safetensors.torch import load_model
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from SAE.utils import geometric_median

from .config import TrainConfig
from .sae import Sae


class SaeTrainer:
    def __init__(self, cfg: TrainConfig, dataset_dict: dict[str, Dataset]):
        self.cfg = cfg
        self.dataset_dict = dataset_dict
        self.num_examples = len(dataset_dict[list(dataset_dict.keys())[0]])
        input_widths = {
            hook: dataset[0]["activations"].shape[-1]
            for hook, dataset in self.dataset_dict.items()
        }
        self.sample_size = dataset_dict[list(dataset_dict.keys())[0]][0][
            "activations"
        ].shape[-2]
        self.distribute_modules()
        device = torch.device(cfg.device)

        self.saes = {
            hook: Sae(input_widths[hook], cfg.sae, device, dtype=cfg.dtype)
            for hook in self.local_hookpoints()
        }
        print(self.saes)
        self.effective_batch_size = self.cfg.effective_batch_size
        self.batch_size = self.effective_batch_size // self.sample_size
        print(f"Batch size: {self.batch_size}")
        self.increment_tokens = (
            self.effective_batch_size
            if not self.saes[self.local_hookpoints()[0]].cfg.sample_topk
            else self.batch_size
        )

        pgs = [
            {
                "params": sae.parameters(),
                # Auto-select LR using 1 / sqrt(d) scaling law from Fig 3 of the paper
                "lr": cfg.lr or 2e-4 / (sae.num_latents / (2**14)) ** 0.5,
            }
            for sae in self.saes.values()
        ]
        # Dedup the learning rates we're using, sort them, round to 2 decimal places
        lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]
        print(f"Learning rates: {lrs}" if len(lrs) > 1 else f"Learning rate: {lrs[0]}")

        if cfg.dtype in {torch.float16, torch.bfloat16}:
            try:
                from bitsandbytes.optim import Adam8bit as Adam

                print("Using 8-bit Adam from bitsandbytes")
            except ImportError:
                print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
                print("Run `pip install bitsandbytes` for less memory usage.")
                from torch.optim import Adam
        else:
            from torch.optim import Adam

            print("Using torch.optim.Adam")
            # from openai repo for Adam in full precision
            for d in pgs:
                d["eps"] = 6.25e-10
                d["fused"] = True

        self.global_step = 0
        self.num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }
        self.optimizer = Adam(pgs)
        self.lr_scheduler = get_scheduler(
            name=cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=(self.num_examples // self.batch_size) * cfg.num_epochs,
        )

    def load_state(self, path: str):
        """Load the trainer state from disk."""
        device = self.cfg.device

        # Load the train state first so we can print the step number
        train_state = torch.load(
            f"{path}/state.pt", map_location=device, weights_only=True
        )
        self.global_step = train_state["global_step"]
        self.num_tokens_since_fired = train_state["num_tokens_since_fired"]

        print(
            f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m"
        )

        lr_state = torch.load(
            f"{path}/lr_scheduler.pt", map_location=device, weights_only=True
        )
        opt_state = torch.load(
            f"{path}/optimizer.pt", map_location=device, weights_only=True
        )
        self.optimizer.load_state_dict(opt_state)
        self.lr_scheduler.load_state_dict(lr_state)

        for name, sae in self.saes.items():
            load_model(sae, f"{path}/{name}/sae.safetensors", device=str(device))

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name,
                    project=self.cfg.wandb_project,
                    config=asdict(self.cfg),
                    save_code=True,
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(
            p.numel() for s in self.saes.values() for p in s.parameters()
        )
        print(f"Number of SAE parameters: {num_sae_params:_}")

        num_batches = (self.num_examples // self.batch_size) * self.cfg.num_epochs

        device = torch.device(self.cfg.device)
        dataloaders = {
            hook: DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                persistent_workers=self.cfg.persistent_workers,
                prefetch_factor=self.cfg.prefetch_factor,
            )
            for hook, ds in self.dataset_dict.items()
        }
        pbar = tqdm(
            desc="Training",
            disable=not rank_zero,
            initial=self.global_step,
            total=num_batches,
        )

        did_fire = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
            for name, sae in self.saes.items()
        }
        num_tokens_in_step = 0
        total_tokens = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        avg_fvu = defaultdict(float)
        avg_l0 = defaultdict(float)
        avg_l2 = defaultdict(float)
        avg_exp_var_mean = defaultdict(float)
        avg_exp_var_std = defaultdict(float)
        avg_multi_topk_fvu = defaultdict(float)
        maybe_wrapped: dict[str, DDP] | dict[str, Sae] = {}
        frac_active_list = []  # track active features

        # Initialize disk I/O stats
        self.initial_disk_io = psutil.disk_io_counters()

        for _ in range(self.cfg.num_epochs):
            for batch_dict in zip(*dataloaders.values()):
                hidden_dict = {}
                start_loading = time()
                for hook, batch in zip(dataloaders.keys(), batch_dict):
                    hidden_dict[hook] = batch["activations"]
                data_loading_time = time() - start_loading

                # Bookkeeping for dead feature detection
                num_tokens_in_step += self.increment_tokens
                total_tokens += self.increment_tokens

                if self.cfg.distribute_modules:
                    hidden_dict = self.scatter_hiddens(hidden_dict)

                for name, hiddens in zip(self.local_hookpoints(), hidden_dict.values()):
                    raw = self.saes[name]  # 'raw' never has a DDP wrapper
                    # On the first iteration, initialize the decoder bias
                    if self.global_step == 0:
                        # NOTE: The all-cat here could conceivably cause an OOM in some
                        # cases, but it's unlikely to be a problem with small world sizes.
                        # We could avoid this by "approximating" the geometric median
                        # across all ranks with the mean (median?) of the geometric medians
                        # on each rank. Not clear if that would hurt performance.
                        hiddens_input = hiddens.view(-1, hiddens.shape[-1])
                        median = geometric_median(self.maybe_all_cat(hiddens_input))
                        median = median.to(raw.device)
                        raw.b_dec.data = median.to(raw.dtype)

                    if not maybe_wrapped:
                        # Wrap the SAEs with Distributed Data Parallel. We have to do this
                        # after we set the decoder bias, otherwise DDP will not register
                        # gradients flowing to the bias after the first step.
                        maybe_wrapped = (
                            {
                                name: DDP(sae, device_ids=[dist.get_rank()])
                                for name, sae in self.saes.items()
                            }
                            if ddp
                            else self.saes
                        )

                    # Make sure the W_dec is still unit-norm
                    if raw.cfg.normalize_decoder:
                        raw.set_decoder_norm_to_unit_norm()

                    acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                    denom = acc_steps * self.cfg.wandb_log_frequency
                    wrapped = maybe_wrapped[name]

                    # Save memory by chunking the activations
                    for chunk_hiddens in hiddens.chunk(self.cfg.micro_acc_steps):
                        chunk_hiddens = chunk_hiddens.to(device)
                        out = wrapped(
                            chunk_hiddens,
                            dead_mask=(
                                self.num_tokens_since_fired[name]
                                > self.cfg.dead_feature_threshold
                                if self.cfg.auxk_alpha > 0
                                else None
                            ),
                        )

                        avg_fvu[name] += float(
                            self.maybe_all_reduce(out.fvu.detach()) / denom
                        )
                        avg_l0[name] += float(
                            self.maybe_all_reduce(out.l0_loss.detach()) / denom
                        )
                        avg_l2[name] += float(
                            self.maybe_all_reduce(out.l2_loss.detach()) / denom
                        )
                        avg_exp_var_mean[name] += float(
                            self.maybe_all_reduce(out.explained_variance.mean().item())
                            / denom
                        )
                        avg_exp_var_std[name] += float(
                            self.maybe_all_reduce(out.explained_variance.std().item())
                            / denom
                        )
                        if self.cfg.auxk_alpha > 0:
                            avg_auxk_loss[name] += float(
                                self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                            )
                        if self.cfg.sae.multi_topk:
                            avg_multi_topk_fvu[name] += float(
                                self.maybe_all_reduce(out.multi_topk_fvu.detach())
                                / denom
                            )

                        loss = (
                            out.fvu
                            + self.cfg.auxk_alpha * out.auxk_loss
                            + out.multi_topk_fvu / 8
                        )
                        loss.div(acc_steps).backward()

                        # Update the did_fire mask
                        did_fire[name][out.latent_indices.flatten()] = True
                        self.maybe_all_reduce(
                            did_fire[name], "max"
                        )  # max is boolean "any"

                    # Clip gradient norm independently for each SAE
                    torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)

                # Check if we need to actually do a training step
                step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)
                if substep == 0:
                    if self.cfg.sae.normalize_decoder:
                        for sae in self.saes.values():
                            sae.remove_gradient_parallel_to_decoder_directions()

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()

                    ###############
                    with torch.no_grad():
                        # Update the dead feature mask
                        for name, counts in self.num_tokens_since_fired.items():
                            counts += num_tokens_in_step
                            counts[did_fire[name]] = 0

                    if (
                        self.cfg.log_to_wandb
                        and (step + 1) % self.cfg.wandb_log_frequency == 0
                    ):
                        info = {}
                        did_fire_counts = {
                            name: mask.sum().item() for name, mask in did_fire.items()
                        }
                        did_fire_percentages = {
                            name: count / self.saes[name].num_latents
                            for name, count in did_fire_counts.items()
                        }

                        for name in self.saes:
                            mask = (
                                self.num_tokens_since_fired[name]
                                > self.cfg.dead_feature_threshold
                            )
                            fire_count = torch.zeros(
                                self.saes[name].num_latents, dtype=torch.long
                            )
                            unique, unique_counts = torch.unique(
                                out.latent_indices.flatten(),
                                return_counts=True,
                            )
                            fire_count[unique] = unique_counts.cpu()
                            frac_active_list.append(fire_count)

                            if len(frac_active_list) > self.cfg.feature_sampling_window:
                                frac_active_in_window = torch.stack(
                                    frac_active_list[
                                        -self.cfg.feature_sampling_window :
                                    ],
                                    dim=0,
                                )
                                feature_sparsity = frac_active_in_window.sum(0) / (
                                    self.cfg.feature_sampling_window
                                    * self.effective_batch_size
                                )
                            else:
                                frac_active_in_window = torch.stack(
                                    frac_active_list, dim=0
                                )
                                feature_sparsity = frac_active_in_window.sum(0) / (
                                    len(frac_active_list) * self.effective_batch_size
                                )

                            log_feature_sparsity = torch.log10(feature_sparsity + 1e-8)

                            info.update(
                                {
                                    f"fvu/{name}": avg_fvu[name],
                                    f"l0/{name}": avg_l0[name],
                                    f"l2/{name}": avg_l2[name],
                                    f"explained_variance/{name}": avg_exp_var_mean[
                                        name
                                    ],
                                    f"explained_variance_std/{name}": avg_exp_var_std[
                                        name
                                    ],
                                    f"dead_pct/{name}": mask.mean(
                                        dtype=torch.float32
                                    ).item(),
                                    f"lr/{name}": self.optimizer.param_groups[0]["lr"],
                                    f"fire_pct/{name}": did_fire_percentages[name],
                                    f"sparsity_below_1e-2/{name}": (
                                        feature_sparsity < 1e-2
                                    )
                                    .float()
                                    .mean()
                                    .item(),
                                    f"sparsity_below_1e-3/{name}": (
                                        feature_sparsity < 1e-3
                                    )
                                    .float()
                                    .mean()
                                    .item(),
                                    f"sparsity_below_1e-4/{name}": (
                                        feature_sparsity < 1e-4
                                    )
                                    .float()
                                    .mean()
                                    .item(),
                                    f"sparsity_below_1e-5/{name}": (
                                        feature_sparsity < 1e-5
                                    )
                                    .float()
                                    .mean()
                                    .item(),
                                    "total_tokens": total_tokens,
                                    "data_load_time": data_loading_time,
                                }
                            )
                            if self.cfg.auxk_alpha > 0:
                                info[f"auxk/{name}"] = avg_auxk_loss[name]
                            if self.cfg.sae.multi_topk:
                                info[f"multi_topk_fvu/{name}"] = avg_multi_topk_fvu[
                                    name
                                ]
                            if (step + 1) % (self.cfg.wandb_log_frequency * 10) == 0:
                                plt.hist(
                                    log_feature_sparsity.tolist(),
                                    bins=50,
                                    color="blue",
                                    alpha=0.7,
                                )
                                plt.title("Feature Density")
                                plt.xlabel("Log Feature Density")
                                plt.tight_layout()
                                info[f"feature_density/{name}"] = wandb.Image(plt.gcf())
                                plt.close()

                        avg_auxk_loss.clear()
                        avg_fvu.clear()
                        avg_l0.clear()
                        avg_l2.clear()
                        avg_exp_var_mean.clear()
                        avg_exp_var_std.clear()
                        avg_multi_topk_fvu.clear()

                        if self.cfg.distribute_modules:
                            outputs = [{} for _ in range(dist.get_world_size())]
                            dist.gather_object(info, outputs if rank_zero else None)
                            info.update(
                                {k: v for out in outputs for k, v in out.items()}
                            )

                        if rank_zero:
                            wandb.log(info, step=step)
                            self.log_disk_io(step=step, denom=denom)

                    # Reset stats for this step
                    with torch.no_grad():
                        num_tokens_in_step = 0
                        for mask in did_fire.values():
                            mask.zero_()

                    if (
                        self.cfg.save_every > 0
                        and (step + 1) % self.cfg.save_every == 0
                    ):
                        self.save()

                self.global_step += 1
                pbar.update()
        self.save()
        pbar.close()

    def local_hookpoints(self) -> list[str]:
        return (
            self.module_plan[dist.get_rank()]
            if self.module_plan
            else self.cfg.hookpoints
        )

    def distribute_modules(self):
        """Prepare a plan for distributing modules across ranks."""
        if not self.cfg.distribute_modules:
            self.module_plan = []
            print(f"Training on modules: {self.cfg.hookpoints}")
            return

        layers_per_rank, rem = divmod(len(self.cfg.hookpoints), dist.get_world_size())
        assert rem == 0, "Number of modules must be divisible by world size"

        # Each rank gets a subset of the layers
        self.module_plan = [
            self.cfg.hookpoints[start : start + layers_per_rank]
            for start in range(0, len(self.cfg.hookpoints), layers_per_rank)
        ]
        for rank, modules in enumerate(self.module_plan):
            print(f"Rank {rank} modules: {modules}")

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x

    def scatter_hiddens(self, hidden_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Scatter & gather the hidden states across ranks."""
        outputs = [
            # Add a new leading "layer" dimension to each tensor
            torch.stack([hidden_dict[hook] for hook in hookpoints], dim=1)
            for hookpoints in self.module_plan
        ]
        local_hooks = self.module_plan[dist.get_rank()]
        shape = next(iter(hidden_dict.values())).shape

        # Allocate one contiguous buffer to minimize memcpys
        buffer = outputs[0].new_empty(
            # The (micro)batch size times the world size
            shape[0] * dist.get_world_size(),
            # The number of layers we expect to receive
            len(local_hooks),
            # All other dimensions
            *shape[1:],
        )

        # Perform the all-to-all scatter
        inputs = buffer.split([len(output) for output in outputs])
        dist.all_to_all([x for x in inputs], outputs)

        # Return a list of results, one for each layer
        return {hook: buffer[:, i] for i, hook in enumerate(local_hooks)}

    def save(self):
        """Save the SAEs to disk."""

        path = (
            f"sae-ckpts/{self.cfg.wandb_project}/{self.cfg.run_name}"
            if self.cfg.run_name
            else f"sae-ckpts/{self.cfg.wandb_project}"
        )
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        if rank_zero or self.cfg.distribute_modules:
            print("Saving checkpoint")

            for hook, sae in self.saes.items():
                assert isinstance(sae, Sae)

                sae.save_to_disk(f"{path}/{hook}")

        if rank_zero:
            torch.save(self.lr_scheduler.state_dict(), f"{path}/lr_scheduler.pt")
            torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
            torch.save(
                {
                    "global_step": self.global_step,
                    "num_tokens_since_fired": self.num_tokens_since_fired,
                },
                f"{path}/state.pt",
            )

            self.cfg.save_json(f"{path}/config.json")

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()

    def log_disk_io(self, step, denom=1):
        """Log disk I/O statistics using psutil and send them to wandb."""
        current_disk_io = psutil.disk_io_counters()
        disk_read_mb = (
            current_disk_io.read_bytes - self.initial_disk_io.read_bytes
        ) / 1e6
        disk_write_mb = (
            current_disk_io.write_bytes - self.initial_disk_io.write_bytes
        ) / 1e6
        disk_read_time_ms = current_disk_io.read_time - self.initial_disk_io.read_time
        disk_write_time_ms = (
            current_disk_io.write_time - self.initial_disk_io.write_time
        )

        # Reset initial stats for the next step
        self.initial_disk_io = current_disk_io

        if self.cfg.log_to_wandb:
            import wandb

            wandb.log(
                {
                    "disk_read_mb": disk_read_mb / denom,
                    "disk_write_mb": disk_write_mb / denom,
                    "disk_read_time_ms": disk_read_time_ms / denom,
                    "disk_write_time_ms": disk_write_time_ms / denom,
                },
                step=step,
            )
