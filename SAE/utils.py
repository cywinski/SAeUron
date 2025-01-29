import inspect
import os
from typing import Any, Type, TypeVar, cast

import torch
from accelerate.utils import send_to_device
from torch import Tensor, nn
from transformers import PreTrainedModel

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: torch.device | str | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def randn_tensor(
    shape: tuple[int, ...],
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = (
            generator.device.type
            if not isinstance(generator, list)
            else generator[0].device.type
        )
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(
                f"Cannot generate a {device} tensor from a generator of type {gen_device_type}."
            )

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(
                shape,
                generator=generator[i],
                device=rand_device,
                dtype=dtype,
                layout=layout,
            )
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(
            shape, generator=generator, device=rand_device, dtype=dtype, layout=layout
        ).to(device)

    return latents


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / torch.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if torch.norm(guess - prev) < tol:
            print("Early stopping in computation of the geometric median")
            break

    return guess


# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return TritonDecoder.apply(top_indices, top_acts, W_dec)


try:
    from .kernels import TritonDecoder
except ImportError:
    decoder_impl = eager_decode
    print("Triton not installed, using eager implementation of SAE decoder.")
else:
    if os.environ.get("SAE_DISABLE_TRITON") == "1":
        print("Triton disabled, using eager implementation of SAE decoder.")
        decoder_impl = eager_decode
    else:
        decoder_impl = triton_decode
