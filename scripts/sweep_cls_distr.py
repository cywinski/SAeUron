"""
Script for hyperparameter sweep for object unlearning.
"""
import os
import pickle
import sys

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from packaging import version
from tqdm import tqdm

import utils.hooks as hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae
from SAE.unlearning_utils import compute_feature_importance

sys.path.append("..")

import fire

from UnlearnCanvas_resources.const import class_available, theme_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

from diffusers.utils.import_utils import is_xformers_available


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_sae(sae_checkpoint, hookpoint, device):
    sae = Sae.load_from_disk(
        os.path.join(sae_checkpoint, hookpoint), device=device
    ).eval()
    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False
    return sae


def main(
    pipe_checkpoint,
    hookpoint,
    class_latents_path,
    sae_checkpoint,
    seed=42,
    steps=100,
    percentiles=[99.99, 99.995, 99.999],
    multipliers=[-1.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0],
    guidance_scale=9.0,
    output_dir="sweep_results/mu_results/class20/",
    limit_themes=50,
):
    accelerator = Accelerator()
    device = accelerator.device

    model = HookedStableDiffusionPipeline.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    model = model.to(device)

    if is_xformers_available():
        import xformers

        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if accelerator.is_main_process:
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        model.enable_xformers_memory_efficient_attention()

    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sae = load_sae(sae_checkpoint, hookpoint, device)
    with open(
        class_latents_path,
        "rb",
    ) as f:
        class_latents_dict = pickle.load(f)

    class_prompt_dict = {class_: [] for class_ in class_available}
    for class_to_unlearn in class_available:
        with open(
            os.path.join(
                "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                f"sd_prompt_{class_to_unlearn}.txt",
            ),
            "r",
        ) as prompt_file:
            prompts = prompt_file.readlines()
            for i, theme in enumerate(theme_available):
                if i >= limit_themes:
                    break
                if theme != "Seed_Images":
                    theme_prompt = prompts[i]
                    theme_prompt = theme_prompt.strip()
                    theme_prompt = (
                        theme_prompt
                        if not theme_prompt.endswith(".")
                        else theme_prompt[:-1]
                    )
                    theme_prompt = f"{theme_prompt} in {theme.replace('_', ' ')} style."
                    class_prompt_dict[class_to_unlearn].append(theme_prompt)

    progress_bar = tqdm(
        total=len(multipliers) * len(class_available) * len(percentiles),
        disable=not accelerator.is_main_process,
    )
    for multiplier in multipliers:
        for percentile in percentiles:
            for class_to_unlearn in class_available:
                if accelerator.is_main_process:
                    progress_bar.set_description(
                        f"Multiplier: {multiplier} Percentile: {percentile} Class: {class_to_unlearn}"
                    )
                output_path = os.path.join(
                    output_dir,
                    f"percentile_{percentile}_multiplier_{multiplier}/{class_to_unlearn}",
                )
                os.makedirs(output_path, exist_ok=True)
                all_prompts = [
                    (class_name, prompt)
                    for class_name, prompts in class_prompt_dict.items()
                    for prompt in prompts
                ]
                input_classes = []
                with accelerator.split_between_processes(all_prompts) as local_tuples:
                    local_prompts = [prompt.strip() for _, prompt in local_tuples]
                    local_classes = [class_name for class_name, _ in local_tuples]
                    steering_hooks = {}
                    steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                        concept_to_unlearn=[class_to_unlearn],
                        percentile=percentile,
                        multiplier=multiplier,
                        feature_importance_fn=compute_feature_importance,
                        concept_latents_dict=class_latents_dict,
                        sae=sae,
                        steps=steps,
                        preserve_error=True,
                    )
                    with torch.no_grad():
                        images = model.run_with_hooks(
                            prompt=local_prompts,
                            generator=generator,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            position_hook_dict=steering_hooks,
                        )
                    input_classes.extend(local_classes)
                accelerator.wait_for_everyone()
                images = gather_object(images)
                input_classes = gather_object(input_classes)
                if accelerator.is_main_process:
                    for i, (img, object_class) in enumerate(zip(images, input_classes)):
                        img.save(
                            os.path.join(
                                output_path,
                                f"{object_class}_seed{seed}_{i}.jpg",
                            )
                        )
                if accelerator.is_main_process:
                    progress_bar.update(1)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)
