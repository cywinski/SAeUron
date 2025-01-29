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

from UnlearnCanvas_resources.const import (
    class_available,
    theme_available,
)

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
    style_latents_path,
    sae_checkpoint,
    seed=188,
    steps=100,
    percentile=99.999,
    multiplier=-1.0,
    guidance_scale=9.0,
    output_dir="eval_results/mu_results/style50/",
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
        style_latents_path,
        "rb",
    ) as f:
        style_latents_dict = pickle.load(f)

    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    progress_bar = tqdm(
        theme_avail, total=len(theme_avail), disable=not accelerator.is_main_process
    )
    for theme_to_unlearn in progress_bar:
        if accelerator.is_main_process:
            progress_bar.set_description(f"Unlearning {theme_to_unlearn}")
        output_path = os.path.join(
            output_dir,
            f"percentile_{percentile}_multiplier_{multiplier}/{theme_to_unlearn}",
        )
        os.makedirs(output_path, exist_ok=True)
        for test_theme in theme_avail:
            input_classes = []
            input_themes = []
            class_theme_pairs = [(c, test_theme) for c in class_available] + [
                (c, "") for c in class_available
            ]
            with accelerator.split_between_processes(
                class_theme_pairs
            ) as local_classes_themes:
                local_prompts = []
                for object_class, theme in local_classes_themes:
                    if theme == "":
                        local_prompts.append(f"An image of {object_class}.")
                    else:
                        local_prompts.append(
                            f"An image of {object_class} in {theme.replace('_', ' ')} style."
                        )
                steering_hooks = {}
                steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                    concept_to_unlearn=[theme_to_unlearn],
                    percentile=percentile,
                    multiplier=multiplier,
                    feature_importance_fn=compute_feature_importance,
                    concept_latents_dict=style_latents_dict,
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
                for object_class, theme in local_classes_themes:
                    input_classes.extend([object_class])
                    input_themes.extend([theme])
            accelerator.wait_for_everyone()
            images = gather_object(images)
            input_classes = gather_object(input_classes)
            input_themes = gather_object(input_themes)
            if accelerator.is_main_process:
                for img, object_class, theme in zip(
                    images, input_classes, input_themes
                ):
                    if theme == "":
                        img.save(
                            os.path.join(
                                output_path,
                                f"{object_class}_seed{seed}.jpg",
                            )
                        )
                    else:
                        img.save(
                            os.path.join(
                                output_path,
                                f"{theme}_{object_class}_seed{seed}.jpg",
                            )
                        )
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)
