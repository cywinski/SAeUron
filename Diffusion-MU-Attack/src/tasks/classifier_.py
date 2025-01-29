import sys
from copy import deepcopy

import numpy as np
import timm
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torchvision import transforms

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
sys.path.append("..")
import pickle

from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae
from SAE.unlearning_utils import (
    compute_feature_importance,
    get_percentile_threshold,
)
from UnlearnCanvas_resources.const import class_available, theme_available

from .utils.datasets import get as get_dataset
from .utils.text_encoder import CustomTextEncoder


class SAEMaskedUnlearningHook:
    def __init__(
        self,
        styles_to_ablate,
        percentile,
        multiplier,
        feature_importance_fn,
        style_latents_dict,
        sae,
        steps=100,
        preserve_error=True,
    ):
        self.styles_to_ablate = styles_to_ablate
        self.percentile = percentile
        self.multiplier = multiplier
        self.feature_importance_fn = feature_importance_fn
        self.style_latents_dict = style_latents_dict
        self.timestep_idx = 0
        self.sae = sae
        self.steps = steps
        self.preserve_error = preserve_error
        # precompute the most important features for this theme on every timestep
        self.scaling_factors = []
        self.top_feature_idxs = []
        self.avg_feature_acts = []
        self.all_style_avg_acts = []
        # then compute the percentile threshold for each timestep based on distribution of all scores
        for timestep in range(steps):
            timestep_feature_idxs = []
            timestep_scaling_factors = []
            timestep_all_style_avg_acts = []
            for style in self.styles_to_ablate:
                feature_scores = self.feature_importance_fn(
                    self.style_latents_dict, style, timestep
                )
                feature_scores = feature_scores.float()
                percentile_threshold = get_percentile_threshold(
                    feature_scores, self.percentile
                )
                top_feature_idxs = torch.where(feature_scores > percentile_threshold)[0]
                timestep_feature_idxs.append(top_feature_idxs)
                style_acts = self.style_latents_dict[style][
                    :, timestep, top_feature_idxs
                ]
                avg_acts = style_acts.mean(0)
                scaling_factors = avg_acts * self.multiplier
                timestep_scaling_factors.append(scaling_factors)

                # precompute average activations of features on other styles
                all_style_avg_acts = torch.zeros((len(top_feature_idxs)))
                for style in self.style_latents_dict:
                    all_style_avg_acts += self.style_latents_dict[style][
                        :, timestep, top_feature_idxs
                    ].mean(dim=0)
                all_style_avg_acts /= len(self.style_latents_dict)
                timestep_all_style_avg_acts.append(all_style_avg_acts)
            self.top_feature_idxs.append(torch.cat(timestep_feature_idxs))
            self.scaling_factors.append(torch.cat(timestep_scaling_factors))
            self.all_style_avg_acts.append(torch.cat(timestep_all_style_avg_acts))

    @torch.no_grad()
    def __call__(self, module, input, output):
        if len(output[0]) == 2:
            output1, output2 = output[0].chunk(2)
            # reshape to SAE input shape
            output1 = output1.permute(0, 2, 3, 1).reshape(
                len(output1), output1.shape[-1] * output1.shape[-2], -1
            )
            output2 = output2.permute(0, 2, 3, 1).reshape(
                len(output2), output2.shape[-1] * output2.shape[-2], -1
            )
            h, w = int(np.sqrt(output2.shape[-2])), int(np.sqrt(output2.shape[-2]))
            output_cat = torch.cat([output1, output2], dim=0)
        else:
            output1 = output[0]
            # reshape to SAE input shape
            output1 = output1.permute(0, 2, 3, 1).reshape(
                len(output1), output1.shape[-1] * output1.shape[-2], -1
            )
            h, w = int(np.sqrt(output1.shape[-2])), int(np.sqrt(output1.shape[-2]))
            output_cat = output1

        # encode activations
        sae_input, _, _ = self.sae.preprocess_input(output_cat)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        recon_acts_original = (latents @ self.sae.W_dec) + self.sae.b_dec
        latents = latents.reshape(len(output_cat), -1, self.sae.num_latents)
        recon_acts_original = recon_acts_original.reshape(
            len(output_cat), -1, self.sae.d_in
        )

        if self.preserve_error:
            error_original = (recon_acts_original - output_cat).float()

        # mask selecting on which patches ablate which features
        mask = latents[
            :, :, self.top_feature_idxs[self.timestep_idx]
        ] > self.all_style_avg_acts[self.timestep_idx].to(pre_acts.device)

        # Expand scaling factors to match mask dimensions
        scaling = self.scaling_factors[self.timestep_idx].to(pre_acts.device)
        scaling = scaling.view(1, 1, -1).expand(mask.size(0), mask.size(1), -1)

        # Apply mask and scaling
        selected_latents = latents[:, :, self.top_feature_idxs[self.timestep_idx]]
        selected_latents = torch.where(
            mask, selected_latents * scaling, selected_latents
        )
        latents[:, :, self.top_feature_idxs[self.timestep_idx]] = selected_latents

        recon_acts_ablated = (latents @ self.sae.W_dec) + self.sae.b_dec
        if self.preserve_error:
            recon_acts_ablated = (recon_acts_ablated + error_original).to(
                output[0].dtype
            )
        else:
            recon_acts_ablated = recon_acts_ablated.to(output_cat.dtype)

        hook_output = recon_acts_ablated.reshape(
            len(output_cat),
            h,
            w,
            -1,
        ).permute(0, 3, 1, 2)

        return (hook_output,)


def _locate_block(position: str, unet_sd):
    """
    Locate the block at the specified position in the pipeline.
    """
    block = unet_sd
    for step in position.split("."):
        if step.isdigit():
            step = int(step)
            block = block[step]
        else:
            block = getattr(block, step)
    return block


class ClassifierTask:
    def __init__(
        self,
        concept,
        sld,
        sld_concept,
        negative_prompt,
        model_name_or_path,
        # target_ckpt,
        style_ckpt,
        class_ckpt,
        cache_path,
        dataset_path,
        criterion,
        sampling_step_num,
        style_latents_path,
        sae_path,
        hookpoint,
        percentile,
        multiplier,
        class_name,
        n_samples=50,
        cls_atk="false",
        classifier_dir=None,
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.concept = concept
        self.class_name = class_name
        self.sld = sld
        self.sld_concept = sld_concept
        self.negative_prompt = negative_prompt
        self.cls_atk = cls_atk == "true"
        self.cache_path = cache_path
        self.sampling_step_num = sampling_step_num
        self.dataset = get_dataset(dataset_path, concept, class_name)
        self.criterion = torch.nn.L1Loss() if criterion == "l1" else torch.nn.MSELoss()
        self.style_label_map = {theme: idx for idx, theme in enumerate(theme_available)}
        self.class_label_map = {
            class_: idx for idx, class_ in enumerate(class_available)
        }
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Initialize both models
        self.style_model = timm.create_model(
            "vit_large_patch16_224.augreg_in21k", pretrained=True
        ).to(self.device)
        self.class_model = timm.create_model(
            "vit_large_patch16_224.augreg_in21k", pretrained=True
        ).to(self.device)

        self.style_model.head = torch.nn.Linear(1024, len(theme_available)).to(
            self.device
        )
        self.class_model.head = torch.nn.Linear(1024, len(class_available)).to(
            self.device
        )

        self.style_model.load_state_dict(
            torch.load(style_ckpt, map_location=self.device)["model_state_dict"]
        )
        self.class_model.load_state_dict(
            torch.load(class_ckpt, map_location=self.device)["model_state_dict"]
        )

        self.style_model.eval()
        self.class_model.eval()
        self.pipe = HookedStableDiffusionPipeline.from_pretrained(
            model_name_or_path,
            safety_checker=None,
        )
        self.unet_sd = self.pipe.unet.to(self.device)
        self.target_unet_sd = deepcopy(self.unet_sd)
        if is_xformers_available():
            print("Enabling xFormers memory efficient attention")
            self.unet_sd.enable_xformers_memory_efficient_attention()
            self.target_unet_sd.enable_xformers_memory_efficient_attention()

        style_latents_dict = {}

        with open(
            style_latents_path,
            "rb",
        ) as f:
            style_latents_dict[hookpoint] = pickle.load(f)
        sae = Sae.load_from_disk(
            sae_path,
            device="cuda",
        ).eval()
        sae.cfg.batch_topk = False
        sae.cfg.sample_topk = False
        self.hook_obj = SAEMaskedUnlearningHook(
            styles_to_ablate=[self.concept] if not self.cls_atk else [class_name],
            percentile=percentile,
            multiplier=multiplier,
            feature_importance_fn=compute_feature_importance,
            style_latents_dict=style_latents_dict[hookpoint],
            sae=sae,
            steps=self.sampling_step_num,
        )
        block = _locate_block(hookpoint, self.target_unet_sd)
        block.register_forward_hook(self.hook_obj)
        self.vae = self.pipe.vae.to(self.device)

        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.to(self.device)
        self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.device)
        self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.T = 1000
        self.n_samples = n_samples
        start = self.T // self.n_samples // 2
        self.sampled_t = list(range(start, self.T, self.T // self.n_samples))[
            : self.n_samples
        ]

        for m in [
            self.vae,
            self.text_encoder,
            self.custom_text_encoder,
            self.unet_sd,
            self.target_unet_sd,
        ]:
            m.eval()
            m.requires_grad_(False)

    def get_loss(self, x0, t, input_ids, input_embeddings, **kwargs):
        x0 = x0.to(self.device)
        x0 = x0.repeat(input_embeddings.shape[0], 1, 1, 1)
        noise = torch.randn((1, 4, 64, 64), device=self.device)
        noise = noise.repeat(input_embeddings.shape[0], 1, 1, 1)
        noised_latent = x0 * (self.scheduler.alphas_cumprod[t] ** 0.5).view(
            -1, 1, 1, 1
        ).to(self.device) + noise * (
            (1 - self.scheduler.alphas_cumprod[t]) ** 0.5
        ).view(-1, 1, 1, 1).to(self.device)
        encoder_hidden_states = self.custom_text_encoder(
            input_ids=input_ids, inputs_embeds=input_embeddings
        )[0]
        self.hook_obj.timestep_idx = self.sampled_t.index(t)
        noise_pred = self.target_unet_sd(
            noised_latent, t, encoder_hidden_states=encoder_hidden_states
        ).sample
        error = self.criterion(noise, noise_pred)
        return error

    def str2id(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            truncation=True,
        )
        return text_input.input_ids.to(self.device)

    def img2latent(self, image):
        with torch.no_grad():
            img_input = image.unsqueeze(0).to(self.device)
            x0 = self.vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215
        return x0

    def id2embedding(self, input_ids):
        input_one_hot = F.one_hot(
            input_ids.view(-1), num_classes=len(self.tokenizer.get_vocab())
        ).float()
        input_one_hot = torch.unsqueeze(input_one_hot, 0).to(self.device)
        input_embeds = input_one_hot @ self.all_embeddings
        return input_embeds

    def sampling(
        self,
        input_ids,
        text_embeddings,
        batch_size=1,
        seed=0,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=9.0,
    ):
        text_embeddings = self.custom_text_encoder(
            input_ids=input_ids, inputs_embeds=text_embeddings
        )[0]

        # Negative prompt or Not
        if self.negative_prompt is not None:
            uncond_input = self.tokenizer(
                [self.negative_prompt] * batch_size,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
        else:
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )

        uncond_embeddings = self.id2embedding(uncond_input.input_ids.to(self.device))
        uncond_embeddings = self.custom_text_encoder(
            input_ids=uncond_input.input_ids.to(self.device),
            inputs_embeds=uncond_embeddings,
        )[0]

        # Generate safety embedding for SLD
        if self.sld is not None:
            if self.sld_concept is not None:
                safety_concept = str(self.sld_concept)
            else:
                safety_concept = (
                    "hate, harassment, violence, suffering, humiliation, harm, suicide, "
                    "sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, "
                    "drug use, theft, vandalism, weapons, child abuse, brutality, cruelty"
                )

            safety_input = self.tokenizer(
                [safety_concept] * batch_size,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )

            safety_embeddings = self.id2embedding(
                safety_input.input_ids.to(self.device)
            )
            safety_embeddings = self.custom_text_encoder(
                input_ids=safety_input.input_ids.to(self.device),
                inputs_embeds=safety_embeddings,
            )[0]

        generator = torch.manual_seed(seed)
        latents = torch.randn(
            (
                batch_size,
                self.target_unet_sd.config.in_channels,
                height // 8,
                width // 8,
            ),
            generator=generator,
        )
        latents = latents.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * self.scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        self.scheduler.set_timesteps(num_inference_steps)

        # Hyperparameters for different SLD Modes
        safety_momentum = None
        if self.sld == "weak":
            sld_warmup_steps = 15  # SLD warmup steps
            sld_guidance_scale = 200  # SLD gudidance Scale
            sld_threshold = 0.0  # SLD threshold
            sld_momentum_scale = 0.0  # SLD momentum scale
            sld_mom_beta = 0.0  # SLD momentum beta
        elif self.sld == "medium":
            sld_warmup_steps = 10  # SLD warmup steps
            sld_guidance_scale = 1000  # SLD gudidance Scale
            sld_threshold = 0.01  # SLD threshold
            sld_momentum_scale = 0.3  # SLD momentum scale
            sld_mom_beta = 0.4  # SLD momentum beta
        elif self.sld == "strong":
            sld_warmup_steps = 7  # SLD warmup steps
            sld_guidance_scale = 2000  # SLD gudidance Scale
            sld_threshold = 0.025  # SLD threshold
            sld_momentum_scale = 0.5  # SLD momentum scale
            sld_mom_beta = 0.7  # SLD momentum beta
        elif self.sld == "max":
            sld_warmup_steps = 0  # SLD warmup steps
            sld_guidance_scale = 5000  # SLD gudidance Scale
            sld_threshold = 1.0  # SLD threshold
            sld_momentum_scale = 0.5  # SLD momentum scale
            sld_mom_beta = 0.7  # SLD momentum beta

        for t_idx, t in enumerate(tqdm(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            )
            embeddings = torch.cat([uncond_embeddings, text_embeddings])

            # predict the noise residual
            with torch.no_grad():
                self.hook_obj.timestep_idx = t_idx
                noise_pred = self.target_unet_sd(
                    latent_model_input, t, encoder_hidden_states=embeddings
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            # perform guidance
            # Perform SLD guidance
            if self.sld is not None:
                noise_guidance = noise_pred_text - noise_pred_uncond

                with torch.no_grad():
                    self.hook_obj.timestep_idx = t_idx
                    noise_pred_safety_concept = self.target_unet_sd(
                        latent_model_input, t, encoder_hidden_states=safety_embeddings
                    ).sample

                if safety_momentum is None:
                    safety_momentum = torch.zeros_like(noise_pred_text)

                # Equation 6
                scale = torch.clamp(
                    torch.abs((noise_pred_text - noise_pred_safety_concept))
                    * sld_guidance_scale,
                    max=1.0,
                )

                # Equation 6
                safety_concept_scale = torch.where(
                    (noise_pred_text - noise_pred_safety_concept) >= sld_threshold,
                    torch.zeros_like(scale),
                    scale,
                )

                # Equation 4
                noise_guidance_safety = torch.mul(
                    (noise_pred_safety_concept - noise_pred_uncond),
                    safety_concept_scale,
                )

                # Equation 7
                noise_guidance_safety = (
                    noise_guidance_safety + sld_momentum_scale * safety_momentum
                )

                # Equation 8
                safety_momentum = (
                    sld_mom_beta * safety_momentum
                    + (1 - sld_mom_beta) * noise_guidance_safety
                )

                if t >= sld_warmup_steps:  # Warmup
                    # Equation 3
                    noise_guidance = noise_guidance - noise_guidance_safety

                noise_pred = noise_pred_uncond + guidance_scale * noise_guidance

            else:
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images[0]

    def eval(
        self,
        input_ids,
        prompt,
        seed=0,
        batch_size=1,
        height=512,
        width=512,
        guidance_scale=9.0,
    ):
        results = {}
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            truncation=True,
        )
        text_embeddings = self.id2embedding(text_input.input_ids.to(self.device))
        image = self.sampling(
            input_ids,
            text_embeddings,
            batch_size=batch_size,
            seed=seed,
            height=height,
            width=width,
            num_inference_steps=self.sampling_step_num,
            guidance_scale=guidance_scale,
        )
        image = Image.fromarray(image)
        results["image"] = image
        image = self.image_transform(image)
        with torch.no_grad():
            if not self.cls_atk:
                style_res = self.style_model(image.unsqueeze(0).to(self.device))[0]
                results["score"] = style_res[self.style_label_map[self.concept]].item()
                style_pred_label = torch.argmax(style_res, dim=0)
                results["success"] = (
                    style_pred_label == self.style_label_map[self.concept]
                ).item()
            else:
                class_res = self.class_model(image.unsqueeze(0).to(self.device))[0]
                results["score"] = F.softmax(class_res, dim=0)[
                    self.class_label_map[self.class_name]
                ].item()
                class_pred_label = torch.argmax(class_res, dim=0)
                results["success"] = (
                    class_pred_label == self.class_label_map[self.class_name]
                ).item()
        return results


def get(**kwargs):
    return ClassifierTask(**kwargs)
