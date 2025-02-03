# SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders
[![arXiv](https://img.shields.io/badge/arXiv-2501.18052-b31b1b.svg?style=flat)](https://arxiv.org/abs/2501.18052)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/bcywinski/SAeUron)


<img src="images/schema_method_nums_new.jpg" width="100%" alt="schema">

> Overview of the SAeUron method.

## 1. 🚀 Setup

### 1.1 Prepare the UnlearnCanvas benchmark
1. Download the dataset from [Google Drive]([text](https://drive.google.com/drive/folders/1-1Sc8h_tGArZv5Y201ugTF0K0D_Xn2lM)).
2. Download [classifier checkpoints](https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61) and [diffusion model checkpoint](https://drive.google.com/drive/folders/18tN-7LuxQ89I-MDSjtB5to2dGHDMHyqb).

### 1.2 Prepare the environment
```bash
pip install uv
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```
### 1.3 Load trained SAEs from Hugging Face Hub
```bash
python scripts/load_from_hub.py --name bcywinski/SAeUron --hookpoint unet.up_blocks.1.attentions.1 --save_path sae-ckpts
```

```bash
python scripts/load_from_hub.py --name bcywinski/SAeUron --hookpoint unet.up_blocks.1.attentions.2 --save_path sae-ckpts
```

If you do not want to train SAEs from scratch, you can skip steps 2 and 3.

## 2. 💾 Collect activations from diffusion model
Create a dataset of activations for SAEs trainings using the _anchor prompts_:

```bash
python scripts/collect_activations_unlearn_canvas.py --hook_names <hook_name> --model_name <path/to/diffusion/model>
```
****
Script supports also distributed setup:
```bash
accelerate launch --num-processes=<N> scripts/collect_activations_unlearn_canvas.py --hook_names <hook_name> --model_name <path/to/diffusion/model>
```

All arguments to the script are listed in `CacheActivationsRunnerConfig`.

To collect dataset for SAEs trained in our paper, run:
```bash
python scripts/collect_activations_unlearn_canvas.py --hook_names unet.up_blocks.1.attentions.1 unet.up_blocks.1.attentions.2 --model_name <path/to/diffusion/model>
```

## 3. 🚆 Train sparse autoencoders
To train SAEs on previously gathered dataset of activations, run:
```bash
python scripts/train.py --dataset_path <path/to/dataset> --hookpoints <hook_name>
```

To reproduce training from our paper for `up.1.1` block run:
```bash
python train.py --dataset_path <path/to/dataset> --hookpoints unet.up_blocks.1.attentions.1 --effective_batch_size 4096 --auxk_alpha 0.03125 --expansion_factor 16 --k 32 --multi_topk False --num_workers 4 --wandb_log_frequency 4000 --num_epochs 5 --dead_feature_threshold 10_000_000 --lr 4e-4 --lr_scheduler linear --lr_warmup_steps 0 --batch_topk True
```

To reproduce training from our paper for `up.1.2` block run:
```bash
python train.py --dataset_path <path/to/dataset> --hookpoints unet.up_blocks.1.attentions.2 --effective_batch_size 4096 --auxk_alpha 0.03125 --expansion_factor 16 --k 32 --multi_topk False --num_workers 4 --wandb_log_frequency 4000 --num_epochs 10 --dead_feature_threshold 10_000_000 --lr 4e-4 --lr_scheduler linear --lr_warmup_steps 0 --batch_topk True
```

Training checkpoints will be stored under `sae-ckpts` directory.

## 4. 💾 Collect SAE feature activations on validation set
To collect SAE feature activations for style unlearning run:
```bash
python scripts/gather_sae_acts_ca_prompts.py --checkpoint_path "<path/to/sae/checkpoint>" --hookpoint "unet.up_blocks.1.attentions.2" --pipe_path "<path/to/diffusion/model>" --save_dir "<path/to/style/activations>"
```

and to collect SAE feature activations for object unlearning run:
```bash
python scripts/gather_sae_acts_ca_prompts_cls.py --checkpoint_path "<path/to/sae/checkpoint>" --hookpoint "unet.up_blocks.1.attentions.1" --pipe_path "<path/to/diffusion/model>" --save_dir "<path/to/object/activations>"
```

Feature activations will be saved as `pickle` files.

## ❌ 5. Unlearning
All sampling scripts support distributed setup using `accelerate`. If no distributed environment is available, set `--num-processes=1`.


### 5.1 Hyperparameter sweep for object unlearning
First run sampling for all possible pairs of hyperparameters:

```bash
accelerate launch --num_processes <N> scripts/sweep_cls_distr.py --percentiles [99.99,99.995,99.999] --multipliers [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30.0]> --seed 42 --output_dir '<path/to/output/dir>' --pipe_checkpoint '<path/to/diffusion/model>' --hookpoint 'unet.up_blocks.1.attentions.1' --class_latents_path '<path/to/object/activations>' --sae_checkpoint '<path/to/sae/checkpoint>' --steps 100
```

Then run classifier prediction on all images:
```bash
python scripts/run_acc_all_cls_sweep.py --percentiles [99.99,99.995,99.999] --multipliers [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30.0]> --input_dir_base <path/to/saved/images> --output_dir_base <path/to/save/results> --class_ckpt <path/to/object/classifier> --batch_size <batch_size> --seed <seed>
```

And find best parameters for each object:
```bash
python scripts/find_best_params_cls_sweep.py --percentiles [99.99,99.995,99.999] --multipliers [-1.0,-5.0,-10.0,-15.0,-20.0,-30.0] --base_path <path/to/saved/images>
```

Best parameters will be saved under `base_path` as `class_params.pth`.

### 5.2 Style unlearning
Run sampling with SAE-based unlearning during the inference:
```bash
accelerate launch --num_processes <N> scripts/sample_unlearning_distr.py --percentile 99.999 --multiplier -1.0 --seed <seed> --output_dir '<path/to/output/dir>' --pipe_checkpoint '<path/to/diffusion/model>' --hookpoint 'unet.up_blocks.1.attentions.2' --style_latents_path '<path/to/style/activations>' --sae_checkpoint '<path/to/sae/checkpoint>' --steps 100
```

All images will be saved as `jpg` files under `<path/to/output/dir>`.

### 5.3 Object unlearning
Run sampling with SAE-based unlearning during the inference:
```bash
accelerate launch --num_processes <N> scripts/sample_unlearning_cls_distr.py --class_params_path <path/to/class/params> --seed <seed> --output_dir '<path/to/output/dir>' --pipe_checkpoint '<path/to/diffusion/model>' --hookpoint 'unet.up_blocks.1.attentions.1' --class_latents_path '<path/to/object/activations>' --sae_checkpoint '<path/to/sae/checkpoint>' --steps 100
```

All images will be saved as `jpg` files under `<path/to/output/dir>`.

## 6. Evaluation
Evaluate unlearning performance on UnlearnCanvas benchmark.

Run evaluation for style unlearning:
```bash
python scripts/run_acc_all_style.py --input_dir <path/to/saved/images> --output_dir <path/to/save/results> --style_ckpt <path/to/style/classifier> --class_ckpt <path/to/object/classifier> --batch_size <batch_size>
```

Run evaluation for object unlearning:
```bash
python scripts/run_acc_all_cls.py --input_dir <path/to/saved/images> --output_dir <path/to/save/results> --class_ckpt <path/to/object/classifier> --batch_size <batch_size>
```

Run FID evaluation:
```bash
python scripts/run_fid_all.py --p1 '<path/to/dataset>' --p2_base '<path/to/saved/images>' --output_path_base '<path/to/save/fid/scores>'
```

## 7. Additional experiments
### 7.1 Sequential unlearning
Run sampling with sequential unlearning:
```bash
accelerate launch --num_processes <N> scripts/sample_unlearning_sequential_distr.py --percentile 99.999 --multiplier -1.0 --seed <seed> --output_dir '<path/to/output/dir>' --pipe_checkpoint '<path/to/diffusion/model>' --hookpoint 'unet.up_blocks.1.attentions.2' --style_latents_path '<path/to/style/activations>' --sae_checkpoint '<path/to/sae/checkpoint>' --steps 100
```

Evaluate sequential unlearning:
```bash
python scripts/run_acc_all_style_sequential.py --input_dir <path/to/saved/images> --output_dir <path/to/save/results> --style_ckpt <path/to/style/classifier> --class_ckpt <path/to/object/classifier> --batch_size <batch_size>
```

### 7.2 UnlearnDiffAtk
Evaluate our unlearning method against adversarial attacks.


Run style unlearning with adversarial attacks:
```bash
cd Diffusion-MU-Attack
python run_atk_all_styles.py --attack_idx <idx> --eval_seed <seed> --sampling_step_num 100
```

Run object unlearning with adversarial attacks:
```bash
cd Diffusion-MU-Attack
python run_atk_all_cls.py --attack_idx <idx> --eval_seed <seed> --class_params_path <path/to/class/params> --sampling_step_num 100
```

Evaluate unlearning performance for style unlearning:
```bash
python scripts/avg_accuracy_style_diffatk.py --input_dir <path/to/saved/images> --attk_idxs [<idxs>]
```

Evaluate unlearning performance for object unlearning:
```bash
python scripts/avg_accuracy_cls_diffatk.py --input_dir <path/to/saved/images> --attk_idxs [<idxs>]
```

## 📚 Bibtex
```bibtex
@article{cywinski2025saeuron,
  title={SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders},
  author={Cywi{\'n}ski, Bartosz and Deja, Kamil},
  journal={arXiv preprint arXiv:2501.18052},
  year={2025}
}
```

## 📚 Acknowledgements
This repository uses code from [ElutherAI repo](https://github.com/EleutherAI/sae) and [Unpacking SDXL-Turbo repo](https://github.com/surkovv/sdxl-unbox).
