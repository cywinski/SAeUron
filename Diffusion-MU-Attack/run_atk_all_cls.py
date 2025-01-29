import subprocess
import sys

import fire

sys.path.append("..")

import os

import torch

from UnlearnCanvas_resources.const import class_available, theme_available


def run_scripts_sequentially(
    attack_idx: int,
    eval_seed: int,
    class_params_path: str,
    sampling_step_num: int = 100,
    start_cls_idx: int = 0,
    end_cls_idx: int = 20,
):
    base_command = (
        "PYTHONPATH=. python src/execs/attack.py "
        "--config-file configs/object/text_grad_esd_object_classifier.json "
        "--attacker.attack_idx {} "
        "--attacker.eval_seed {} "
        "--logger.name attack_idx_{} "
        "--task.class_name '{}' "
        "--task.concept {} "
        "--logger.json.root diffatk_eval/{}/{} "
        "--task.sampling_step_num {} "
        "--task.percentile {} "
        "--task.multiplier {} "
        "--task.cls_atk true "
    )
    class_params = torch.load(class_params_path)
    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    class_avail = class_available[start_cls_idx:end_cls_idx]

    for cls in class_avail:
        for theme in theme_avail:
            json_root = f"diffatk_eval/{cls}/{theme}/attack_idx_{attack_idx}"
            if os.path.exists(json_root) and os.listdir(json_root):
                print(f"Skipping {theme} {cls} {attack_idx} because it already exists")
                continue
            command = base_command.format(
                attack_idx,
                eval_seed,
                attack_idx,
                cls,
                theme,
                cls,
                theme,
                sampling_step_num,
                class_params[cls]["percentile"],
                class_params[cls]["multiplier"],
            )
            print(f"Running command: {command}")
            process = subprocess.run(command, shell=True)
            if process.returncode != 0:
                print(
                    f"Error: Script failed with return code {process.returncode} for cls '{cls}'"
                )
                break
            else:
                print(f"Successfully completed script for cls '{cls}'")


if __name__ == "__main__":
    fire.Fire(run_scripts_sequentially)
