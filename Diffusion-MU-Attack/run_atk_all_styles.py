import subprocess
import sys

import fire

sys.path.append("..")

import os

from UnlearnCanvas_resources.const import class_available, theme_available


def run_scripts_sequentially(
    attack_idx: int,
    eval_seed: int,
    sampling_step_num: int = 50,
    percentile: float = 99.999,
    multiplier: float = -1.0,
    reverse: bool = False,
    start_theme_idx: int = 0,
    end_theme_idx: int = 50,
):
    base_command = (
        "PYTHONPATH=. python src/execs/attack.py "
        "--config-file configs/style/text_grad_esd_style_classifier.json "
        "--attacker.attack_idx {} "
        "--attacker.eval_seed {} "
        "--logger.name attack_idx_{} "
        "--task.class_name '{}' "
        "--task.concept {} "
        "--logger.json.root diffatk_eval/{}/{} "
        "--task.sampling_step_num {} "
        "--task.percentile {} "
        "--task.multiplier {} "
    )
    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    if reverse:
        theme_avail = theme_avail[::-1]

    theme_avail = theme_avail[start_theme_idx:end_theme_idx]

    for theme in theme_avail:
        for cls in class_available:
            json_root = f"diffatk_eval/{theme}/{cls}/attack_idx_{attack_idx}"
            if os.path.exists(json_root) and os.listdir(json_root):
                print(f"Skipping {theme} {cls} {attack_idx} because it already exists")
                continue
            command = base_command.format(
                attack_idx,
                eval_seed,
                attack_idx,
                cls,
                theme,
                theme,
                cls,
                sampling_step_num,
                percentile,
                multiplier,
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
