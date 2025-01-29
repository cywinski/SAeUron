import os

import fire
import numpy as np
import torch
from tqdm import tqdm

from UnlearnCanvas_resources.const import class_available, theme_available


def main(input_dir: str):
    avg_ua = np.zeros((6, 6))
    avg_ra = np.zeros(6)
    sequential_themes_to_unlearn = {
        0: ["Abstractionism"],
        1: ["Abstractionism", "Byzantine"],
        2: ["Abstractionism", "Byzantine", "Cartoon"],
        3: ["Abstractionism", "Byzantine", "Cartoon", "Cold_Warm"],
        4: ["Abstractionism", "Byzantine", "Cartoon", "Cold_Warm", "Ukiyoe"],
        5: [
            "Abstractionism",
            "Byzantine",
            "Cartoon",
            "Cold_Warm",
            "Ukiyoe",
            "Van_Gogh",
        ],
    }
    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    progress_bar = tqdm(sequential_themes_to_unlearn.keys(), desc="Processing tasks")
    for curr_task_idx in progress_bar:
        curr_theme = "_".join(sequential_themes_to_unlearn[curr_task_idx])
        progress_bar.set_description(f"Processing task {curr_task_idx}: {curr_theme}")
        data_style = torch.load(os.path.join(input_dir, f"{curr_theme}.pth"))
        data_class = torch.load(os.path.join(input_dir, f"{curr_theme}_cls.pth"))
        acc_data_style = data_style["acc"]
        acc_data_class = data_class["acc"]
        for prev_task_idx in range(curr_task_idx + 1):
            prev_themes = sequential_themes_to_unlearn[prev_task_idx]
            for prev_theme in prev_themes:
                avg_ua[prev_task_idx, curr_task_idx] += 1 - acc_data_style[prev_theme]
            avg_ua[prev_task_idx, curr_task_idx] /= len(prev_themes)
            other_themes = [t for t in theme_avail if t not in prev_themes]
            curr_avg_ira = 0.0
            for other_theme in other_themes:
                curr_avg_ira += acc_data_style[other_theme]
            curr_avg_ira /= len(other_themes)
            curr_avg_cra = 0.0
            for class_ in class_available:
                curr_avg_cra += acc_data_class[class_]
            curr_avg_cra /= len(class_available)
            avg_ra[prev_task_idx] = (curr_avg_ira + curr_avg_cra) / 2

    print("UA table:")
    print(avg_ua)
    print("RA table:")
    print(avg_ra)


if __name__ == "__main__":
    fire.Fire(main)
