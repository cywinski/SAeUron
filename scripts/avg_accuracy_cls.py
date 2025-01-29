import os

import fire
import torch
from tqdm import tqdm

from UnlearnCanvas_resources.const import class_available, theme_available


def main(input_dir: str):
    avg_ua = 0.0
    avg_ira = 0.0
    avg_cra = 0.0
    progress_bar = tqdm(class_available, desc="Processing classes")
    for cls in progress_bar:
        progress_bar.set_description(f"Processing class {cls}")
        data_style = torch.load(os.path.join(input_dir, f"{cls}.pth"))
        data_class = torch.load(os.path.join(input_dir, f"{cls}_cls.pth"))
        acc_data_class = data_class["acc"]
        avg_ua += 1 - acc_data_class[cls]
        curr_avg_ira = 0.0
        for cls_to_compare in class_available:
            if cls_to_compare != cls:
                curr_avg_ira += acc_data_class[cls_to_compare]
        avg_ira += curr_avg_ira / (len(class_available) - 1)
        acc_data_style = data_style["acc"]
        curr_avg_cra = 0.0
        for theme in theme_available:
            if theme != "Seed_Images":
                curr_avg_cra += acc_data_style[theme]
        avg_cra += curr_avg_cra / (len(theme_available) - 1)
    avg_ua /= len(class_available)
    avg_ira /= len(class_available)
    avg_cra /= len(class_available)
    print(f"Average UA: {avg_ua * 100:.2f}%")
    print(f"Average IRA: {avg_ira * 100:.2f}%")
    print(f"Average CRA: {avg_cra * 100:.2f}%")


if __name__ == "__main__":
    fire.Fire(main)
