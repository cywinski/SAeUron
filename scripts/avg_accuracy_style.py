import os

import fire
import torch
from tqdm import tqdm

from UnlearnCanvas_resources.const import class_available, theme_available


def main(input_dir: str):
    avg_ua = 0.0
    avg_ira = 0.0
    avg_cra = 0.0
    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    progress_bar = tqdm(theme_avail, desc="Processing themes")
    for theme in progress_bar:
        progress_bar.set_description(f"Processing theme {theme}")
        data_style = torch.load(os.path.join(input_dir, f"{theme}.pth"))
        data_class = torch.load(os.path.join(input_dir, f"{theme}_cls.pth"))
        acc_data_style = data_style["acc"]
        avg_ua += 1 - acc_data_style[theme]
        curr_avg_ira = 0.0
        for theme_to_compare in theme_avail:
            if theme_to_compare != theme:
                curr_avg_ira += acc_data_style[theme_to_compare]
        avg_ira += curr_avg_ira / (len(theme_avail) - 1)
        acc_data_class = data_class["acc"]
        curr_avg_cra = 0.0
        for class_ in class_available:
            curr_avg_cra += acc_data_class[class_]
        avg_cra += curr_avg_cra / len(class_available)
    avg_ua /= len(theme_avail)
    avg_ira /= len(theme_avail)
    avg_cra /= len(theme_avail)
    print(f"Average UA: {avg_ua * 100:.2f}%")
    print(f"Average IRA: {avg_ira*100:.2f}%")
    print(f"Average CRA: {avg_cra * 100:.2f}%")


if __name__ == "__main__":
    fire.Fire(main)
