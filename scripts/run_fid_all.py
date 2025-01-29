import subprocess

import fire
import numpy as np
import torch

from UnlearnCanvas_resources.const import theme_available


def run_scripts_sequentially(p1, p2_base, output_path_base, forget_themes):
    base_command = (
        "PYTHONPATH=. python scripts/fid_unlearncanvas.py "
        f"--p1 '{p1}' "
        f"--p2 '{p2_base}/{{}}/' "
        f"--output_path '{output_path_base}/{{}}/fid_score.pth' "
        "--forget_theme {} "
    )

    for theme in forget_themes:
        command = base_command.format(theme, theme, theme)
        print(f"Running command: {command}")
        process = subprocess.run(command, shell=True)
        if process.returncode != 0:
            print(
                f"Error: Script failed with return code {process.returncode} for theme '{theme}'"
            )
            break
        else:
            print(f"Successfully completed script for theme '{theme}'")


def main(p1, p2_base, output_path_base):
    themes_to_unlearn = [t for t in theme_available if t != "Seed_Images"]
    run_scripts_sequentially(p1, p2_base, output_path_base, themes_to_unlearn)
    fid_scores = []
    for theme in themes_to_unlearn:
        fid_scores.append(torch.load(f"{output_path_base}/{theme}/fid_score.pth"))
    print(np.mean(fid_scores))


if __name__ == "__main__":
    fire.Fire(main)
