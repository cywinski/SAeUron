import json
import os

import fire
from tqdm import tqdm

from UnlearnCanvas_resources.const import class_available, theme_available


def main(input_dir: str, attk_idxs: list[int]):
    total_success_before_attack = 0
    total_success_after_attack = 0
    total_attempts = 0
    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    progress_bar = tqdm(
        total=len(attk_idxs) * len(theme_avail) * len(class_available),
    )
    for theme in theme_avail:
        for cls in class_available:
            for attk_idx in attk_idxs:
                json_path = os.path.join(
                    input_dir, theme, cls, f"attack_idx_{attk_idx}", "log.json"
                )
                if not os.path.exists(json_path):
                    continue
                try:
                    json_log = json.load(open(json_path))
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
                    continue
                total_success_after_attack += json_log[-1]["success"]
                total_success_before_attack += json_log[0]["success"]
                total_attempts += 1
                progress_bar.update(1)
                progress_bar.set_description(
                    f"UA before: {((total_attempts - total_success_before_attack) / total_attempts) * 100:.2f}%, UA after: {((total_attempts - total_success_after_attack) / total_attempts) * 100:.2f}%"
                )
    print(
        f"Average UA before: {((total_attempts - total_success_before_attack) / total_attempts) * 100:.2f}%, Average UA after: {((total_attempts - total_success_after_attack) / total_attempts) * 100:.2f}%"
    )
    print(
        f"{total_success_before_attack=}, {total_success_after_attack=}, {total_attempts=}"
    )


if __name__ == "__main__":
    fire.Fire(main)
