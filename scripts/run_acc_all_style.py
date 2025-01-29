import subprocess

import fire

from UnlearnCanvas_resources.const import theme_available


def run_scripts_sequentially(
    themes_to_unlearn, input_dir, output_dir, style_ckpt, class_ckpt, batch_size
):
    base_command = (
        "PYTHONPATH=. python scripts/accuracy_unlearncanvas_fast.py "
        f"--input_dir '{input_dir}' "
        f"--output_dir '{output_dir}' "
        f"--style_ckpt '{style_ckpt}' "
        f"--class_ckpt '{class_ckpt}' "
        "--theme '{}' "
        f"--batch_size {batch_size}"
    )

    for theme in themes_to_unlearn:
        command = base_command.format(theme)
        print(f"Running command: {command}")
        process = subprocess.run(command, shell=True)
        if process.returncode != 0:
            print(
                f"Error: Script failed with return code {process.returncode} for theme '{theme}'"
            )
            break
        else:
            print(f"Successfully completed script for theme '{theme}'")


def main(
    input_dir,
    output_dir,
    style_ckpt,
    class_ckpt,
    batch_size,
    avg_accuracy_input_dir,
):
    run_scripts_sequentially(
        [t for t in theme_available if t != "Seed_Images"],
        input_dir,
        output_dir,
        style_ckpt,
        class_ckpt,
        batch_size,
    )
    subprocess.run(
        f"PYTHONPATH=. python scripts/avg_accuracy_style.py '{avg_accuracy_input_dir}'",
        shell=True,
    )


if __name__ == "__main__":
    fire.Fire(main)
