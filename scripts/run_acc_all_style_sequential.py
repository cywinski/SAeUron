import subprocess

import fire


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


def main(input_dir, output_dir, style_ckpt, class_ckpt, batch_size):
    sequential_themes_to_unlearn = [
        ["Abstractionism"],
        ["Abstractionism", "Byzantine"],
        ["Abstractionism", "Byzantine", "Cartoon"],
        ["Abstractionism", "Byzantine", "Cartoon", "Cold_Warm"],
        ["Abstractionism", "Byzantine", "Cartoon", "Cold_Warm", "Ukiyoe"],
        ["Abstractionism", "Byzantine", "Cartoon", "Cold_Warm", "Ukiyoe", "Van_Gogh"],
    ]
    run_scripts_sequentially(
        ["_".join(s) for s in sequential_themes_to_unlearn],
        input_dir,
        output_dir,
        style_ckpt,
        class_ckpt,
        batch_size,
    )
    subprocess.run(
        f"PYTHONPATH=. python scripts/avg_accuracy_style_sequential.py '{output_dir}'",
        shell=True,
    )


if __name__ == "__main__":
    fire.Fire(main)
