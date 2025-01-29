import subprocess

import fire

from UnlearnCanvas_resources.const import class_available


def run_scripts_sequentially(
    classes_to_unlearn, input_dir, output_dir, style_ckpt, class_ckpt, batch_size
):
    base_command = (
        "PYTHONPATH=. python scripts/accuracy_unlearncanvas_cls_fast.py "
        f"--input_dir '{input_dir}' "
        f"--output_dir '{output_dir}' "
        f"--style_ckpt '{style_ckpt}' "
        f"--class_ckpt '{class_ckpt}' "
        "--cls '{}' "
        f"--batch_size {batch_size}"
    )

    for cls in classes_to_unlearn:
        command = base_command.format(cls)
        print(f"Running command: {command}")
        process = subprocess.run(command, shell=True)
        if process.returncode != 0:
            print(
                f"Error: Script failed with return code {process.returncode} for cls '{cls}'"
            )
            break
        else:
            print(f"Successfully completed script for cls '{cls}'")


def main(input_dir, output_dir, style_ckpt, class_ckpt, batch_size):
    run_scripts_sequentially(
        class_available, input_dir, output_dir, style_ckpt, class_ckpt, batch_size
    )
    process = subprocess.run(
        f"PYTHONPATH=. python scripts/avg_accuracy_cls.py '{output_dir}'",
        shell=True,
    )
    if process.returncode != 0:
        print("Error: Failed to run average accuracy calculation")


if __name__ == "__main__":
    fire.Fire(main)
