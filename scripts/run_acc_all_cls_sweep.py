import subprocess

import fire

from UnlearnCanvas_resources.const import class_available


def run_scripts_sequentially(
    classes_to_unlearn,
    multiplier,
    percentile,
    input_dir_base,
    output_dir_base,
    class_ckpt,
    batch_size,
    seed,
):
    base_command = (
        "PYTHONPATH=. python scripts/accuracy_unlearncanvas_cls_sweep_fast.py "
        f"--input_dir '{input_dir_base}/percentile_{percentile}_multiplier_{multiplier}/' "
        f"--output_dir '{output_dir_base}/percentile_{percentile}_multiplier_{multiplier}/' "
        f"--class_ckpt '{class_ckpt}' "
        f"--cls '{{}}' --batch_size {batch_size} --seed [{seed}]"
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


def main(
    multipliers,
    percentiles,
    input_dir_base,
    output_dir_base,
    class_ckpt,
    batch_size,
    seed,
):
    for multiplier in multipliers:
        for percentile in percentiles:
            run_scripts_sequentially(
                class_available,
                multiplier,
                percentile,
                input_dir_base,
                output_dir_base,
                class_ckpt,
                batch_size,
                seed,
            )
            process = subprocess.run(
                "PYTHONPATH=. python scripts/avg_accuracy_cls_sweep.py "
                f"'{output_dir_base}/percentile_{percentile}_multiplier_{multiplier}/'",
                shell=True,
            )
            if process.returncode != 0:
                print("Error: Failed to run average accuracy calculation")


if __name__ == "__main__":
    fire.Fire(main)
