from collections import defaultdict
from pathlib import Path

import fire
import torch


def find_best_parameters(
    percentiles: list[float], multipliers: list[float], base_path: str
):
    """
    Analyze metric files to find best parameters maximizing average of UA and IRA metrics
    for each class separately.

    Args:
        percentiles: List of percentiles to analyze
        multipliers: List of multipliers to analyze
        base_path: Base path to the results directory
    """
    # Initialize dictionary to store best results per class
    best_params = defaultdict(
        lambda: {"best_avg": float("-inf"), "params": None, "metrics": None}
    )

    for percentile in percentiles:
        for multiplier in multipliers:
            # Construct path
            result_path = (
                Path(base_path)
                / f"percentile_{percentile}_multiplier_{multiplier}"
                / "class_metrics.pth"
            )

            if not result_path.exists():
                print(f"Warning: File not found: {result_path}")
                continue

            # Load metrics
            metrics = torch.load(result_path)

            # Analyze each class
            for class_name, class_metrics in metrics.items():
                # Calculate average of UA and IRA for this class
                avg_metrics = (class_metrics["UA"] + class_metrics["IRA"]) / 2

                if avg_metrics > best_params[class_name]["best_avg"]:
                    best_params[class_name]["best_avg"] = avg_metrics
                    best_params[class_name]["params"] = (percentile, multiplier)
                    best_params[class_name]["metrics"] = class_metrics

    # Print results and save parameters
    print("\nBest parameters for each class (maximizing average of UA and IRA):")
    print("-" * 80)

    # Create dictionary for saving parameters
    params_dict = {}

    for class_name, data in best_params.items():
        print(f"\nClass: {class_name}")
        percentile, multiplier = data["params"]
        print(f"Parameters: percentile={percentile}, multiplier={multiplier}")
        print(f"UA:  {data['metrics']['UA']:.4f}")
        print(f"IRA: {data['metrics']['IRA']:.4f}")
        print(f"Average: {data['best_avg']:.4f}")

        params_dict[class_name] = {"percentile": percentile, "multiplier": multiplier}

    # Save parameters
    save_path = Path(base_path) / "class_params.pth"
    torch.save(params_dict, save_path)


if __name__ == "__main__":
    fire.Fire(find_best_parameters)
