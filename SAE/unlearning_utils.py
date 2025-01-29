import torch


def compute_feature_importance(
    style_latents_dict, target_style, timestep, epsilon=1e-8
):
    if target_style not in style_latents_dict:
        raise ValueError(f"target_style '{target_style}' not found.")

    # Mean activation for the target style (shape: [num_features])
    latents_x = style_latents_dict[target_style][:, timestep, :].float()
    mean_x = latents_x.mean(dim=0)

    # All other styles
    other_styles = [s for s in style_latents_dict if s != target_style]
    if not other_styles:
        # If there's only one style, can't compare.
        return mean_x  # or torch.zeros_like(mean_x), depending on your needs

    # Mean activation for the combined "others"
    latents_others = torch.cat(
        [style_latents_dict[s][:, timestep, :].float() for s in other_styles], dim=0
    )
    mean_others = latents_others.mean(dim=0)

    # Denominators: total activation across all features
    total_x = mean_x.sum() + epsilon
    total_others = mean_others.sum() + epsilon

    # Proportions
    p_x = mean_x / total_x
    p_others = mean_others / total_others

    # Difference-based score
    scores = p_x - p_others

    return scores


def get_percentile_threshold(scores, percentile=95):
    """
    Returns the threshold for the given percentile.

    Args:
        scores (torch.Tensor): 1D tensor of unnormalized scores, shape [num_features].
        percentile (float):    Percentile in [0,100].

    Returns:
        threshold (float): The score value at the given percentile.
    """
    # Convert percentile from 0–100 to a fraction (0–1)
    fraction = percentile / 100.0

    # Use PyTorch's built-in quantile function (available in PyTorch 1.7+)
    threshold = torch.quantile(scores, fraction)

    return threshold
