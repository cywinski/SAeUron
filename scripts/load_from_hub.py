from pathlib import Path

import fire

from SAE.sae import Sae


def load_sae_model(
    name: str,
    hookpoint: str | None = None,
    device: str = "cpu",
    decoder: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Load a SAE model from Huggingface Hub and optionally save it locally.

    Args:
        model_name: Name of the model on Huggingface Hub
        hookpoint: Specific layer/hookpoint to load (optional)
        device: Device to load the model on ('cpu' or 'cuda')
        decoder: Whether to load the decoder weights
        save_path: Local path to save the model (optional)
    """
    try:
        if hookpoint:
            print(f"Loading SAE model {name} for hookpoint {hookpoint}...")
            sae = Sae.load_from_hub(
                name, hookpoint=hookpoint, device=device, decoder=decoder
            )
        else:
            print(f"Loading all SAE models from {name}...")
            sae = Sae.load_many(name, device=device, decoder=decoder)

        print("Model loaded successfully!")

        if save_path:
            save_path = Path(save_path)
            print(f"Saving model to {save_path}...")
            if isinstance(sae, dict):
                # If multiple models were loaded
                for hook, model in sae.items():
                    hook_path = save_path / hook
                    model.save_to_disk(hook_path)
            else:
                # If single model was loaded
                sae.save_to_disk(save_path)
            print("Model saved successfully!")

    except Exception as e:
        print(f"Error loading model: {str(e)}")


if __name__ == "__main__":
    fire.Fire(load_sae_model)
