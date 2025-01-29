import os
from glob import glob

import timm
import torch
from torchvision import transforms

torch.hub.set_dir("cache")
import sys

import fire
from PIL import Image
from tqdm import tqdm

sys.path.append("")
from UnlearnCanvas_resources.const import class_available


def main(
    input_dir,
    output_dir,
    class_ckpt,
    cls=None,
    seed=[188, 288, 588, 688, 888],
    dry_run=False,
    limit_classes=-1,
    batch_size=32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dir = os.path.join(input_dir, cls) if cls is not None else input_dir

    # Create folder if not exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize both models
    class_model = timm.create_model(
        "vit_large_patch16_224.augreg_in21k", pretrained=True
    ).to(device)

    class_model.head = torch.nn.Linear(1024, len(class_available)).to(device)

    class_model.load_state_dict(
        torch.load(class_ckpt, map_location=device)["model_state_dict"]
    )

    class_model.eval()

    # Initialize results dictionaries for both tasks
    class_results = {
        "test_theme": cls if cls is not None else "sd",
        "input_dir": input_dir,
        "loss": {class_: 0.0 for class_ in class_available},
        "acc": {class_: 0.0 for class_ in class_available},
        "pred_loss": {class_: 0.0 for class_ in class_available},
        "misclassified": {
            class_: {other_class: 0 for other_class in class_available}
            for class_ in class_available
        },
    }

    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, labels):
            self.image_paths = image_paths
            self.labels = labels

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path)
            image = image_transform(image)
            return image, self.labels[idx]

    classes_to_use = (
        class_available[:limit_classes] if limit_classes > 0 else class_available
    )
    class_image_paths = []
    class_labels = []
    class_label_map = {class_: idx for idx, class_ in enumerate(classes_to_use)}

    for s in seed:
        for object_class in classes_to_use:
            for f in glob(os.path.join(input_dir, f"{object_class}_seed{s}_*.jpg")):
                class_image_paths.append(f)
                class_labels.append(class_label_map[object_class])

    class_dataset = ImageDataset(class_image_paths, class_labels)
    class_dataloader = torch.utils.data.DataLoader(
        class_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    for batch_images, batch_class_labels in tqdm(class_dataloader):
        batch_images = batch_images.to(device)
        batch_class_labels = batch_class_labels.to(device)

        with torch.no_grad():
            class_res = class_model(batch_images)
            class_loss = torch.nn.functional.cross_entropy(
                class_res, batch_class_labels, reduction="none"
            )
            class_softmax = torch.nn.functional.softmax(class_res, dim=1)
            class_pred_labels = torch.argmax(class_res, dim=1)
            class_pred_success = class_pred_labels == batch_class_labels

            for i in range(len(batch_class_labels)):
                object_class = class_available[batch_class_labels[i].item()]
                class_results["loss"][object_class] += class_loss[i].item()
                class_results["pred_loss"][object_class] += class_softmax[i][
                    batch_class_labels[i]
                ].item()
                class_results["acc"][object_class] += (
                    class_pred_success[i].item()
                    * 1.0
                    / (len(classes_to_use) * len(seed))
                )
                misclassified_as = class_available[class_pred_labels[i].item()]
                class_results["misclassified"][object_class][misclassified_as] += 1

        if not dry_run:
            class_output_path = os.path.join(output_dir, f"{cls}_cls.pth")
            torch.save(class_results, class_output_path)


if __name__ == "__main__":
    fire.Fire(main)
