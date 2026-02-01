import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class CocoCropClassificationDataset(Dataset):
    """
    Converts COCO detection dataset into
    object-level classification samples.

    Each bounding box becomes one training example.
    """

    def __init__(self, image_dir, ann_file, transforms=None):
        self.coco = COCO(ann_file)
        self.image_dir = Path(image_dir)
        self.transforms = transforms

        self.samples = []

        for img_id in self.coco.imgs:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                if ann.get("iscrowd", 0):
                    continue

                self.samples.append(
                    (
                        img_id,
                        ann["bbox"],          # x,y,w,h
                        ann["category_id"],   # label
                    )
                )

        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_idx = {
            cat_id: i for i, cat_id in enumerate(self.cat_ids)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, bbox, cat_id = self.samples[idx]

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.image_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")

        x, y, w, h = bbox
        crop = image.crop((x, y, x + w, y + h))

        if self.transforms:
            crop = self.transforms(crop)

        label = self.cat_id_to_idx[cat_id]

        return crop, label


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ResNet50_Weights.IMAGENET1K_V1.meta["mean"],
                std=ResNet50_Weights.IMAGENET1K_V1.meta["std"],
            ),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ResNet50_Weights.IMAGENET1K_V1.meta["mean"],
                std=ResNet50_Weights.IMAGENET1K_V1.meta["std"],
            ),
        ]
    )

    train_ds = CocoCropClassificationDataset(
        image_dir=os.path.join(args.data, "images/train"),
        ann_file=os.path.join(args.data, "annotations/instances_train.json"),
        transforms=train_tf,
    )

    val_ds = CocoCropClassificationDataset(
        image_dir=os.path.join(args.data, "images/val"),
        ann_file=os.path.join(args.data, "annotations/instances_val.json"),
        transforms=val_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    num_classes = len(train_ds.cat_ids)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training classifier on {len(train_ds)} crops")
    print(f"Number of classes: {num_classes}")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        print(
            f"Epoch {epoch+1:02d} | "
            f"loss={train_loss:.4f} | "
            f"acc={train_acc*100:.2f}%"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "cat_id_to_idx": train_ds.cat_id_to_idx,
        },
        "resnet50_openimages_animals.pth",
    )

    print("Model saved.")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
