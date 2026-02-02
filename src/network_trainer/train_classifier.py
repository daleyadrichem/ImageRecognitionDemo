import os
import json
import argparse
from pathlib import Path
import glob
import re

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

    def __init__(self, image_dir, ann_file, classes_file, transforms=None):
        self.coco = COCO(ann_file)
        self.image_dir = Path(image_dir)
        self.transforms = transforms

        # ------------------------------------------------
        # Load allowed classes explicitly
        # ------------------------------------------------
        with open(classes_file) as f:
            self.class_names = [c.strip() for c in f if c.strip()]

        self.class_to_idx = {
            name: i for i, name in enumerate(self.class_names)
        }

        self.idx_to_class = {
            i: name for name, i in self.class_to_idx.items()
        }

        print(f"Using {len(self.class_names)} classes:")
        for c in self.class_names:
            print(" ", c)

        # ------------------------------------------------
        # Build samples
        # ------------------------------------------------
        self.samples = []

        for img_id in self.coco.imgs:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                if ann.get("iscrowd", 0):
                    continue

                cat = self.coco.loadCats(ann["category_id"])[0]["name"]

                if cat not in self.class_to_idx:
                    continue

                self.samples.append(
                    (
                        img_id,
                        ann["bbox"],              # x,y,w,h
                        self.class_to_idx[cat],   # remapped label
                    )
                )

        print(f"Total classification samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, bbox, label = self.samples[idx]

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.image_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")

        x, y, w, h = bbox
        crop = image.crop((x, y, x + w, y + h))

        if self.transforms:
            crop = self.transforms(crop)

        return crop, label



# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def find_latest_checkpoint(ckpt_dir, prefix):
    pattern = re.compile(rf"{prefix}_epoch_(\d+)\.pth")
    checkpoints = glob.glob(os.path.join(ckpt_dir, f"{prefix}_epoch_*.pth"))

    if not checkpoints:
        return None, 0

    epochs = []
    for ckpt in checkpoints:
        match = pattern.search(os.path.basename(ckpt))
        if match:
            epochs.append((int(match.group(1)), ckpt))

    if not epochs:
        return None, 0

    latest_epoch, latest_ckpt = max(epochs, key=lambda x: x[0])
    return latest_ckpt, latest_epoch + 1

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

    weights = ResNet50_Weights.IMAGENET1K_V1

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            weights.transforms(),
        ]
    )

    val_tf = weights.transforms()


    train_ds = CocoCropClassificationDataset(
        image_dir=os.path.join(args.data, "images/train"),
        ann_file=os.path.join(args.data, "annotations/instances_train.json"),
        classes_file=os.path.join(args.data, "classes.txt"),
        transforms=train_tf,
    )

    val_ds = CocoCropClassificationDataset(
        image_dir=os.path.join(args.data, "images/val"),
        ann_file=os.path.join(args.data, "annotations/instances_val.json"),
        classes_file=os.path.join(args.data, "classes.txt"),
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

    num_classes = len(train_ds.class_names)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    ckpt_dir = "models/classification"
    prefix = "resnet50_openimages_animals"
    os.makedirs(ckpt_dir, exist_ok=True)

    if not args.rerun:
        ckpt_path, start_epoch = find_latest_checkpoint(ckpt_dir, prefix)

        if ckpt_path is not None:
            print(f"Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)

            if checkpoint["classes"] != train_ds.class_names:
                raise ValueError("Class list in checkpoint does not match current dataset")

            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1

    print(f"Training classifier on {len(train_ds)} crops")
    print(f"Number of classes: {num_classes}")

    for epoch in range(start_epoch, args.epochs):
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
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "classes": train_ds.class_names,
                "class_to_idx": train_ds.class_to_idx,
            },
            f"{ckpt_dir}/{prefix}_epoch_{epoch:03d}.pth",
        )

    torch.save(
        {
            "model": model.state_dict(),
            "classes": train_ds.class_names,
            "class_to_idx": train_ds.class_to_idx
        },
        f"{ckpt_dir}/{prefix}.pth",
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
    parser.add_argument("--rerun", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
