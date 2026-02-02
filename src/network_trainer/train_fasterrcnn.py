import os
import argparse
import glob
import re

import torch
import torchvision
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------

class CocoDetectionWrapper(CocoDetection):
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)

        image_id = self.ids[idx]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
        }

        return F.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model


# ---------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------

def find_latest_checkpoint(ckpt_dir, prefix):
    pattern = re.compile(rf"{prefix}_epoch_(\d+)\.pth")
    checkpoints = glob.glob(os.path.join(ckpt_dir, f"{prefix}_epoch_*.pth"))

    if not checkpoints:
        return None, 0

    parsed = []
    for ckpt in checkpoints:
        m = pattern.search(os.path.basename(ckpt))
        if m:
            parsed.append((int(m.group(1)), ckpt))

    if not parsed:
        return None, 0

    epoch, path = max(parsed, key=lambda x: x[0])
    return path, epoch + 1


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, max_batches=None):
    """
    Train the model for one epoch.

    Args:
        model: Detection model
        loader: DataLoader
        optimizer: Optimizer
        device: Torch device
        max_batches: Optional limit on number of batches (debug mode)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    # Determine how many batches tqdm should expect
    total = min(len(loader), max_batches) if max_batches else len(loader)

    progress_bar = tqdm(
        enumerate(loader),
        total=total,
        desc="Training",
        leave=False,
    )

    for i, (images, targets) in progress_bar:
        if max_batches is not None and i >= max_batches:
            break

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = losses.item()
        total_loss += loss_value

        # Update tqdm display
        progress_bar.set_postfix(loss=f"{loss_value:.4f}")

    return total_loss / (i + 1)



# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = "models/detection"
    prefix = "fasterrcnn_openimages_animals"
    os.makedirs(ckpt_dir, exist_ok=True)

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = CocoDetectionWrapper(
        root=os.path.join(args.data, "images/train"),
        annFile=os.path.join(args.data, "annotations/instances_train.json"),
    )

    val_dataset = CocoDetectionWrapper(
        root=os.path.join(args.data, "images/val"),
        annFile=os.path.join(args.data, "annotations/instances_val.json"),
    )

    num_classes = len(train_dataset.coco.getCatIds()) + 1  # + background

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = build_model(num_classes)
    model.to(device)

    # -----------------------------------------------------
    # Optimizer setup (initially backbone frozen)
    # -----------------------------------------------------

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    start_epoch = 0

    # -----------------------------------------------------
    # Resume logic
    # -----------------------------------------------------

    if not args.rerun:
        ckpt_path, start_epoch = find_latest_checkpoint(ckpt_dir, prefix)

        if ckpt_path is not None:
            print(f"Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)

            required_keys = {"epoch", "model", "optimizer", "num_classes"}
            missing = required_keys - checkpoint.keys()
            if missing:
                raise KeyError(f"Checkpoint missing keys: {missing}")

            if checkpoint["num_classes"] != num_classes:
                raise ValueError(
                    f"Class mismatch: checkpoint={checkpoint['num_classes']} "
                    f"dataset={num_classes}"
                )

            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1

    # -----------------------------------------------------
    # Warmup phase
    # -----------------------------------------------------

    print("Training detection heads only...")

    for epoch in range(start_epoch, min(args.warmup_epochs, args.epochs)):
        loss = train_one_epoch(model, train_loader, optimizer, device, max_batches=args.max_batches)
        print(f"[Warmup {epoch+1}] loss={loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "num_classes": num_classes,
            },
            f"{ckpt_dir}/{prefix}_epoch_{epoch:03d}.pth",
        )

    # -----------------------------------------------------
    # Fine-tuning phase
    # -----------------------------------------------------

    for p in model.backbone.parameters():
        p.requires_grad = True

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr * 0.1,
        momentum=0.9,
        weight_decay=1e-4,
    )

    print("Fine-tuning entire network...")

    for epoch in range(max(start_epoch, args.warmup_epochs), args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, max_batches=args.max_batches)
        print(f"[Epoch {epoch+1}] loss={loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "num_classes": num_classes,
            },
            f"{ckpt_dir}/{prefix}_epoch_{epoch:03d}.pth",
        )

    # -----------------------------------------------------
    # Final model (inference-only)
    # -----------------------------------------------------

    torch.save(
        {
            "model": model.state_dict(),
            "num_classes": num_classes,
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
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches per epoch (debug mode)")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--rerun", action="store_true")

    args = parser.parse_args()
    main(args)
