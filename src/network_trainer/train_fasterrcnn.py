import os
import torch
import torchvision
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import argparse


# ---------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------

class CocoDetectionWrapper(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        image_id = self.ids[idx]
        anns = target

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
# Training
# ---------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(loader)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CocoDetectionWrapper(
        root=os.path.join(args.data, "images/train"),
        annFile=os.path.join(args.data, "annotations/instances_train.json"),
    )

    val_dataset = CocoDetectionWrapper(
        root=os.path.join(args.data, "images/val"),
        annFile=os.path.join(args.data, "annotations/instances_val.json"),
    )

    num_classes = len(train_dataset.coco.getCatIds()) + 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes)
    model.to(device)

    # -----------------------------------------
    # Transfer learning strategy
    # -----------------------------------------

    # Freeze backbone for first few epochs
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    print("Training detection heads only...")

    for epoch in range(args.warmup_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Warmup {epoch+1}] loss={loss:.4f}")

    # Unfreeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr * 0.1,
        momentum=0.9,
        weight_decay=1e-4,
    )

    print("Fine-tuning entire network...")

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch+1}] loss={loss:.4f}")

    torch.save(model.state_dict(), "fasterrcnn_openimages_animals.pth")
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
    parser.add_argument("--lr", type=float, default=0.005)

    args = parser.parse_args()
    main(args)
