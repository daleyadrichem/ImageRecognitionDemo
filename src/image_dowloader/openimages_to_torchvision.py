#!/usr/bin/env python3
"""
Download selected Open Images V7 classes via FiftyOne
and export them into a torchvision-compatible COCO layout.

Final structure:

openimages_animals/
├── images/
│   ├── train/
│   └── val/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── classes.txt
"""

import argparse
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.openimages import get_classes


# -------------------------------
# Validate Open Images classes
# -------------------------------

def validate_openimages_classes(requested_classes):
    available = sorted(get_classes())
    available_set = set(available)

    valid = []
    invalid = []

    for cls in requested_classes:
        if cls in available_set:
            valid.append(cls)
        else:
            invalid.append(cls)

    if invalid:
        print("\n⚠️ Invalid Open Images classes:\n")
        for c in invalid:
            print(f"  {c}")

        print("\nAvailable classes include:\n")
        for c in available[:40]:
            print(f"  {c}")

        answer = input(
            "\nDo you want to continue without the invalid classes? [y/N]: "
        ).strip().lower()

        if answer not in ("y", "yes"):
            print("Aborted.")
            exit(1)

    if not valid:
        raise ValueError("No valid Open Images classes selected")

    print("\n✓ Using classes:")
    for c in valid:
        print(f"  {c}")
    print()

    return valid


# ---------------------------------
# Convert FiftyOne export layout
# ---------------------------------

def convert_fiftyone_export_to_torchvision(output_dir: Path):
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    ann_dir = output_dir / "annotations"

    images_train.mkdir(parents=True, exist_ok=True)
    images_val.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("📦 Moving images and annotations...")

    shutil.move(str(train_dir / "data"), str(images_train))
    shutil.move(str(val_dir / "data"), str(images_val))

    shutil.move(
        str(train_dir / "labels.json"),
        str(ann_dir / "instances_train.json"),
    )

    shutil.move(
        str(val_dir / "labels.json"),
        str(ann_dir / "instances_val.json"),
    )

    shutil.rmtree(train_dir)
    shutil.rmtree(val_dir)

    print("✅ Torchvision dataset ready\n")

def flatten_image_dirs(output_dir: Path):
    """
    Moves images from images/*/data into images/*/
    and removes the data directories.
    """

    for split in ["train", "val"]:
        split_dir = output_dir / "images" / split
        data_dir = split_dir / "data"

        if not data_dir.exists():
            continue

        print(f"📂 Flattening {split} images...")

        for img in data_dir.iterdir():
            shutil.move(str(img), str(split_dir / img.name))

        shutil.rmtree(data_dir)


# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset directory",
    )

    parser.add_argument(
        "--classes",
        nargs="*",
        default=[],
        help="Open Images class names",
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
        help="Number of training images",
    )

    parser.add_argument(
        "--val-samples",
        type=int,
        default=1000,
        help="Number of validation images",
    )

    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()

    print("\n📁 Export directory resolved to:")
    print(f"  {output_dir}\n")

    classes = validate_openimages_classes(args.classes)

    print("Downloading Open Images V7")
    print(f"Classes: {classes}")
    print(f"Train samples: {args.train_samples}")
    print(f"Val samples: {args.val_samples}\n")

    # -----------------------
    # Load datasets
    # -----------------------

    train_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections", "segmentations"],
        classes=classes,
        max_samples=args.train_samples,
        shuffle=True,
        dataset_name="openimages_train_tmp",
    )

    val_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections", "segmentations"],
        classes=classes,
        max_samples=args.val_samples,
        shuffle=True,
        dataset_name="openimages_val_tmp",
    )

    # -----------------------
    # Export
    # -----------------------

    print("Exporting train split...")

    train_dataset.export(
        export_dir=str(output_dir / "train"),
        dataset_type=fo.types.COCODetectionDataset,
        export_media=True,
    )

    print("Exporting val split...")

    val_dataset.export(
        export_dir=str(output_dir / "val"),
        dataset_type=fo.types.COCODetectionDataset,
        export_media=True,
    )

    # -----------------------
    # Convert layout
    # -----------------------

    convert_fiftyone_export_to_torchvision(output_dir)

    flatten_image_dirs(output_dir)

    # -----------------------
    # Save classes
    # -----------------------

    with open(output_dir / "classes.txt", "w") as f:
        for c in classes:
            f.write(c + "\n")

    print("Final dataset:")
    print("  images/train/")
    print("  images/val/")
    print("  annotations/instances_train.json")
    print("  annotations/instances_val.json")
    print("  classes.txt\n")

    print("Notes:")
    print("• Raw Open Images cached in: ~/fiftyone/")
    print("• Dataset usable directly by torchvision\n")


if __name__ == "__main__":
    main()
