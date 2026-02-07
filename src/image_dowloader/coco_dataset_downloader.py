#!/usr/bin/env python3
"""
Balanced Open Images downloader using FiftyOne

Downloads N images per class (train + val) and merges them
into a single balanced COCO dataset.

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


# --------------------------------------------------------
# Validate Open Images classes
# --------------------------------------------------------

def validate_openimages_classes(requested_classes):
    available = set(get_classes())

    valid = []
    invalid = []

    for c in requested_classes:
        if c in available:
            valid.append(c)
        else:
            invalid.append(c)

    if invalid:
        print("\n⚠ Invalid classes:")
        for c in invalid:
            print(f"  {c}")
        raise ValueError("Invalid Open Images class names")

    return valid


# --------------------------------------------------------
# Balanced loader
# --------------------------------------------------------

def load_balanced_split(split, classes, samples_per_class, dataset_name):
    """
    Downloads images per class and merges them into
    a single balanced FiftyOne dataset.
    """

    merged_dataset = fo.Dataset(dataset_name)

    print(f"\nDownloading balanced split '{split}'")

    for cls in classes:
        print(f"  → Downloading {samples_per_class} images for {cls}")

        tmp_name = f"{dataset_name}_{cls}"

        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=[cls],
            max_samples=samples_per_class,
            shuffle=True,
            dataset_name=tmp_name,
        )

        # Add samples while avoiding duplicates
        merged_dataset.merge_samples(ds)

        fo.delete_dataset(tmp_name)

    print(f"✓ Finished balanced {split} download")
    print(f"Total samples in merged dataset: {len(merged_dataset)}")

    return merged_dataset


# --------------------------------------------------------
# Convert layout to torchvision COCO structure
# --------------------------------------------------------

def convert_fiftyone_export_to_torchvision(output_dir: Path):
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    ann_dir = output_dir / "annotations"

    images_train.mkdir(parents=True, exist_ok=True)
    images_val.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

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


def flatten_image_dirs(output_dir: Path):
    for split in ["train", "val"]:
        split_dir = output_dir / "images" / split
        data_dir = split_dir / "data"

        if not data_dir.exists():
            continue

        for img in data_dir.iterdir():
            shutil.move(str(img), str(split_dir / img.name))

        shutil.rmtree(data_dir)


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--train-per-class",
        type=int,
        default=500,
        help="Images per class for training",
    )

    parser.add_argument(
        "--val-per-class",
        type=int,
        default=100,
        help="Images per class for validation",
    )

    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    classes = validate_openimages_classes(args.classes)

    print("\nUsing classes:")
    for c in classes:
        print(" ", c)

    # ----------------------------------------------------
    # Balanced download
    # ----------------------------------------------------

    train_dataset = load_balanced_split(
        "train",
        classes,
        args.train_per_class,
        "openimages_balanced_train",
    )

    val_dataset = load_balanced_split(
        "validation",
        classes,
        args.val_per_class,
        "openimages_balanced_val",
    )

    # ----------------------------------------------------
    # Export
    # ----------------------------------------------------

    print("\nExporting train split...")
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

    # ----------------------------------------------------
    # Convert layout
    # ----------------------------------------------------

    convert_fiftyone_export_to_torchvision(output_dir)
    flatten_image_dirs(output_dir)

    # ----------------------------------------------------
    # Save class list
    # ----------------------------------------------------

    with open(output_dir / "classes.txt", "w") as f:
        for c in classes:
            f.write(c + "\n")

    print("\n✅ Balanced dataset ready!")


if __name__ == "__main__":
    main()
