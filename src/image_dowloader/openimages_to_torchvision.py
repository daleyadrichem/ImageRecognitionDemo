"""
Open Images V7 → Torchvision COCO exporter

- Optional limits on sample counts
- Optional class filtering
- Unlimited download if limits are not provided

Note:
Raw data is cached in ~/fiftyone (normal behavior).
"""

import argparse
from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz


# -------------------------------------------------
# Default animal classes
# -------------------------------------------------

DEFAULT_CLASSES = [
    "Cat",
    "Dog",
    "Horse",
    "Elephant",
    "Lion",
    "Tiger",
    "Bear",
    "Zebra",
    "Giraffe",
    "Monkey",
    "Kangaroo",
    "Panda",
]


# -------------------------------------------------
# Argument parsing
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Open Images V7 and export to torchvision COCO format"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for torchvision dataset",
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Max number of training samples (default: unlimited)",
    )

    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help="Max number of validation samples (default: unlimited)",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Open Images class names (default: built-in animal list)",
    )

    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    args = parse_args()

    classes = args.classes if args.classes is not None else DEFAULT_CLASSES

    output_dir = Path(args.output).expanduser().resolve()
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)

    print(f"\nExport directory:\n  {output_dir}\n")

    print("Downloading Open Images V7")
    print(f"Classes: {classes}")
    print("Train samples:", args.train_samples or "ALL")
    print("Val samples:", args.val_samples or "ALL")
    print()

    train_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections", "segmentations"],
        classes=classes,
        max_samples=args.train_samples,   # None = unlimited
        shuffle=True,
        dataset_name="openimages_train_tmp",
    )

    val_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections", "segmentations"],
        classes=classes,
        max_samples=args.val_samples,     # None = unlimited
        shuffle=True,
        dataset_name="openimages_val_tmp",
    )

    print("Exporting train split...")
    train_dataset.export(
        export_dir=str(output_dir),
        dataset_type=fo.types.COCODetectionDataset,
        split="train",
        label_field="detections",
        classes=classes,
    )

    print("Exporting val split...")
    val_dataset.export(
        export_dir=str(output_dir),
        dataset_type=fo.types.COCODetectionDataset,
        split="val",
        label_field="detections",
        classes=classes,
    )

    # write class list
    with open(output_dir / "classes.txt", "w") as f:
        for cls in classes:
            f.write(cls + "\n")

    print("\n✅ Finished successfully")
    print("\nFinal dataset:")
    print("  images/train/")
    print("  images/val/")
    print("  annotations/instances_train.json")
    print("  annotations/instances_val.json")
    print("  classes.txt")

    print("\nNotes:")
    print("• Raw Open Images cached in: ~/fiftyone/")
    print("• Dataset usable directly by torchvision")


if __name__ == "__main__":
    main()
