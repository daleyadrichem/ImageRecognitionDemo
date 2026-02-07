import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO format")
    parser.add_argument("--coco-root", type=Path, required=True,
                        help="Root directory of COCO dataset")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Output directory for YOLO dataset")
    parser.add_argument("--split", type=str, choices=["train", "val"], required=True,
                        help="Dataset split to convert (train or val)")
    return parser.parse_args()

def main():
    args = parse_args()

    coco_root = args.coco_root
    output_root = args.output_root
    split = args.split

    images_dir = coco_root / "images" / split
    coco_json = coco_root / "annotations" / f"instances_{split}.json"

    classes_file = coco_root / "classes.txt"
    with open(classes_file) as f:
        allowed_classes = [c.strip() for c in f.readlines()]

    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split

    # Create output folders
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    with open(coco_json, "r") as f:
        coco = json.load(f)

    # Category mapping
    categories = [
        c for c in coco["categories"]
        if c["name"] in allowed_classes
    ]

    cat_id_to_yolo = {
        cat["id"]: idx for idx, cat in enumerate(categories)
    }
    allowed_cat_ids = set(cat_id_to_yolo.keys())

    # Image lookup
    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann["category_id"] not in allowed_cat_ids:
            continue
        annotations[ann["image_id"]].append(ann)
        
    for image_id, image_info in images.items():
        file_name = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]

        src_image = images_dir / file_name
        dst_image = output_images / file_name

        if not src_image.exists():
            print(f"⚠️ Missing image: {src_image}")
            continue

        shutil.copy2(src_image, dst_image)

        label_path = output_labels / f"{Path(file_name).stem}.txt"

        with open(label_path, "w") as f:
            for ann in annotations.get(image_id, []):
                x, y, w, h = ann["bbox"]

                # COCO → YOLO
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w /= width
                h /= height

                class_id = cat_id_to_yolo[ann["category_id"]]

                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                )

    # Write data.yaml (only once)
    yaml_path = output_root / "data.yaml"
    if not yaml_path.exists():
        with open(yaml_path, "w") as f:
            f.write(f"path: {output_root.resolve()}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n\n")
            f.write("names:\n")
            for idx, cat in enumerate(categories):
                f.write(f"  {idx}: {cat['name']}\n")

    print(f"✅ Converted COCO → YOLO for split: {split}")

if __name__ == "__main__":
    main()
