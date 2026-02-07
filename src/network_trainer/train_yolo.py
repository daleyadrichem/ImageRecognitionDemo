import os
import re
import glob
import argparse
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------

def find_latest_checkpoint(ckpt_dir, prefix):
    """
    Find the latest YOLO checkpoint based on epoch number.
    """
    pattern = re.compile(rf"{prefix}_epoch_(\d+)\.pt")
    checkpoints = glob.glob(os.path.join(ckpt_dir, f"{prefix}_epoch_*.pt"))

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

def train(
    model,
    data_yaml,
    device,
    epochs,
    batch_size,
    lr,
    ckpt_dir,
):
    """
    Train YOLOv8 model with manual epoch control so we can
    checkpoint similarly to Faster R-CNN training.
    """

    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        lr0=lr,
        device=device,
        imgsz=640,
        workers=4,
        save=True,
        save_period=1,   # 👈 checkpoint every epoch
        project=ckpt_dir,
        name="yolov8s_openimages_animals",
        exist_ok=True,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path("models/detection")
    prefix = "yolov8s_openimages_animals"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml at {data_yaml}")

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = YOLO("yolov8s.pt")

    start_epoch = 0

    # -----------------------------------------------------
    # Resume logic
    # -----------------------------------------------------

    if not args.rerun:
        last_ckpt = Path(ckpt_dir) / "yolov8s_openimages_animals" / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"Resuming from {last_ckpt}")
            model = YOLO(last_ckpt)


    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    train(
        model=model,
        data_yaml=str(data_yaml),
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ckpt_dir=ckpt_dir,
    )

    # -----------------------------------------------------
    # Final model
    # -----------------------------------------------------

    final_path = ckpt_dir / f"{prefix}.pt"
    model.save(final_path)

    print(f"Final model saved to: {final_path}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to YOLO dataset root (contains data.yaml)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, cpu")
    parser.add_argument("--rerun", action="store_true")

    args = parser.parse_args()
    main(args)
