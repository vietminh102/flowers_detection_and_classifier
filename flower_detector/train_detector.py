import argparse
import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO


def inspect_and_prepare_yaml(data_dir: Path) -> Path:
    """Kiểm tra nhanh dữ liệu và tạo file YAML với đường dẫn tuyệt đối chuẩn xác."""
    train_imgs = data_dir / "train" / "images"
    val_imgs = data_dir / "valid" / "images"
    test_imgs = data_dir / "test" / "images"

    assert train_imgs.exists() and val_imgs.exists(), (
        f"Không tìm thấy train/images hoặc valid/images trong {data_dir}"
    )

    n_train = len(list(train_imgs.glob("*.*")))
    n_val = len(list(val_imgs.glob("*.*")))
    n_test = len(list(test_imgs.glob("*.*"))) if test_imgs.exists() else 0
    print(f"Thống kê ảnh | Train: {n_train} | Valid: {n_val} | Test: {n_test}")

    resolved_cfg = {
        "path": str(data_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": ["flower"],
    }
    if test_imgs.exists():
        resolved_cfg["test"] = "test/images"

    resolved_yaml = data_dir / "data_resolved.yaml"
    with open(resolved_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_cfg, f, sort_keys=False)

    return resolved_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.getenv("YOLO_DATA_DIR", "data"))
    parser.add_argument("--weights", type=str, default="yolo26m.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=int(os.getenv("YOLO_NUM_WORKERS", "2")))
    parser.add_argument("--save-best", type=str, default=os.getenv("YOLO_SAVE_BEST", "checkpoints/best_yolo.pt"))
    args = parser.parse_args()

    resolved_yaml = inspect_and_prepare_yaml(Path(args.data_dir))
    model = YOLO(args.weights)

    results = model.train(
        data=str(resolved_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        amp=True,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=15.0,
        scale=0.6,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.15,
        erasing=0.2,
        close_mosaic=10,
        project="runs",
        name="flower_oiv7",
        exist_ok=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        save_target = Path(args.save_best)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, save_target)
        print(f"\nĐã lưu model YOLO tốt nhất tại: {save_target}")

        best_model = YOLO(str(best_pt))
        if (Path(args.data_dir) / "test" / "images").exists():
            print("\n=== Đánh giá trên tập Test ===")
            best_model.val(data=str(resolved_yaml), split="test")


if __name__ == "__main__":
    main()