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


def atomic_copy(src: Path, dst: Path):
    """Copy qua file tạm rồi đổi tên để tránh hỏng file .pt trên Drive nếu Colab ngắt đột ngột."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def attach_drive_sync(model: YOLO, save_best: Path, save_last: Path):
    """Tự động lưu last.pt và best_yolo.pt lên Google Drive ngay sau mỗi epoch."""
    def _on_model_save(trainer):
        last_src = Path(trainer.last)
        best_src = Path(trainer.best)

        if last_src.exists() and last_src.resolve() != save_last.resolve():
            atomic_copy(last_src, save_last)
        if best_src.exists() and best_src.resolve() != save_best.resolve():
            atomic_copy(best_src, save_best)

    model.add_callback("on_model_save", _on_model_save)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.getenv("YOLO_DATA_DIR", "data"))
    parser.add_argument("--weights", type=str, default="yolo26m.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=int(os.getenv("YOLO_NUM_WORKERS", "2")))
    parser.add_argument("--project", type=str, default=os.getenv("YOLO_PROJECT_DIR", "runs"))
    parser.add_argument("--name", type=str, default="flower_oiv7")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Truyền --resume để tự tìm last.pt, hoặc --resume /duong/dan/last.pt",
    )
    parser.add_argument("--save-best", type=str, default=os.getenv("YOLO_SAVE_BEST", "checkpoints/best_yolo.pt"))
    parser.add_argument("--save-last", type=str, default=os.getenv("YOLO_SAVE_LAST", "checkpoints/last.pt"))
    args = parser.parse_args()

    resolved_yaml = inspect_and_prepare_yaml(Path(args.data_dir))
    project_dir = Path(args.project).resolve()
    save_best = Path(args.save_best).resolve()
    save_last = Path(args.save_last).resolve()

    # Tìm checkpoint để train tiếp (ưu tiên file last.pt trong checkpoints/ trên Drive)
    resume_ckpt = None
    if args.resume == "auto":
        for candidate in [save_last, project_dir / args.name / "weights" / "last.pt"]:
            if candidate.exists():
                resume_ckpt = candidate
                break
        if resume_ckpt is None:
            print(f"Chưa có last.pt tại {save_last} -> Bắt đầu train mới từ đầu.")
    elif args.resume:
        resume_ckpt = Path(args.resume).resolve()
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint để resume: {resume_ckpt}")

    if resume_ckpt is not None:
        print(f"\n=== TIẾP TỤC HUẤN LUYỆN TỪ: {resume_ckpt} ===")
        model = YOLO(str(resume_ckpt))
        attach_drive_sync(model, save_best, save_last)
        model.train(resume=True)
    else:
        print(f"\n=== BẮT ĐẦU HUẤN LUYỆN MỚI ({args.weights}) ===")
        model = YOLO(args.weights)
        attach_drive_sync(model, save_best, save_last)
        model.train(
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
            project=str(project_dir),
            name=args.name,
            exist_ok=True,
        )

    if save_best.exists():
        print(f"\nModel tốt nhất đã được lưu tại: {save_best}")
        print(f"Model gần nhất đã được lưu tại: {save_last}")

        if (Path(args.data_dir) / "test" / "images").exists():
            print("\n=== Đánh giá trên tập Test ===")
            YOLO(str(save_best)).val(data=str(resolved_yaml), split="test")


if __name__ == "__main__":
    main()