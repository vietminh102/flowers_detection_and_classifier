import random
import shutil
from pathlib import Path
from typing import Optional, Set, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_and_save_valid_jpg(src_img: Path, dst_img: Path) -> bool:
    """Đọc ảnh bằng OpenCV (fallback sang Pillow nếu cần) rồi lưu chuẩn .jpg."""
    data = np.fromfile(str(src_img), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        try:
            with Image.open(src_img) as im:
                im = im.convert("RGB")
                img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        except Exception:
            return False

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return False
    buf.tofile(str(dst_img))
    return True


def normalize_label_file(
    src_lbl: Path,
    dst_lbl: Path,
    keep_class_ids: Optional[Set[int]] = None,
) -> Tuple[int, int]:
    """
    Đưa toàn bộ nhãn hợp lệ về class_id = 0 (flower).
    Tự động chuyển đổi nếu file nhãn chứa tọa độ polygon (>5 phần tử/dòng).
    Trả về: (số box giữ lại, số dòng bị bỏ qua).
    """
    if not src_lbl.exists():
        dst_lbl.write_text("", encoding="utf-8")
        return 0, 0

    clean_lines = []
    skipped = 0

    for raw_line in src_lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue

        try:
            cls_id = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            skipped += 1
            continue

        if keep_class_ids is not None and cls_id not in keep_class_ids:
            skipped += 1
            continue

        if len(coords) == 4:
            xc, yc, w, h = coords
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # Chuyển đổi nhãn polygon sang bounding box
            xs = coords[0::2]
            ys = coords[1::2]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
        else:
            skipped += 1
            continue

        # Giới hạn an toàn trong [0, 1] và bỏ box bị suy biến
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)

        if w <= 0.002 or h <= 0.002:
            skipped += 1
            continue

        clean_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    dst_lbl.write_text("".join(clean_lines), encoding="utf-8")
    return len(clean_lines), skipped


def copy_yolo_source(
    src_root: Path,
    out_root: Path,
    prefix: str,
    keep_class_ids: Optional[Set[int]] = None,
):
    """Chép toàn bộ các tập train / valid / test từ một bộ YOLO vào out_root."""
    yaml_file = src_root / "data.yaml"
    if yaml_file.exists():
        cfg = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        print(f"[{prefix}] Các lớp gốc trong {yaml_file.name}: {cfg.get('names')}")

    stats = {"images": 0, "boxes": 0, "corrupt": 0}

    for split in ["train", "valid", "test"]:
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        if not img_dir.exists():
            continue

        out_img_dir = out_root / split / "images"
        out_lbl_dir = out_root / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        for i, img_p in enumerate(img_files, start=1):
            stem = f"{prefix}_{split}_{i:05d}"
            dst_img = out_img_dir / f"{stem}.jpg"
            dst_lbl = out_lbl_dir / f"{stem}.txt"

            if not load_and_save_valid_jpg(img_p, dst_img):
                stats["corrupt"] += 1
                continue

            src_lbl = lbl_dir / f"{img_p.stem}.txt"
            n_boxes, _ = normalize_label_file(src_lbl, dst_lbl, keep_class_ids=keep_class_ids)
            stats["images"] += 1
            stats["boxes"] += n_boxes

    print(
        f"[{prefix}] Đã gộp: {stats['images']} ảnh | "
        f"{stats['boxes']} boxes | Loại bỏ ảnh hỏng: {stats['corrupt']}"
    )


def copy_negatives(
    neg_dir: Path,
    out_root: Path,
    val_ratio: float = 0.15,
    max_negatives: Optional[int] = None,
    seed: int = 42,
):
    """Chia ảnh nền không có hoa vào train/valid kèm file nhãn .txt rỗng."""
    if not neg_dir.exists():
        print(f"[neg] Không tìm thấy thư mục: {neg_dir}")
        return

    imgs = sorted([p for p in neg_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    rng = random.Random(seed)
    rng.shuffle(imgs)

    if max_negatives is not None and len(imgs) > max_negatives:
        imgs = imgs[:max_negatives]

    n_val = int(len(imgs) * val_ratio)
    ok_train, ok_val, corrupt = 0, 0, 0

    for i, img_p in enumerate(imgs, start=1):
        split = "valid" if i <= n_val else "train"
        out_img_dir = out_root / split / "images"
        out_lbl_dir = out_root / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        stem = f"neg_{split}_{i:05d}"
        if load_and_save_valid_jpg(img_p, out_img_dir / f"{stem}.jpg"):
            (out_lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")
            if split == "valid":
                ok_val += 1
            else:
                ok_train += 1
        else:
            corrupt += 1

    print(
        f"[neg] Đã gộp ảnh nền -> Train: {ok_train} | Valid: {ok_val} | Ảnh hỏng: {corrupt}"
    )


def main():
    base_dir = Path(__file__).resolve().parent
    old_data = base_dir / "data"
    bouquets_data = base_dir / "extra_data" / "bouquets_raw"
    negatives_dir = base_dir / "extra_data" / "negatives_raw"
    out_dir = base_dir / "data_merged"

    # Xóa thư mục đích cũ nếu đã chạy trước đó để đảm bảo sạch tuyệt đối
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print("=== BẮT ĐẦU TRỘN DỮ LIỆU ===")
    # 1. Bộ dữ liệu gốc
    copy_yolo_source(old_data, out_dir, prefix="base")

    # 2. Bộ bó hoa bổ sung (nếu data.yaml có lớp không phải hoa, điền ID lớp hoa vào keep_class_ids={0, 1, ...})
    copy_yolo_source(bouquets_data, out_dir, prefix="bq", keep_class_ids=None)

    # 3. Ảnh nền không có hoa (tối đa ~400 ảnh để giữ tỷ lệ nền hợp lý 10-15%)
    copy_negatives(negatives_dir, out_dir, val_ratio=0.15, max_negatives=450)

    # 4. Tạo data.yaml chuẩn 1 lớp
    merged_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": ["flower"],
    }
    if (out_dir / "test" / "images").exists():
        merged_yaml["test"] = "test/images"

    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(merged_yaml, f, sort_keys=False)

    # Thống kê cuối cùng
    print("\n=== TỔNG KẾT THƯ MỤC data_merged ===")
    for split in ["train", "valid", "test"]:
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        if not img_dir.exists():
            continue
        n_imgs = len(list(img_dir.glob("*.jpg")))
        n_empty = sum(1 for p in lbl_dir.glob("*.txt") if p.stat().st_size == 0)
        print(f"  - {split:5s}: {n_imgs:5d} ảnh (trong đó {n_empty} ảnh nền không có hoa)")

    print(f"\nHoàn tất! Thư mục sẵn sàng nén: {out_dir}")


if __name__ == "__main__":
    main()