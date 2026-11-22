import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Thêm thư mục flower102_classifier vào sys.path để import build_model
ROOT_DIR = Path(__file__).resolve().parent.parent
CLS_DIR = ROOT_DIR / "flower102_classifier"
sys.path.insert(0, str(CLS_DIR))
from model import build_model


def imread_unicode(path: str) -> np.ndarray:
    """Đọc ảnh an toàn trên Windows kể cả đường dẫn có dấu tiếng Việt."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def crop_and_pad_square(img_bgr: np.ndarray, box: tuple, margin_ratio: float = 0.08) -> np.ndarray:
    """Cắt vùng hoa có nới biên nhẹ và pad vuông bằng phản xạ biên (tránh méo hình & viền đen)."""
    h_img, w_img = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    pad_x = int(bw * margin_ratio)
    pad_y = int(bh * margin_ratio)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w_img, x2 + pad_x), min(h_img, y2 + pad_y)

    crop = img_bgr[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return crop

    diff = abs(ch - cw)
    pad1 = diff // 2
    pad2 = diff - pad1
    if ch < cw:
        square = cv2.copyMakeBorder(crop, pad1, pad2, 0, 0, cv2.BORDER_REFLECT_101)
    elif cw < ch:
        square = cv2.copyMakeBorder(crop, 0, 0, pad1, pad2, cv2.BORDER_REFLECT_101)
    else:
        square = crop
    return square


class TwoStageFlowerPipeline:
    def __init__(self, yolo_ckpt: str, cls_ckpt: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Đang nạp pipeline trên thiết bị: {self.device}")

        # Stage 1: YOLO Detector
        self.detector = YOLO(yolo_ckpt)

        # Stage 2: ConvNeXt-Tiny 102-Class Classifier
        ckpt = torch.load(cls_ckpt, map_location=self.device, weights_only=False)
        self.idx_to_name = ckpt["idx_to_name"]
        self.classifier = build_model(
            model_name=ckpt["model_name"],
            num_classes=ckpt["num_classes"],
            dropout=ckpt["dropout"],
            pretrained=False,
        )
        self.classifier.load_state_dict(ckpt["model_state_dict"])
        self.classifier.to(self.device).eval()

        self.cls_transform = transforms.Compose([
            transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=ckpt["mean"], std=ckpt["std"]),
        ])

    @torch.no_grad()
    def predict(
        self,
        img_bgr: np.ndarray,
        det_conf: float = 0.35,
        det_iou: float = 0.50,
        cls_conf: float = 0.45,
        min_box_size: int = 24,
    ) -> List[Dict[str, Any]]:
        yolo_res = self.detector(img_bgr, conf=det_conf, nms=False, verbose=False)[0]

        valid_boxes = []
        tensors = []

        for box in yolo_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_score = float(box.conf[0].item())

            if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                continue

            crop_sq = crop_and_pad_square(img_bgr, (x1, y1, x2, y2))
            if crop_sq.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop_sq, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensors.append(self.cls_transform(pil_img))
            valid_boxes.append(((x1, y1, x2, y2), det_score))

        if not tensors:
            return []

        batch = torch.stack(tensors).to(self.device)
        logits = self.classifier(batch)
        probs = F.softmax(logits, dim=1)
        top_probs, top_idxs = probs.topk(3, dim=1)

        outputs = []
        for ((x1, y1, x2, y2), det_score), p_row, i_row in zip(valid_boxes, top_probs, top_idxs):
            best_prob = float(p_row[0].item())
            best_name = self.idx_to_name[int(i_row[0].item())]
            label = best_name if best_prob >= cls_conf else f"Unknown ({best_name}?)"

            top3 = [
                (self.idx_to_name[int(idx.item())], float(pr.item()))
                for pr, idx in zip(p_row, i_row)
            ]
            outputs.append({
                "box": (x1, y1, x2, y2),
                "det_conf": det_score,
                "label": label,
                "cls_conf": best_prob,
                "top3": top3,
            })
        return outputs


def draw_predictions(img_bgr: np.ndarray, predictions: List[Dict[str, Any]]) -> np.ndarray:
    annotated = img_bgr.copy()
    for item in predictions:
        x1, y1, x2, y2 = item["box"]
        is_unknown = item["label"].startswith("Unknown")
        color = (0, 165, 255) if is_unknown else (36, 200, 70)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{item['label']} ({item['cls_conf'] * 100:.1f}%)"

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = y1 - 8 if y1 - th - 12 > 0 else y1 + th + 10
        cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 6, ty + baseline), color, -1)
        cv2.putText(annotated, text, (x1 + 3, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def show_resized(win_name: str, img_bgr: np.ndarray, max_w: int = 1280, max_h: int = 800):
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr
    cv2.imshow(win_name, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detector_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", type=str,
        default=r"D:\Python\ComputerVison\Flowers\flowers-7135053_1280.jpg",
        help="Đường dẫn ảnh đầu vào"
    )
    parser.add_argument("--yolo-ckpt", type=str, default=str(detector_dir / "checkpoints" / "best_yolo.pt"))
    parser.add_argument("--cls-ckpt", type=str, default=str(CLS_DIR / "checkpoints" / "best_model.pth"))
    parser.add_argument("--det-conf", type=float, default=0.35)
    parser.add_argument("--det-iou", type=float, default=0.50)
    parser.add_argument("--cls-conf", type=float, default=0.45)
    parser.add_argument("--save-out", type=str, default=str(detector_dir / "result_output.jpg"))
    args = parser.parse_args()

    img = imread_unicode(args.image)
    assert img is not None, f"Không thể đọc ảnh tại: {args.image}"

    pipeline = TwoStageFlowerPipeline(args.yolo_ckpt, args.cls_ckpt)
    preds = pipeline.predict(img, det_conf=args.det_conf, det_iou=args.det_iou, cls_conf=args.cls_conf)

    print(f"\nPhát hiện {len(preds)} bông hoa trong ảnh:")
    for idx, p in enumerate(preds, start=1):
        top3_str = ", ".join(f"{name}: {prob*100:.1f}%" for name, prob in p["top3"])
        print(f"  #{idx} Box={p['box']} (YOLO {p['det_conf']*100:.1f}%) -> {p['label']} | Top-3: [{top3_str}]")

    out_img = draw_predictions(img, preds)
    cv2.imwrite(args.save_out, out_img)
    print(f"Đã lưu ảnh kết quả tại: {args.save_out}")
    show_resized("Flower Detection + 102-Species Classification", out_img)