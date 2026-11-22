import argparse
from typing import List, Tuple
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_transforms
from model import build_model


class Flower102Predictor:
    """Class đóng gói dùng độc lập hoặc nhúng trực tiếp vào pipeline YOLOv8."""
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.idx_to_name = ckpt["idx_to_name"]
        _, self.transform = get_transforms(ckpt["img_size"], ckpt["mean"], ckpt["std"])

        self.model = build_model(
            model_name=ckpt["model_name"],
            num_classes=ckpt["num_classes"],
            dropout=ckpt["dropout"],
            pretrained=False
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_pil(self, image: Image.Image, topk: int = 5) -> List[Tuple[str, float]]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]

        top_probs, top_idxs = probs.topk(topk)
        return [
            (self.idx_to_name[int(idx)], float(prob))
            for prob, idx in zip(top_probs, top_idxs)
        ]

    def predict_path(self, image_path: str, topk: int = 5) -> List[Tuple[str, float]]:
        with Image.open(image_path) as img:
            return self.predict_pil(img, topk=topk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help=r"Duong dan đến ảnh hoa")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    predictor = Flower102Predictor(args.ckpt)
    results = predictor.predict_path(args.image, topk=args.topk)

    print(f"Kết quả dự đoán cho: {args.image}")
    for rank, (name, prob) in enumerate(results, start=1):
        print(f"  {rank}. {name:30s} ({prob * 100:.2f}%)")