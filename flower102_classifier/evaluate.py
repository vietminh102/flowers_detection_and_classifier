from collections import Counter
import torch
from sklearn.metrics import f1_score
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import get_transforms,has_class_folders
from model import build_model



@torch.no_grad()
def evaluate_test():
    cfg = Config()
    ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE, weights_only=False)
    idx_to_name = ckpt["idx_to_name"]

    _, eval_tf = get_transforms(ckpt["img_size"], ckpt["mean"], ckpt["std"])
    eval_dir = cfg.TEST_DIR if has_class_folders(cfg.TEST_DIR) else cfg.VALID_DIR
    test_ds = datasets.ImageFolder(eval_dir, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    model = build_model(ckpt["model_name"], ckpt["num_classes"], ckpt["dropout"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(cfg.DEVICE).eval()

    all_preds, all_labels = [], []
    top5_correct, total = 0, 0
    mistakes = Counter()

    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        logits = model(images)

        _, top5 = logits.topk(5, dim=1)
        top1 = top5[:, 0]

        top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

        for true_y, pred_y in zip(labels.cpu().tolist(), top1.cpu().tolist()):
            all_labels.append(true_y)
            all_preds.append(pred_y)
            if true_y != pred_y:
                mistakes[(idx_to_name[true_y], idx_to_name[pred_y])] += 1

    top1_acc = sum(t == p for t, p in zip(all_labels, all_preds)) / total
    top5_acc = top5_correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"\nKết quả trên tập Test ({total} ảnh):")
    print(f"  - Top-1 Accuracy : {top1_acc * 100:.2f}%")
    print(f"  - Top-5 Accuracy : {top5_acc * 100:.2f}%")
    print(f"  - Macro F1-Score : {macro_f1 * 100:.2f}%")

    print("\nTop 15 cặp loài hoa bị nhầm lẫn nhiều nhất (Thực tế -> Dự đoán):")
    for (true_name, pred_name), count in mistakes.most_common(15):
        print(f"  - {true_name:30s} -> {pred_name:30s} ({count} ảnh)")


if __name__ == "__main__":
    evaluate_test()