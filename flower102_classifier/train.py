import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders
from model import build_model, set_backbone_trainable, get_param_groups


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    use_amp = device == "cuda"

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, top1_correct, top5_correct, total = 0.0, 0, 0, 0
    use_amp = device == "cuda"

    for images, labels in tqdm(loader, desc="Valid", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, top5_preds = logits.topk(5, dim=1)
        top1_correct += (top5_preds[:, 0] == labels).sum().item()
        top5_correct += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    return running_loss / total, top1_correct / total, top5_correct / total


def save_checkpoint(path, model, cfg, class_to_idx, idx_to_name, val_acc):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": cfg.MODEL_NAME,
        "num_classes": cfg.NUM_CLASSES,
        "dropout": cfg.DROPOUT,
        "img_size": cfg.IMG_SIZE,
        "mean": cfg.MEAN,
        "std": cfg.STD,
        "class_to_idx": class_to_idx,
        "idx_to_name": idx_to_name,
        "best_val_acc": val_acc,
    }, path)


def main():
    cfg = Config()
    seed_everything(cfg.SEED)
    print(f"Thiết bị: {cfg.DEVICE} | Kiến trúc: {cfg.MODEL_NAME}")

    train_loader, valid_loader, _, class_to_idx, idx_to_name = create_dataloaders(cfg)
    model = build_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, cfg.DROPOUT, pretrained=True).to(cfg.DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.DEVICE == "cuda"))

    best_val_acc = 0.0

    # --- GIAI ĐOẠN 1: ĐÓNG BĂNG BACKBONE, KHỞI ĐỘNG CLASSIFIER HEAD ---
    print(f"\n=== STAGE 1: Train Head ({cfg.STAGE1_EPOCHS} epochs) ===")
    set_backbone_trainable(model, trainable=False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.STAGE1_LR, weight_decay=cfg.WEIGHT_DECAY
    )

    for epoch in range(1, cfg.STAGE1_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, cfg.DEVICE)
        va_loss, va_top1, va_top5 = validate(model, valid_loader, criterion, cfg.DEVICE)
        print(f"[S1 {epoch:02d}/{cfg.STAGE1_EPOCHS}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Top1: {va_top1:.4f} Top5: {va_top5:.4f}")

        if va_top1 > best_val_acc:
            best_val_acc = va_top1
            save_checkpoint(cfg.BEST_MODEL_PATH, model, cfg, class_to_idx, idx_to_name, best_val_acc)

    # --- GIAI ĐOẠN 2: MỞ BĂNG TOÀN MẠNG, FINE-TUNE VỚI DISCRIMINATIVE LR ---
    print(f"\n=== STAGE 2: Fine-tune Full Model ({cfg.STAGE2_EPOCHS} epochs) ===")
    set_backbone_trainable(model, trainable=True)
    param_groups = get_param_groups(model, cfg.STAGE2_BACKBONE_LR, cfg.STAGE2_HEAD_LR)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.STAGE2_EPOCHS, eta_min=1e-6)

    no_improve = 0
    for epoch in range(1, cfg.STAGE2_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, cfg.DEVICE)
        va_loss, va_top1, va_top5 = validate(model, valid_loader, criterion, cfg.DEVICE)
        scheduler.step()

        print(f"[S2 {epoch:02d}/{cfg.STAGE2_EPOCHS}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Top1: {va_top1:.4f} Top5: {va_top5:.4f}")

        if va_top1 > best_val_acc:
            best_val_acc = va_top1
            no_improve = 0
            save_checkpoint(cfg.BEST_MODEL_PATH, model, cfg, class_to_idx, idx_to_name, best_val_acc)
            print(f"  -> Đã lưu model tốt nhất (Val Top-1: {best_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg.PATIENCE:
                print(f"Early stopping tại epoch {epoch}. Best Val Top-1: {best_val_acc:.4f}")
                break


if __name__ == "__main__":
    main()