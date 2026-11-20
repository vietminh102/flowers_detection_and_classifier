import json
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import Config


def get_transforms(img_size: int = Config.IMG_SIZE, mean=Config.MEAN, std=Config.STD):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf


def build_idx_to_name(class_to_idx: Dict[str, int], cat_to_name_path: Path) -> Dict[int, str]:
    with open(cat_to_name_path, "r", encoding="utf-8") as f:
        cat_to_name = json.load(f)

    idx_to_name = {}
    for folder_id, idx in class_to_idx.items():
        if folder_id not in cat_to_name:
            raise KeyError(f"Thư mục '{folder_id}' không tồn tại trong {cat_to_name_path}")
        idx_to_name[idx] = cat_to_name[folder_id]
    return idx_to_name


def create_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    train_tf, eval_tf = get_transforms(cfg.IMG_SIZE, cfg.MEAN, cfg.STD)

    train_ds = datasets.ImageFolder(cfg.TRAIN_DIR, transform=train_tf)
    valid_ds = datasets.ImageFolder(cfg.VALID_DIR, transform=eval_tf)
    test_ds = datasets.ImageFolder(cfg.TEST_DIR, transform=eval_tf) if cfg.TEST_DIR.exists() else None

    assert train_ds.class_to_idx == valid_ds.class_to_idx, (
        "Danh sách thư mục lớp giữa train và valid không khớp nhau!"
    )

    idx_to_name = build_idx_to_name(train_ds.class_to_idx, cfg.CAT_TO_NAME_PATH)
    pin_mem = cfg.DEVICE == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=pin_mem, drop_last=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=pin_mem
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=pin_mem
        )

    return train_loader, valid_loader, test_loader, train_ds.class_to_idx, idx_to_name