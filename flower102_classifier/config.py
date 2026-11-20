import os
from pathlib import Path
import torch

class Config:
    DATA_DIR = Path(os.getenv("FLOWER_DATA_DIR", "data"))
    TRAIN_DIR = DATA_DIR / "train"
    VALID_DIR = DATA_DIR / "valid"
    TEST_DIR = DATA_DIR / "test"
    CAT_TO_NAME_PATH = DATA_DIR / "cat_to_name.json"

    CHECKPOINT_DIR = Path(os.getenv("FLOWER_CKPT_DIR", "checkpoints"))
    BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

    MODEL_NAME = "convnext_tiny"
    NUM_CLASSES = 102
    DROPOUT = 0.3

    IMG_SIZE = 224
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    BATCH_SIZE = 32
    NUM_WORKERS = int(os.getenv("FLOWER_NUM_WORKERS", "2"))
    STAGE1_EPOCHS = 5
    STAGE1_LR = 1e-3

    STAGE2_EPOCHS = 30
    STAGE2_BACKBONE_LR = 5e-5
    STAGE2_HEAD_LR = 5e-4

    WEIGHT_DECAY = 1e-2
    LABEL_SMOOTHING = 0.1
    PATIENCE = 7
    SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"