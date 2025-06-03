import os
import torch

class Config:
    DATA_DIR = "/home/jag/codes/VIM_lora/data"
    FULL_TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train")
    FULL_VAL_DATA_PATH = os.path.join(DATA_DIR, "val")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True

    IMAGE_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

   
    class FINETUNE:
        OUT_DIR = "./results/finetune"
        MODEL_PATH = os.path.join(OUT_DIR, "43.pth")

   
    class FORGET:
        OUT_DIR = "./results/forget"
        EPOCHS = 30
        LR = 2e-4
        WEIGHT_DECAY = 1e-4
        BND = 10
        BETA = 0.2 
        COPM_RESUME = False  
        DATA_RATIO = 0.1  

    class TRAIN:
        OUT_DIR = "./results/train"
        BATCH_SIZE = 64
        EPOCHS = 300
        LR = 3e-4
        WEIGHT_DECAY = 0.05
        SNAPSHOT_INTERVAL = 10
        MAX_SNAPSHOTS = 5

        @staticmethod
        def model_path():
            return os.path.join(Config.TRAIN.OUT_DIR, "best_model.pth")

        @staticmethod
        def progress_path():
            return os.path.join(Config.TRAIN.OUT_DIR, "progress.json")

        @staticmethod
        def snapshot_path(epoch, progress_pct):
            return os.path.join(Config.TRAIN.OUT_DIR, f"snapshot_epoch{epoch}_p{progress_pct}.pth")

for dir_path in [
    Config.DATA_DIR,
    Config.TRAIN.OUT_DIR,
    Config.FINETUNE.OUT_DIR,
    Config.FORGET.OUT_DIR,
]:
    os.makedirs(dir_path, exist_ok=True)

print(f"✅ Device used: {Config.DEVICE}")
