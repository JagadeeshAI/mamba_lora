import os
import torch

class Config:
    # Data paths
    DATA_DIR = "/home/jag/codes/VIM_lora/data"
    FULL_TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train")
    FULL_VAL_DATA_PATH = os.path.join(DATA_DIR, "val")
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Data preprocessing
    IMAGE_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    class TRAIN:
        OUT_DIR = "./results/train"
        BATCH_SIZE = 4
        EPOCHS = 300
        LR = 3e-4
        WEIGHT_DECAY = 0.05
        SNAPSHOT_INTERVAL = 10  # Save snapshot every 10% of epoch progress
        MAX_SNAPSHOTS = 5  # Keep only last 5 snapshots per epoch
        
        @staticmethod
        def model_path():
            return os.path.join(Config.TRAIN.OUT_DIR, "best_model.pth")
        
        @staticmethod
        def progress_path():
            return os.path.join(Config.TRAIN.OUT_DIR, "progress.json")
        
        @staticmethod
        def snapshot_path(epoch, progress_pct):
            return os.path.join(Config.TRAIN.OUT_DIR, f"snapshot_epoch{epoch}_p{progress_pct}.pth")

# Ensure all output directories exist
for dir_path in [
    Config.DATA_DIR,
    Config.TRAIN.OUT_DIR,
]:
    os.makedirs(dir_path, exist_ok=True)

print(f"✅ Device used: {Config.DEVICE}")