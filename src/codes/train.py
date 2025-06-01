import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import json
import torch
import shutil
import logging
from tqdm import tqdm
from pathlib import Path
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.model.arch2 import VisionMamba
from src.config import Config
from data import get_full_train_loader, get_full_val_loader,get_missing_loader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model():
    """Initialize VisionMamba model"""
    model = VisionMamba(
        patch_size=16,
        stride=8,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True,
        num_classes=100,              
        drop_rate=0.0,               
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=Config.IMAGE_SIZE,
    )
    return model

def evaluate(model, loader, device):
    """Evaluate model on validation set"""
    model.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss

def cleanup_snapshots(outdir, epoch, max_keep=None):
    """Clean up old snapshots for a specific epoch"""
    if max_keep is None:
        max_keep = Config.TRAIN.MAX_SNAPSHOTS
    
    pattern = f"snapshot_epoch{epoch}_p*.pth"
    snapshots = sorted(Path(outdir).glob(pattern), key=os.path.getmtime)
    
    while len(snapshots) > max_keep:
        old_snapshot = snapshots.pop(0)
        try:
            os.remove(old_snapshot)
            logger.info(f"🗑️ Cleaned old snapshot: {old_snapshot.name}")
        except OSError:
            pass

def load_progress():
    """Load training progress from JSON file"""
    progress_path = Config.TRAIN.progress_path()
    
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r") as f:
                progress = json.load(f)
                logger.info(f"📄 Loaded progress from {progress_path}")
                return progress
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"⚠️ Invalid progress file, starting fresh: {e}")
    
    return {
        "best_val_acc": 0.0,
        "best_model_path": None,
        "current_epoch": 0,
        "current_epoch_progress": 0,
        "current_snapshot_path": None,
        "total_epochs": Config.TRAIN.EPOCHS,
        "completed": False
    }

def save_progress(epoch, epoch_progress, best_val_acc, best_model_path, current_snapshot_path=None):
    """Save training progress to JSON file"""
    progress_data = {
        "best_val_acc": float(best_val_acc),
        "best_model_path": str(best_model_path) if best_model_path else None,
        "current_epoch": int(epoch),
        "current_epoch_progress": int(epoch_progress),
        "current_snapshot_path": str(current_snapshot_path) if current_snapshot_path else None,
        "total_epochs": Config.TRAIN.EPOCHS,
        "completed": epoch >= Config.TRAIN.EPOCHS
    }
    
    progress_path = Config.TRAIN.progress_path()
    with open(progress_path, "w") as f:
        json.dump(progress_data, f, indent=4)

def save_snapshot(model, optimizer, scheduler, epoch, progress_pct):
    """Save training snapshot"""
    snapshot_path = Config.TRAIN.snapshot_path(epoch, progress_pct)
    
    snapshot_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'progress_pct': progress_pct
    }
    
    torch.save(snapshot_data, snapshot_path)
    logger.info(f"📸 Saved snapshot: {Path(snapshot_path).name}")
    
    # Clean up old snapshots for this epoch
    cleanup_snapshots(Config.TRAIN.OUT_DIR, epoch)
    
    return snapshot_path

def load_snapshot(model, optimizer, scheduler, snapshot_path):
    """Load training snapshot"""
    if not os.path.exists(snapshot_path):
        logger.warning(f"⚠️ Snapshot not found: {snapshot_path}")
        return False
    
    try:
        checkpoint = torch.load(snapshot_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"🔄 Loaded snapshot: {Path(snapshot_path).name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load snapshot {snapshot_path}: {e}")
        return False

def train():
    """Main training function"""
    device = Config.DEVICE
    outdir = Path(Config.TRAIN.OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🧠 Starting VisionMamba training from scratch...")
    logger.info(f"📁 Output directory: {outdir}")
    logger.info(f"🖥️ Device: {device}")
    
    # Load data
    train_loader = get_full_train_loader()
    val_loader = get_full_val_loader()
    
    logger.info(f"📊 Train samples: {len(train_loader.dataset)}")
    logger.info(f"📊 Val samples: {len(val_loader.dataset)}")
    logger.info(f"📊 Train batches: {len(train_loader)}")
    logger.info(f"📊 Val batches: {len(val_loader)}")
    
    # Initialize model and training components
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=Config.TRAIN.LR, weight_decay=Config.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.TRAIN.EPOCHS)
    
    # Load progress and resume if needed
    progress = load_progress()
    start_epoch = progress["current_epoch"]
    best_val_acc = progress["best_val_acc"]
    best_model_path = progress["best_model_path"]
    current_snapshot_path = progress["current_snapshot_path"]
    
    # Resume from snapshot if available
    if current_snapshot_path and load_snapshot(model, optimizer, scheduler, current_snapshot_path):
        logger.info(f"🔄 Resumed from epoch {start_epoch}, progress {progress['current_epoch_progress']}%")
    elif best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"🔄 Loaded best model: {best_model_path} (Val Acc: {best_val_acc:.2f}%)")
    
    if progress["completed"]:
        logger.info("✅ Training already completed!")
        return
    
    # Training loop
    for epoch in range(max(1, start_epoch), Config.TRAIN.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        epoch_start_batch = 0
        
        # Resume from specific batch if snapshot exists
        if epoch == start_epoch and progress["current_epoch_progress"] > 0:
            epoch_start_batch = int(len(train_loader) * progress["current_epoch_progress"] / 100)
            logger.info(f"🔄 Resuming epoch {epoch} from batch {epoch_start_batch}")
        
        # Training progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.TRAIN.EPOCHS}", unit="batch")
        
        for batch_idx, (images, labels) in enumerate(loop):
            # Skip batches if resuming
            if batch_idx < epoch_start_batch:
                continue
                
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            
            # Calculate progress percentage
            progress_pct = int((batch_idx + 1) * 100 / len(train_loader))
            
            # Save snapshot at specified intervals
            if progress_pct % Config.TRAIN.SNAPSHOT_INTERVAL == 0 and progress_pct > 0:
                if progress_pct != (batch_idx * 100 // len(train_loader)):  # Avoid duplicate saves
                    snapshot_path = save_snapshot(model, optimizer, scheduler, epoch, progress_pct)
                    save_progress(epoch, progress_pct, best_val_acc, best_model_path, snapshot_path)
        
        # End of epoch
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        logger.info(f"📉 Epoch {epoch} — Training Loss: {avg_loss:.4f}")
        val_acc, val_loss = evaluate(model, val_loader, device)
        logger.info(f"🧪 Epoch {epoch} — Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = Config.TRAIN.model_path()
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"💾 New best model saved! Val Acc: {val_acc:.2f}% → {best_model_path}")
        
        # Save progress (epoch completed)
        save_progress(epoch, 100, best_val_acc, best_model_path, None)
        
        # Clean up snapshots from this epoch (keep only final snapshot)
        cleanup_snapshots(Config.TRAIN.OUT_DIR, epoch, max_keep=1)
    
    logger.info("✅ Training completed successfully!")
    logger.info(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"💾 Best model saved at: {best_model_path}")

if __name__ == "__main__":
    train()