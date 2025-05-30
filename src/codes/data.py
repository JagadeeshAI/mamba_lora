import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from src.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load data samples and create class mappings"""
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} does not exist")
            return
        
        # Get class directories
        class_dirs = [d for d in os.listdir(self.data_path) 
                     if os.path.isdir(os.path.join(self.data_path, d))]
        class_dirs.sort()
        
        self.classes = class_dirs
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        
        # Load samples
        for class_name in class_dirs:
            class_path = os.path.join(self.data_path, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_path, filename)
                    self.samples.append((img_path, class_idx))
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor if image fails to load
            if self.transform:
                dummy_image = self.transform(Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE)))
            else:
                dummy_image = torch.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            return dummy_image, label

def get_train_transforms():
    """Get training data transformations"""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])

def get_val_transforms():
    """Get validation/test data transformations"""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])

def get_full_train_loader():
    """Get full training data loader"""
    transform = get_train_transforms()
    dataset = ImageDataset(Config.FULL_TRAIN_DATA_PATH, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=Config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True
    )

def get_full_val_loader():
    """Get full validation data loader"""
    transform = get_val_transforms()
    dataset = ImageDataset(Config.FULL_VAL_DATA_PATH, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=Config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

def main():
    """Test all data loaders"""
    print("=" * 60)
    print("TESTING DATA LOADERS")
    print("=" * 60)
    
    # Test train loader
    print("\n🔍 Testing Train Loader...")
    try:
        train_loader = get_full_train_loader()
        print(f"✅ Train dataset size: {len(train_loader.dataset)}")
        print(f"✅ Train batches: {len(train_loader)}")
        
        # Test one batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"✅ Train batch shape: {images.shape}, Labels shape: {labels.shape}")
            print(f"✅ Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"✅ Unique labels in batch: {torch.unique(labels).tolist()}")
            if len(train_loader.dataset.classes) > 0:
                print(f"✅ Classes: {train_loader.dataset.classes[:5]}...")  # Show first 5 classes
            break
    except Exception as e:
        print(f"❌ Train loader failed: {e}")
    
    # Test validation loader
    print("\n🔍 Testing Validation Loader...")
    try:
        val_loader = get_full_val_loader()
        print(f"✅ Val dataset size: {len(val_loader.dataset)}")
        print(f"✅ Val batches: {len(val_loader)}")
        
        for batch_idx, (images, labels) in enumerate(val_loader):
            print(f"✅ Val batch shape: {images.shape}, Labels shape: {labels.shape}")
            break
    except Exception as e:
        print(f"❌ Validation loader failed: {e}")
    
    
    # Configuration summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"📁 Data directory: {Config.DATA_DIR}")
    print(f"🖥️  Device: {Config.DEVICE}")
    print(f"🖼️  Image size: {Config.IMAGE_SIZE}")
    print(f"📊 Normalization - Mean: {Config.MEAN}")
    print(f"📊 Normalization - Std: {Config.STD}")
    print(f"🔄 Num workers: {Config.NUM_WORKERS}")
    print(f"📌 Pin memory: {Config.PIN_MEMORY}")    
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()