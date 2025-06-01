import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import logging
import random
import numpy as np
from typing import List, Dict, Optional
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None, class_filter=None):
        self.data_path = data_path
        self.transform = transform
        self.class_filter = class_filter
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        self._load_data()
    
    def _load_data(self):
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} does not exist")
            return
        
        all_class_dirs = [d for d in os.listdir(self.data_path) 
                          if os.path.isdir(os.path.join(self.data_path, d))]
        all_class_dirs.sort()
        
        if self.class_filter is not None:
            class_dirs = [d for d in all_class_dirs if d in self.class_filter]
        else:
            class_dirs = all_class_dirs
        
        self.classes = class_dirs
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        
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
            size = getattr(Config, "IMAGE_SIZE", 224)
            dummy_image = self.transform(Image.new("RGB", (size, size))) if self.transform else torch.zeros((3, size, size))
            return dummy_image, label


class ContinualForgettingManager:
    def __init__(self, num_classes=100, classes_per_task=20, seed=42):
        self.num_classes = num_classes
        self.classes_per_task = classes_per_task
        self.num_tasks = num_classes // classes_per_task
        self.seed = seed
        self.all_classes = None
        self.task_splits = None
        
    def set_class_names(self, class_names: List[str]):
        if len(class_names) != self.num_classes:
            logger.warning(f"Expected {self.num_classes} classes, got {len(class_names)}")
            self.num_classes = len(class_names)
            self.num_tasks = self.num_classes // self.classes_per_task
        
        self.all_classes = sorted(class_names)
        self._create_task_splits()
    
    def _create_task_splits(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        shuffled_classes = self.all_classes.copy()
        random.shuffle(shuffled_classes)
        self.task_splits = {}
        for task_id in range(self.num_tasks):
            start_idx = task_id * self.classes_per_task
            end_idx = start_idx + self.classes_per_task
            self.task_splits[task_id] = shuffled_classes[start_idx:end_idx]
        logger.info(f"Created {self.num_tasks} task splits with {self.classes_per_task} classes each")
    
    def get_task_classes(self, task_id: int) -> Dict[str, List[str]]:
        if task_id == 0:
            return {
                'retained': self.all_classes.copy(),
                'current_forgotten': [],
                'old_forgotten': []
            }
        forgotten_classes = []
        for t in range(1, task_id + 1):
            forgotten_classes.extend(self.task_splits[t])
        old_forgotten_classes = []
        for t in range(1, task_id):
            old_forgotten_classes.extend(self.task_splits[t])
        current_forgotten_classes = self.task_splits[task_id]
        retained_classes = [cls for cls in self.all_classes if cls not in forgotten_classes]
        return {
            'retained': retained_classes,
            'current_forgotten': current_forgotten_classes,
            'old_forgotten': old_forgotten_classes
        }
    
    def print_task_summary(self, task_id: int):
        task_classes = self.get_task_classes(task_id)
        print(f"\n{'='*60}")
        print(f"TASK T{task_id} SUMMARY")
        print(f"{'='*60}")
        print(f"📊 Retained classes: {len(task_classes['retained'])}")
        print(f"🗑️  Current forgotten: {len(task_classes['current_forgotten'])}")
        print(f"📚 Old forgotten: {len(task_classes['old_forgotten'])}")
        if task_classes['retained']:
            print(f"✅ Retained: {task_classes['retained'][:3]}...{task_classes['retained'][-3:]}")
        if task_classes['current_forgotten']:
            print(f"❌ Current forgotten: {task_classes['current_forgotten'][:3]}...{task_classes['current_forgotten'][-3:]}")
        if task_classes['old_forgotten']:
            print(f"📋 Old forgotten: {task_classes['old_forgotten'][:3]}...{task_classes['old_forgotten'][-3:]}")


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])


def get_full_train_loader():
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
    transform = get_val_transforms()
    dataset = ImageDataset(Config.FULL_VAL_DATA_PATH, transform=transform)
    return DataLoader(
        dataset,
        batch_size=Config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )


def get_task_validation_loaders(task_id: int, forgetting_manager: ContinualForgettingManager) -> Dict[str, DataLoader]:
    transform = get_val_transforms()
    task_classes = forgetting_manager.get_task_classes(task_id)
    loaders = {}

    def loader_for_classes(class_list, name):
        if not class_list:
            return None
        dataset = ImageDataset(Config.FULL_VAL_DATA_PATH, transform=transform, class_filter=class_list)
        return DataLoader(dataset, batch_size=Config.TRAIN.BATCH_SIZE, shuffle=False,
                          num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    loaders['retained'] = loader_for_classes(task_classes['retained'], 'retained')
    loaders['current_forgotten'] = loader_for_classes(task_classes['current_forgotten'], 'current_forgotten')
    loaders['old_forgotten'] = loader_for_classes(task_classes['old_forgotten'], 'old_forgotten')

    return {k: v for k, v in loaders.items() if v is not None}


def get_task_forgetting_loader(task_id: int, forgetting_manager: ContinualForgettingManager) -> DataLoader:
    transform = get_train_transforms()
    task_classes = forgetting_manager.get_task_classes(task_id)
    dataset = ImageDataset(
        Config.FULL_TRAIN_DATA_PATH,
        transform=transform,
        class_filter=task_classes['current_forgotten']
    )
    total = len(dataset)
    indices = np.random.choice(total, int(0.1 * total), replace=False)
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=Config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True
    )


def get_continual_forgetting_manager() -> ContinualForgettingManager:
    temp_dataset = ImageDataset(Config.FULL_VAL_DATA_PATH)
    manager = ContinualForgettingManager(
        num_classes=len(temp_dataset.classes),
        classes_per_task=20,
        seed=42
    )
    manager.set_class_names(temp_dataset.classes)
    return manager


def calculate_harmonic_mean(acc_r: float, acc_f: float) -> float:
    if acc_r + acc_f == 0:
        return 0.0
    return 2 * acc_r * acc_f / (acc_r + acc_f)


def main():
    print("=" * 80)
    print("🔍 TESTING CONTINUAL FORGETTING LOADERS")
    print("=" * 80)

    # ----------------------------------------------------------------------------
    print("\n✅ Testing Full Training Loader")
    try:
        train_loader = get_full_train_loader()
        print(f"   - Train dataset size: {len(train_loader.dataset)}")
        print(f"   - Train batches: {len(train_loader)}")
        batch = next(iter(train_loader))
        print(f"   - Example batch: images={batch[0].shape}, labels={batch[1].shape}")
    except Exception as e:
        print(f"❌ Full train loader failed: {e}")

    # ----------------------------------------------------------------------------
    print("\n✅ Testing Full Validation Loader")
    try:
        val_loader = get_full_val_loader()
        print(f"   - Val dataset size: {len(val_loader.dataset)}")
        print(f"   - Val batches: {len(val_loader)}")
        batch = next(iter(val_loader))
        print(f"   - Example batch: images={batch[0].shape}, labels={batch[1].shape}")
    except Exception as e:
        print(f"❌ Full val loader failed: {e}")

    # ----------------------------------------------------------------------------
    print("\n✅ Testing Task-specific Loaders for Continual Forgetting")
    try:
        forgetting_manager = get_continual_forgetting_manager()
        for task_id in range(1, 5):
            forgetting_manager.print_task_summary(task_id)
            task_loaders = get_task_validation_loaders(task_id, forgetting_manager)
            for name, loader in task_loaders.items():
                print(f"   - Task {task_id} [{name}] -> {len(loader.dataset)} samples, {len(loader)} batches")
                batch = next(iter(loader))
                print(f"     · Sample batch: images={batch[0].shape}, labels={batch[1].shape}")
    except Exception as e:
        print(f"❌ Task-specific loaders failed: {e}")

    # ----------------------------------------------------------------------------
    print("\n✅ Testing Forgetting Loader (10% Sample)")
    try:
        for task_id in range(1, 5):
            loader = get_task_forgetting_loader(task_id, forgetting_manager)
            print(f"   - Task {task_id} forgetting loader: {len(loader.dataset)} samples, {len(loader)} batches")
            batch = next(iter(loader))
            print(f"     · Sample batch: images={batch[0].shape}, labels={batch[1].shape}")
    except Exception as e:
        print(f"❌ Forgetting loader failed: {e}")


if __name__ == "__main__":
    main()

