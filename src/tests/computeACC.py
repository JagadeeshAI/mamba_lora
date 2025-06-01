import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import json
import torch
from torch import nn
from collections import defaultdict
from tqdm import tqdm

from src.model.arch2 import VisionMamba
from src.config import Config
from src.codes.data import get_full_val_loader

def get_model():
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

def evaluate_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if pred.item() == label.item():
                    class_correct[label.item()] += 1

    overall_acc = 100.0 * total_correct / total_samples
    class_accuracies = {
        str(cls): 100.0 * class_correct[cls] / class_total[cls]
        for cls in class_total
    }

    return overall_acc, class_accuracies

def main():
    device = Config.DEVICE
    model = get_model().to(device)

    model_path = Config.TRAIN.model_path()
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from: {model_path}")

    val_loader = get_full_val_loader()
    print(f"🔍 Evaluating on {len(val_loader.dataset)} samples...")

    overall_acc, class_accuracies = evaluate_model(model, val_loader, device)

    results = {
        "overall_accuracy": overall_acc,
        "class_accuracies": class_accuracies
    }

    os.makedirs("results", exist_ok=True)
    with open("results/test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results saved to results/test_results.json")
    print(f"📊 Overall Accuracy: {overall_acc:.2f}%")

if __name__ == "__main__":
    main()
