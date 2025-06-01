# ==================== compute_stats.py ====================

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

from data import prepare_data_loaders_for_task
from config import Config
from forget import backbone_forgetting
from arch2 import VisionMamba

import os
import csv
import torch
from tqdm import tqdm
import torch.nn.functional as F
import json
import numpy as np

BASE_ACCURACY = 62.36  # Best baseline accuracy

def print_lora_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)

    print(f"\n📊 Model Parameters Summary:")
    print(f"🔹 Total Parameters:     {total_params:,}")
    print(f"🔹 Trainable Parameters: {trainable_params:,}")
    print(f"🔹 LoRA Parameters Only: {lora_params:,}")

def load_pretrained_model():
    local_ckpt = os.path.join(Config.FINETUNE.OUT_DIR, "43.pth")
    if not os.path.exists(local_ckpt):
        raise FileNotFoundError(f"❌ Pretrained model not found at {local_ckpt}")

    checkpoint = torch.load(local_ckpt, map_location="cpu")

    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=384, depth=24,
        rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type="mean", if_abs_pos_embed=True, if_rope=False,
        if_rope_residual=False, bimamba_type="v2", if_cls_token=True,
        if_devide_out=True, use_middle_cls_token=True, num_classes=100,
        drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None,
        img_size=224, lora_out_proj=True, lora_r=96, lora_alpha=0.1
    )

    model.load_state_dict(checkpoint, strict=False)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    print_lora_stats(model)
    return model

def compute_accuracy(model, dataloader, device, name=""):
    model.eval()
    correct = total = 0
    labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=f"Evaluating {name}", leave=False):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            labels.extend(y.cpu().numpy())

    if name:
        unique_labels = np.unique(labels)
        print(f"🧪 {name} contains labels: {sorted(unique_labels.tolist())}")
        print(f"✅ {name}: {correct}/{total} correct")

    return 100.0 * correct / total if total > 0 else 0.0

def compute_harmonic_mean(accr, accf):
    accf_complement = 1 - accf / 100.0
    accr_ratio = accr / 100.0
    if accf_complement + accr_ratio == 0:
        return 0.0
    h = 2 * accf_complement * accr_ratio / (accf_complement + accr_ratio)
    return h * 100.0

def normalize_accr(accr, base=BASE_ACCURACY):
    return min((accr / base) * 100, 100.0)

def compute_normalized_h(accr, accf, base=BASE_ACCURACY):
    accr_norm = normalize_accr(accr, base)
    return compute_harmonic_mean(accr_norm, accf)

def run_all_tasks():
    device = Config.DEVICE
    results = []
    resume_path = os.path.join(Config.FORGET.OUT_DIR, "compute_resume.json")
    out_csv = os.path.join(Config.FORGET.OUT_DIR, "task_results.csv")

    if Config.FORGET.COPM_RESUME and os.path.exists(resume_path):
        with open(resume_path, "r") as f:
            resume_data = json.load(f)
        start_task = resume_data.get("task", 0)
        model_path = resume_data.get("model_path", os.path.join(Config.FINETUNE.OUT_DIR, "43.pth"))
        start_epoch = resume_data.get("epoch", 0)
        model = load_pretrained_model()
    else:
        start_task = 0
        start_epoch = 0
        model = load_pretrained_model()

    if os.path.exists(out_csv):
        with open(out_csv, "r") as f:
            reader = csv.DictReader(f)
            results = list(reader)
        print(f"🔁 Resuming stats log with {len(results)} existing entries.")

    for task_id in range(start_task, 4):
        print(f"\n================ Task {task_id} ================")
        loaders = prepare_data_loaders_for_task(task_id, Config.DATA_DIR)

        model = backbone_forgetting(
            model=model,
            device=device,
            loaders=loaders,
            task_id=task_id,
            config=Config,
            start_epoch=start_epoch,
            resume_path=resume_pathz
        )

        accf = compute_accuracy(model, loaders["val_f"], device, name="val_f (forget)")
        accr = compute_accuracy(model, loaders["val_r"], device, name="val_r (retain)")

        acco = 0.0
        if task_id > 0:
            acco = compute_accuracy(model, loaders["val_o"], device, name="val_o (old forget)")
        else:
            print("ℹ️ Skipping Acco for Task 0 (no prior forgets)")

        hmean = compute_harmonic_mean(accr, accf)
        norm_accr = normalize_accr(accr, BASE_ACCURACY)
        norm_h = compute_normalized_h(accr, accf, BASE_ACCURACY)

        if accf > 30:
            print(f"⚠️ High Accf ({accf:.2f}%) → Forgetting may have failed.")
        if hmean > 90 and accf > 20:
            print(f"⚠️ Suspiciously high H ({hmean:.2f}%) given Accf = {accf:.2f}%")

        results.append({
            "Task": task_id,
            "Accr": round(accr, 2),
            "Accf": round(accf, 2),
            "Acco": round(acco, 2),
            "H": round(hmean, 2),
            "NormAccr": round(norm_accr, 2),
            "NormH": round(norm_h, 2)
        })

        print(f"✅ Task {task_id} Results:")
        print(f"   Accr: {accr:.2f}% | NormAccr: {norm_accr:.2f}%")
        print(f"   Accf: {accf:.2f}%")
        print(f"   Acco: {acco:.2f}%")
        print(f"   H:    {hmean:.2f}% | NormH: {norm_h:.2f}%")

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Task", "Accr", "Accf", "Acco", "H", "NormAccr", "NormH"])
            writer.writeheader()
            writer.writerows(results)

    print(f"\n📁 All task stats saved to {out_csv}")

if __name__ == "__main__":
    run_all_tasks()
