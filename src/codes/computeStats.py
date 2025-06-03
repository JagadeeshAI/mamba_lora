import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import json
import csv
import torch
import numpy as np
from tqdm import tqdm
from backboneForget import backbone_forgetting
from src.config import Config
from src.codes.data import (
    get_continual_forgetting_manager,
    get_task_validation_loaders,
    get_task_forgetting_loader,
    get_full_train_loader,
    get_val_transforms,
    ImageDataset
)

from src.model.arch2 import VisionMamba


BASE_ACCURACY = 58.26


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

def print_lora_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)

    trainable_pct = (trainable_params / total_params) * 100 if total_params else 0
    lora_pct = (lora_params / total_params) * 100 if total_params else 0

    print(f"\n📊 Model Parameters Summary:")
    print(f"🔹 Total Parameters:     {total_params:,}")
    print(f"🔹 Trainable Parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"🔹 LoRA Parameters Only: {lora_params:,} ({lora_pct:.2f}%)")



def load_pretrained_model():
    local_ckpt = "results/train/best_model.pth"
    if not os.path.exists(local_ckpt):
        raise FileNotFoundError(f"❌ Pretrained model not found at {local_ckpt}")

    checkpoint = torch.load(local_ckpt, map_location="cpu")

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
    img_size=224,
    use_peft=True,
    lora_out_proj=True,
    lora_d=True,
    lora_B=True,
    lora_in_proj=True,
    lora_out_proj_mamba=True,  
    lora_x_proj=True,
    additional_scan=True,
    learnable_D_v2=True,
    learnable_bias_v2=True,
    lora_r=16,
    lora_alpha=32
    )


    model.load_state_dict(checkpoint, strict=False)

    print_lora_stats(model)

    return model


def run_all_tasks():
    device = Config.DEVICE
    results = []
    resume_path = os.path.join(Config.FORGET.OUT_DIR, "compute_resume.json")
    out_csv = os.path.join(Config.FORGET.OUT_DIR, "task_results.csv")

    # Resume setup
    if Config.FORGET.COPM_RESUME and os.path.exists(resume_path):
        with open(resume_path, "r") as f:
            resume_data = json.load(f)
        start_task = resume_data.get("task", 0)
        start_epoch = resume_data.get("epoch", 0)
    else:
        start_task = 4
        start_epoch = 0

    model = load_pretrained_model()
    forgetting_manager = get_continual_forgetting_manager()

    # Resume CSV logging
    if os.path.exists(out_csv):
        with open(out_csv, "r") as f:
            reader = csv.DictReader(f)
            results = list(reader)
        print(f"🔁 Resuming stats log with {len(results)} existing entries.")

    for task_id in range(start_task, 5):
        print(f"\n================ Task {task_id} ================")

        # ✅ Skip if forgetting data is not available
        train_f_loader = get_task_forgetting_loader(task_id, forgetting_manager)
        if train_f_loader is None:
            print(f"⏭️  Skipping Task {task_id}: No forgetting data available.")
            continue

        val_loaders = get_task_validation_loaders(task_id, forgetting_manager)
        train_r_loader = get_full_train_loader()

        model = backbone_forgetting(
            model=model,
            device=device,
            loaders={
                "train_f": train_f_loader,
                "train_r": train_r_loader,
                "val_r": val_loaders.get("retained"),
                "val_f": val_loaders.get("current_forgotten"),
                "val_o": val_loaders.get("old_forgotten")
            },
            task_id=task_id,
            config=Config,
            start_epoch=start_epoch,
            resume_path=resume_path
        )

        accf = compute_accuracy(model, val_loaders["current_forgotten"], device, name="val_f (forget)")
        accr = compute_accuracy(model, val_loaders["retained"], device, name="val_r (retain)")

        acco = 0.0
        if "old_forgotten" in val_loaders and val_loaders["old_forgotten"] is not None:
            acco = compute_accuracy(model, val_loaders["old_forgotten"], device, name="val_o (old forget)")
        else:
            print("ℹ️ Skipping Acco for Task 0")

        hmean = compute_harmonic_mean(accr, accf)
        norm_accr = normalize_accr(accr, BASE_ACCURACY)
        norm_h = compute_normalized_h(accr, accf, BASE_ACCURACY)

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
