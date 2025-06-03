import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def retention_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def forgetting_loss(logits, labels, bnd):
    ce = F.cross_entropy(logits, labels)
    return F.relu(bnd - ce)


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def compute_hmean(accr, accf):
    accf_complement = 1 - accf / 100.0
    accr_ratio = accr / 100.0
    if accf_complement + accr_ratio == 0:
        return 0.0
    return 2 * accf_complement * accr_ratio / (accf_complement + accr_ratio) * 100.0


def backbone_forgetting(model, device, loaders, task_id, config, start_epoch=0, resume_path=None):
    model.to(device).train()
    print(f"\n🚀 Starting Forgetting on Task {task_id} (from epoch {start_epoch})...")
    os.makedirs(config.FORGET.OUT_DIR, exist_ok=True)

    loader_r = loaders["train_r"]
    loader_f = loaders["train_f"]

    optimizer = AdamW(model.parameters(), lr=config.FORGET.LR, weight_decay=config.FORGET.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.FORGET.EPOCHS)

    forget_cycle = cycle(loader_f)

    best_hmean = -1
    patience_counter = 0
    best_ckpt_path = os.path.join(config.FORGET.OUT_DIR, f"task{task_id}.pth")

    for epoch in range(start_epoch + 1, config.FORGET.EPOCHS + 1):
        model.train()
        loop = tqdm(loader_r, total=len(loader_r), desc=f"[Task {task_id}] Epoch {epoch}", unit="batch")

        for xr, yr in loop:
            xf, yf = next(forget_cycle)
            xr, yr = xr.to(device), yr.to(device)
            xf, yf = xf.to(device), yf.to(device)

            logits_r = model(xr)
            loss_r = retention_loss(logits_r, yr)

            logits_f = model(xf)
            loss_f = forgetting_loss(logits_f, yf, bnd=config.FORGET.BND)

            loss = loss_r + config.FORGET.BETA * loss_f

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loop.set_postfix({
                "Ret": f"{loss_r:.4f}",
                "For": f"{loss_f:.4f}"
            })

        scheduler.step()

        # Evaluate after each epoch
        val_acc_r = compute_accuracy(model, loaders["val_r"], device)
        val_acc_f = compute_accuracy(model, loaders["val_f"], device)
        val_acc_o = compute_accuracy(model, loaders.get("val_o"), device) if loaders.get("val_o") else 0.0
        hmean = compute_hmean(val_acc_r, val_acc_f)

        print(f"🧪 [Epoch {epoch:3d}] Retain Acc: {val_acc_r:6.2f}% | Forget Acc: {val_acc_f:6.2f}% | Old Forget Acc: {val_acc_o:6.2f}% | H: {hmean:.2f}%")

        # Early stopping: update best H
        if hmean > best_hmean:
            best_hmean = hmean
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"💾 Saved best model so far: {best_ckpt_path}")
        else:
            patience_counter += 1
            print(f"⏳ No H improvement ({patience_counter}/10)")

        # Resume tracking
        if resume_path:
            resume_info = {
                "task": task_id,
                "epoch": epoch,
                "model_path": best_ckpt_path
            }
            with open(resume_path, "w") as f:
                json.dump(resume_info, f, indent=4)
            print(f"📝 Updated resume info at {resume_path}")

        # Stop early if patience exceeded
        if patience_counter >= 10:
            print(f"⛔ Early stopping triggered at epoch {epoch} for task {task_id}.")
            break

    return model
