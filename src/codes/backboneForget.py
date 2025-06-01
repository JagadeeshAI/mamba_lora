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

def backbone_forgetting(model, device, loaders, task_id, config, start_epoch=0, resume_path=None):
    model.to(device).train()
    print(f"\n🚀 Starting Forgetting on Task {task_id} (from epoch {start_epoch})...")
    os.makedirs(config.FORGET.OUT_DIR, exist_ok=True)

    loader_r = loaders["train_r"]
    loader_f = loaders["train_f"]

    optimizer = AdamW(model.parameters(), lr=config.FORGET.LR, weight_decay=config.FORGET.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.FORGET.EPOCHS)

    forget_cycle = cycle(loader_f)

    for epoch in range(start_epoch + 1, config.FORGET.EPOCHS + 1):
        model.train()
        total_loss_r = total_loss_f = 0
        steps = 0
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

            total_loss_r += loss_r.item()
            total_loss_f += loss_f.item()
            steps += 1

            loop.set_postfix({
                "Ret": f"{loss_r:.4f}",
                "For": f"{loss_f:.4f}"
            })

        scheduler.step()

        val_acc_r = compute_accuracy(model, loaders["val_r"], device)
        val_acc_f = compute_accuracy(model, loaders["val_f"], device)

        retain_ok = "✅" if val_acc_r >= 95 else "❌"
        forget_ok = "✅" if val_acc_f <= 5 else "❌"
        print(f"🧪 [Epoch {epoch:3d}] Retain Acc: {val_acc_r:6.2f}% {retain_ok} | Forget Acc: {val_acc_f:6.2f}% {forget_ok}")

        # 🔐 Save only the latest epoch checkpoint for this task
        ckpt_name = f"task{task_id}epoch{epoch}.pth"
        ckpt_path = os.path.join(config.FORGET.OUT_DIR, ckpt_name)

        # Delete previous checkpoint if exists
        prev_ckpt = f"task{task_id}epoch{epoch - 1}.pth"
        prev_ckpt_path = os.path.join(config.FORGET.OUT_DIR, prev_ckpt)
        if os.path.exists(prev_ckpt_path):
            os.remove(prev_ckpt_path)

        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

        # 💾 Update compute_resume.json after each epoch
        if resume_path:
            resume_info = {
                "task": task_id,
                "epoch": epoch,
                "model_path": ckpt_path
            }
            with open(resume_path, "w") as f:
                json.dump(resume_info, f, indent=4)
            print(f"📝 Updated resume info at {resume_path}")

    return model
