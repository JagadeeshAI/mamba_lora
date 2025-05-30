# ==================== compute_stats.py ====================

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

import os

from src.model.arch2 import VisionMamba

def print_lora_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "lora_" in n and p.requires_grad
    )

    trainable_pct = (trainable_params / total_params) * 100 if total_params else 0
    lora_pct = (lora_params / total_params) * 100 if total_params else 0

    print(f"\n📊 Model Parameters Summary:")
    print(f"🔹 Total Parameters:     {total_params:,}")
    print(f"🔹 Trainable Parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"🔹 LoRA Parameters Only: {lora_params:,} ({lora_pct:.2f}%)")


def load_pretrained_model():
    local_ckpt = os.path.join("/home/jag/codes/U/src/43.pth")
    if not os.path.exists(local_ckpt):
        raise FileNotFoundError(f"❌ Pretrained model not found at {local_ckpt}")

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
        lora_out_proj=True,
        lora_r=96,
        lora_alpha=0.1,
    )


    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    print_lora_stats(model)
    return model


if __name__ == "__main__":
    load_pretrained_model()
    # run_all_tasks()
