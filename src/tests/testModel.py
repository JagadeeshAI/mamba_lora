import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch

from src.model.arch2 import VisionMamba

def print_lora_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)

    print(f"\n📊 Model Parameters Summary:")
    print(f"🔹 Total Parameters:     {total_params:,}")
    print(f"🔹 Trainable Parameters: {trainable_params:,}")
    print(f"🔹 LoRA Parameters Only: {lora_params:,}")

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
    lora_alpha=0.1
).to("cuda")

for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

print_lora_stats(model)



dummy_input = torch.randn(1, 3, 224, 224).to("cuda")

output = model(dummy_input)

# Print output shape and predicted class
print("Output shape:", output.shape)    
print("Predicted class:", torch.argmax(output, dim=1).item())