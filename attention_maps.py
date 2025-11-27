import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import v2

# ================================
# 1. Load local Meta DINOv3 model
# ================================

REPO_DIR = '/users/student/pg/pg23/vaibhav.rathore/D_GCD/DG/project/dinov3'

# IMPORTANT — your checkpoint URL
link = "https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=...."

# Load model
global_model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=link)
global_model.eval()

print("Loaded DINOv3 model.")


# ===========================
# 2. Preprocess image function
# ===========================

def make_transform(size=224):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)),
    ])


# ===========================
# 3. Load image
# ===========================

img_path = "/janaki/backup/users/student/pg/pg23/vaibhav.rathore/datasets/FG-DGCD/CUB/real/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"

orig_img = Image.open(img_path).convert("RGB")
transform = make_transform(224)
img_tensor = transform(orig_img).unsqueeze(0)   # [1,3,224,224]

print("Image loaded.")


# ==========================================
# 4. Hook qkv projection to extract q and k
# ==========================================

qk_cache = {}

def qkv_hook(module, inp, out):
    qkv = out.detach()
    B, N, threeC = qkv.shape
    C = threeC // 3
    q = qkv[:, :, :C]
    k = qkv[:, :, C:2*C]
    qk_cache["q"] = q
    qk_cache["k"] = k

# Attach hook to the last block’s attention qkv layer
last_attn_block = global_model.blocks[-5].attn
hook = last_attn_block.qkv.register_forward_hook(qkv_hook)
print("Hook attached to qkv.")


# ============================
# 5. Forward pass
# ============================

with torch.no_grad():
    _ = global_model(img_tensor)

hook.remove()
print("Forward pass done. Hook removed.")


# ================================================
# 6. Manually compute attention from Q and K
# ================================================

q = qk_cache["q"]    # [1, N, C]
k = qk_cache["k"]    # [1, N, C]

num_heads = last_attn_block.num_heads
head_dim = q.shape[-1] // num_heads

# Reshape into multi-head format
q = q.reshape(1, -1, num_heads, head_dim).transpose(1, 2)  # [1, H, N, D]
k = k.reshape(1, -1, num_heads, head_dim).transpose(1, 2)  # [1, H, N, D]

# Compute attention
attn = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
attn = torch.softmax(attn, dim=-1)       # [1, H, N, N]

# Average across heads
attn = attn.mean(1)[0]                   # [N, N]

# CLS → patch attentions (exclude CLS token at index 0)
attn = attn[0, 1:]                       # [N-1]
print(f"Number of patch tokens: {attn.shape[0]}")


# =============================================
# 7. Build heatmap safely (handle non-square)
# =============================================

attn_map = attn.cpu().numpy()
num_tokens = attn_map.shape[0]

# Compute side length and pad if necessary
side_len = int(np.ceil(np.sqrt(num_tokens)))
pad_size = side_len ** 2 - num_tokens
if pad_size > 0:
    attn_map = np.pad(attn_map, (0, pad_size), mode='constant')

attn_map = attn_map.reshape(side_len, side_len)

# Normalize
attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

# Resize to match original image
heatmap_img = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(orig_img.size)


# ============================
# 8. Save attention heatmap
# ============================

plt.imshow(heatmap_img, cmap='jet')
plt.axis("off")
plt.savefig("dinov3_attention_heatmap_5.png", bbox_inches="tight")
plt.close()

# # Overlay on original image
# overlay = Image.blend(orig_img.convert("RGBA"), heatmap_img.convert("RGBA"), alpha=0.45)
# overlay.save("dinov3_attention_overlay_2.png")

print("Saved: dinov3_attention_heatmap_5.png")
# print("Saved: dinov3_attention_overlay_2.png")
