import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import zfpy
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

import subprocess
from skimage.color import rgb2lab, lab2rgb

import utils
from train import Config
from model import Auto_Encoder

cfg = Config()

def preprocess_image(image_path, patch_size):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    new_w = (w // patch_size) * patch_size; new_h = (h // patch_size) * patch_size
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    patches = [utils.rgb_to_lab(img_resized.crop((j, i, j + patch_size, i + patch_size))).unsqueeze(0) for i in range(0, new_h, patch_size) for j in range(0, new_w, patch_size)]
    return torch.cat(patches, 0), (new_h, new_w)

def merge_patches_into_image(patches_rgb, image_shape):
    
    # (N, C, H, W) to (N, H, W, C)
    if patches_rgb.dim() == 4:
        patches_rgb = patches_rgb.permute(0, 2, 3, 1)
    
    h, w = image_shape
    patch_size = patches_rgb.shape[1]
    
    full_image = np.zeros((h, w, 3), dtype=np.uint8)
    patch_idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = (patches_rgb[patch_idx] * 255).clip(0, 255).cpu().detach().numpy().astype(np.uint8)
            full_image[i:i+patch_size, j:j+patch_size, :] = patch
            patch_idx += 1
    return full_image

image_path = "test/easy.jpg"
tolerance = 1e-3 # basically works for any image, zfp will give no gains tho

# actual inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Auto_Encoder(cfg.patch_size, cfg.latent_size)
model.load_state_dict(torch.load(cfg.save_path, map_location=device))
model.to(device)
model.eval()

patches_tensor, original_shape = preprocess_image(image_path, cfg.patch_size)
patches_tensor = patches_tensor.to(device)

# the research this repo is based off of was about compression, edit this to be whatever

with torch.no_grad():
    latents, out = model.forward(patches_tensor)

latent = latents.cpu().numpy().astype(np.float32)
compressed_data = zfpy.compress_numpy(latent, tolerance=tolerance)
zfp_path = "compressed_latent.zfp"

# lossy zfp stage
with open(zfp_path, "wb") as f: 
    f.write(compressed_data)

zip_p = "C:/Program Files/7-Zip/7z.exe"

# lossless 7z stage
seven_zip_path = "compressed_latents.zfp.7z"
subprocess.run([zip_p, 'a', seven_zip_path, zfp_path, '-mx=9'])

print("\nmetrics:\n")
original_size_kb = os.path.getsize(image_path) / 1024
uncompressed_latent_size_kb = latent.nbytes / 1024
zfp_size_kb = os.path.getsize(zfp_path) / 1024
seven_zip_size_kb = os.path.getsize(seven_zip_path) / 1024
final_ratio = original_size_kb / seven_zip_size_kb

print(f"Original Image Size: ......... {original_size_kb:8.2f} KB")
print(f"Uncompressed Latent (Indices): {uncompressed_latent_size_kb:8.2f} KB")
print(f"Size after ZFP (lossy): ...... {zfp_size_kb:8.2f} KB")
print(f"Size after 7-Zip (lossless): . {seven_zip_size_kb:8.2f} KB")
print(f"Final compression Ratio: {final_ratio:.2f}x")

# verifies image visually be decompressing from compressed path
with open(zfp_path, "rb") as f: 
    loaded = f.read()

decompressed = zfpy.decompress_numpy(loaded)
latents = torch.from_numpy(decompressed).to(torch.float32).to(device)

with torch.no_grad():
    reconstructed_patches_lab = model.dec(latents)

reconstructed_patches_rgb = utils.lab_to_rgb(reconstructed_patches_lab)
reconstructed_image = merge_patches_into_image(reconstructed_patches_rgb, original_shape)

original_image = Image.open(image_path)
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

axs[0].imshow(original_image)
axs[0].set_title(f"Original Image ({original_size_kb:.2f} KB)")
axs[0].axis("off")

axs[1].imshow(reconstructed_image)
axs[1].set_title(f"Reconstructed Image ({seven_zip_size_kb:.2f} KB | {final_ratio:.2f}x Compression)"); axs[1].axis("off")

plt.tight_layout()
plt.show()
