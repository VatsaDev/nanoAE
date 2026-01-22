import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb
# from torchmetrics.image import StructuralSimilarityIndexMeasure

from torchvision import transforms
from torchvision.models import vgg16
from torch.utils.data import Dataset, DataLoader

import utils
from model import Auto_Encoder

class Config:
    
    patch_size = 64
    latent_size = 16
    batch_size = 20
    learning_rate = 1e-3
    data_dir = "data/fin_min"
    epochs = 100
    save_path = "checkpoints/AE.pth"

cfg = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading 
# just drop all your images inside a folder in data

class ImagePatchDataset(Dataset):
    def __init__(self, img_dir, patch_size):

        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.resize = transforms.Resize((patch_size, patch_size))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image_rgb = Image.open(img_path).convert('RGB')
        image_rgb = self.resize(image_rgb)

        # Convert to Lab for training, keep RGB for perceptual loss
        image_lab_tensor = utils.rgb_to_lab(image_rgb)
        image_rgb_tensor = transforms.ToTensor()(image_rgb)

        return image_lab_tensor, image_rgb_tensor

dataset = ImagePatchDataset(cfg.data_dir, cfg.patch_size)
train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

# setups 

model = Auto_Encoder(cfg.patch_size, cfg.latent_size)

recon_loss_fn = nn.L1Loss()
# ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device) # replace package

optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)

def train():

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("total params:", total_params)
    print(f"\ntraining for {cfg.epochs} epochs")

    for epoch in range(cfg.epochs):

        model.train()
    
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}")

        for lab_images, rgb_images in progress_bar:
            lab_images = lab_images.to(device)
            rgb_images = rgb_images.to(device)

            optimizer.zero_grad()

            _, output = model(lab_images) # latent/output

            # losses        
        
            recon_loss = recon_loss_fn(output, lab_images)
            #ssim_loss = 1 - ssim(output[:, :1, :, :], lab_images[:, :1, :, :])
            #loss = (0.5 * recon_loss + 0.5 * ssim_loss)
        
            loss = recon_loss
    
            loss.backward()
        
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({
                'Total': f"{loss.item():.4f}",
                'Recon': f"{recon_loss.item():.4f}",
                #'SSIM': f"{ssim_loss.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Loss: {avg_loss:.6f}")

        torch.save(model.state_dict(), cfg.save_path) # epoch ckpt save

if __name__ == '__main__':
    train()
