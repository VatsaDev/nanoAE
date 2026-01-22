import torch
import numpy as np
from skimage.color import rgb2lab, lab2rgb

def rgb_to_lab(img):
    
    """PIL RGB image to a normalized Lab tensor"""
    
    lab = rgb2lab(np.array(img))
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    l = l / 50.0 - 1.0 # Normalize L channel from [0, 100] to [-1, 1]
    a = a / 128.0      # norm channels from [-127, 127] to [-1, 1]
    b = b / 128.0

    lab_normalized = np.stack([l, a, b], axis=2)
    
    return torch.from_numpy(lab_normalized).float().permute(2, 0, 1)


def lab_to_rgb(lab_tensor):
    lab_tensor = lab_tensor.detach().cpu().permute(0, 2, 3, 1)
    lab = lab_tensor.numpy()

    # explicit clipping to valid LAB ranges
    l_chan = np.clip((lab[..., 0] + 1.0) * 50.0, 0, 100)
    a_chan = np.clip(lab[..., 1] * 128.0, -128, 127)
    b_chan = np.clip(lab[..., 2] * 128.0, -128, 127)
    
    lab_clipped = np.stack([l_chan, a_chan, b_chan], axis=-1)

    rgb_images = []
    for i in range(lab_clipped.shape[0]):
        # skimage lab2rgb expects float64 in these specific ranges
        rgb = lab2rgb(lab_clipped[i].astype(np.float64))
        rgb_images.append(torch.from_numpy(rgb).float().permute(2, 0, 1))

    return torch.stack(rgb_images)
