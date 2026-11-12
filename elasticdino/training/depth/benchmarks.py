import numpy as np
import torchvision
import torch
import os
from PIL import Image

class NyuDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, image_size):
        self.paths = os.listdir(os.path.join(base_path, "rgb"))
        self.base_path = base_path
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        rgb = np.load(os.path.join(self.base_path, "rgb", path))[:, :480, :480] / 255.0
        depth = np.load(os.path.join(self.base_path, "depth", path))[:480, :480]
        rgb = torch.nn.functional.interpolate(torch.from_numpy(rgb).unsqueeze(0), self.image_size).squeeze(0)
        depth = torch.nn.functional.interpolate(torch.from_numpy(depth).unsqueeze(0).unsqueeze(0), self.image_size).squeeze(0)
        masks = torch.ones_like(depth).to(dtype=torch.bool)
        return rgb, depth, masks

def get_nyu_dataloader(base_path, batch_size, image_size):
    dataset = NyuDataset(base_path, image_size)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)


class DiodeDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, image_size):
        self.truncated_paths = []
        self.image_size = image_size
        for root, dirs, files in os.walk(base_path):
            for file in files:
              if "png" in file.lower():
                image_id = file.split(".")[0]
                self.truncated_paths.append(os.path.join(root, image_id))
                
    def __len__(self):
        return len(self.truncated_paths)

    def __getitem__(self, idx):
        truncated_path = self.truncated_paths[idx]

        rgb = Image.open(f"{truncated_path}.png").crop((0, 0, 768, 768)).resize((self.image_size, self.image_size))
        rgb = torchvision.transforms.functional.pil_to_tensor(rgb) / 255.0
        
        depth = np.load(f"{truncated_path}_depth.npy")[:768, :768]
        depth_mask = np.load(f"{truncated_path}_depth_mask.npy")[:768, :768]
                
        depth = torch.nn.functional.interpolate(torch.from_numpy(depth).squeeze().unsqueeze(0).unsqueeze(0), self.image_size, mode="bilinear").squeeze(0)
        depth_mask = torch.nn.functional.interpolate(torch.from_numpy(depth_mask).squeeze().unsqueeze(0).unsqueeze(0), self.image_size, mode="nearest").squeeze(0)
        
        return rgb, depth, depth_mask > 0

def get_diode_dataloader(base_path, batch_size, image_size):
    dataset = DiodeDataset(base_path, image_size)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

