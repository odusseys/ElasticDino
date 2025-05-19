import torch
import numpy as np
import torchvision
from PIL import Image
import os

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

def get_diode_dataloader(base_path, batch_size, image_size, shuffle):
    dataset = DiodeDataset(base_path, image_size)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16)
