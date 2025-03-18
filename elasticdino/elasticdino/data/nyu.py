import numpy as np
import torch
import os

    

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

def get_nyu_dataloader(base_path, batch_size, image_size, shuffle):
    dataset = NyuDataset(base_path, image_size)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16)
