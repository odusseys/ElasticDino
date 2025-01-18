import torchvision.transforms.functional
from datasets import  VerificationMode, load_dataset
from torch.utils.data import DataLoader
import torchvision
import os
import torch


NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 1))

imagenet_data_files = [
    f"imagenet22k-train-{i:04}.tar" for i in range(50)
]
imagenet = load_dataset("timm/imagenet-22k-wds",
                        split="train",
                        data_files=imagenet_data_files,
                        verification_mode=VerificationMode.NO_CHECKS,
                        num_proc=NUM_WORKERS)



def process(sample):
    img = sample["jpg"].convert("RGB")
    l = min(img.width, img.height)
    img = img.crop((0, 0, l, l))
    img = img.resize((256, 256))
    return torchvision.transforms.functional.pil_to_tensor(img) / 255.0

def collate_fn(samples):
   return torch.stack([process(s) for s in samples])

def load_imagenet(batch_size):
  return DataLoader(imagenet, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
  