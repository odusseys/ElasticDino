from datasets import  VerificationMode, load_dataset
from torch.utils.data import DataLoader


imagenet_data_files = [
    f"imagenet22k-train-{i:04}.tar" for i in range(50)
]
imagenet = load_dataset("timm/imagenet-22k-wds",
                        split="train",
                        data_files=imagenet_data_files,
                        verification_mode=VerificationMode.NO_CHECKS,
                        num_proc=16)


def collate_function(x):
  def process(img):
    img = img.convert("RGB")
    l = min(img.width, img.height)
    img = img.crop((0, 0, l, l))
    img = img.resize((256, 256))
    return img
  x = [process(t["jpg"]) for t in x]
  return dict(images=x)

def load_imagenet(batch_size):
  dataloader = DataLoader(imagenet, batch_size=batch_size, collate_fn=collate_function, shuffle=True)
  for x in dataloader:
    yield x