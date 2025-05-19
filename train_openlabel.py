import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from IPython.display import display
import torchvision.transforms.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs

import sys
from elasticdino.model.elasticdino import ElasticDino

def get_edino():
    edino = ElasticDino.from_pretrained("path/to/dino", "elasticdino-32-L", dino_repo="?")
    edino.eval()
    edino.requires_grad_ = False
    return edino

edino = get_edino()
# edino = torch.compile(get_edino().cuda())

from transformers import SiglipProcessor, AutoModel


siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224")
siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

class SiglipTextWrapper(nn.Module):
    def __init__(self, siglip):
        super().__init__()
        self.siglip = siglip
    
    def forward(self, x):
        return self.siglip.get_text_features(x)

siglip = SiglipTextWrapper(siglip)

sys.path.append("path/to/PhraseCutDataset")
from utils.refvg_loader import RefVGLoader
img_fpath = "path/to/PhraseCutDataset/data/VGPhraseCut_v0/images"
refvg_loader = RefVGLoader(split='train')

from concurrent.futures import ThreadPoolExecutor

import os

def check_file(i):
    p = os.path.join(img_fpath, f'{i}.jpg')
    if os.path.isfile(p):
        return i
    else:
        return None

with ThreadPoolExecutor(64) as executor:
    results = list(executor.map(check_file, refvg_loader.img_ids))

valid_ids = [i for i in results if i is not None]
import skimage
import random

IMAGE_SIZE = 128

def get_biggest_polygon(polygons, h, w):
    l = min(h, w)
    biggest = -1
    index = 0
    selected = None
    for i in range(len(polygons)):
        mask = None
        for plist in polygons[i]:
            for p in plist:
                p = [[x, y] for y, x in p]
                m = skimage.draw.polygon2mask([h, w],p)[:l, :l]
                if mask is None:
                    mask = m
                else:
                    mask = mask | m
        area = np.sum(mask)

        choose = True if selected is None else random.random() < 0.5
        if area > biggest and choose:
            biggest = area
            index = i
            selected = mask
    return selected, index
            
            
def get_random_polygon(polygons, h, w):
    l = min(h, w)
    i = random.randint(0, len(polygons) - 1)
    mask = None
    for plist in polygons[i]:
        for p in plist:
            p = [[x, y] for y, x in p]
            m = skimage.draw.polygon2mask([h, w],p)[:l, :l]
            if mask is None:
                mask = m
            else:
                mask = mask | m
    return mask, i
    
def to_samples(image_id):
    img = Image.open(os.path.join(img_fpath, '%d.jpg' % image_id)).convert("RGB")
    l = min(img.height, img.width)
    metadata = refvg_loader.get_img_ref_data(image_id)
    image = torchvision.transforms.functional.pil_to_tensor(img.crop((0, 0, l, l))).unsqueeze(0)/255.0
    
    
    mask, i = get_random_polygon(metadata["gt_Polygons"], img.height, img.width)
    phrase = metadata["phrases"][i]
    
    mask = mask[:l, :l]
    mask = torch.tensor(mask)
    image = torch.nn.functional.interpolate(image, IMAGE_SIZE)
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).to(dtype=torch.float16), IMAGE_SIZE, mode="nearest").to(dtype=torch.long)
    return image.squeeze(), mask.squeeze(), phrase

class PhraseDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.img_ids = valid_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        return to_samples(self.img_ids[idx])


class ResidualFC(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.BatchNorm1d(n_features),
            nn.GELU(),
            nn.Linear(n_features, n_features),
            nn.BatchNorm1d(n_features),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.layers(x)

class FC(nn.Module):
    def __init__(self, fin, out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(fin, out),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualBlockConv(nn.Module):
    def __init__(self, n_features=1024): 
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(n_features,n_features,1, bias=False),
            nn.BatchNorm2d(n_features),
            nn.GELU(),
            nn.Conv2d(n_features,n_features,1, bias=False),
            nn.BatchNorm2d(n_features),
            nn.GELU(),
        )
    def forward(self, x):
        return x + self.layers(x)

class EncoderModel(nn.Module):
    def __init__(self, n_features=2048, temperature = 0.2):
        super().__init__()

        self.temperature = temperature

        self.text_encoder = nn.Sequential(
            nn.Linear(768, n_features),
            nn.BatchNorm1d(n_features),
            ResidualFC(n_features),
            ResidualFC(n_features),
            ResidualFC(n_features),
            ResidualFC(n_features),
            ResidualFC(n_features),
            nn.Linear(n_features, 1024),
        )

    def forward(self, image_features, text_features):
        text_features = torch.nn.functional.normalize(self.text_encoder(text_features), dim=1)
        return torch.einsum("bchw,bc->bhw", image_features, text_features) / self.temperature
        
@torch.no_grad
def make_saliency_map_flat(model, image, text, mask=None):
    C, H, W = image.shape
    features = edino(image.unsqueeze(0))
    objects_features = siglip(siglip_processor(text=[text])["input_ids"].cuda())
    res = torch.sigmoid(model(features, objects_features))
    salience = F.to_pil_image(res[0]).resize((128, 128)).convert("RGB")
    image = F.to_pil_image(image).resize((128, 128))
    x = [image, salience]
    if mask is not None:
        x.append(F.to_pil_image(mask.to(dtype=torch.float32)).resize((128, 128)).convert("RGB"))
    return Image.fromarray(np.hstack(x))
    

def debug_step():
    return

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, p=2, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Flatten tensors
        inputs = torch.sigmoid(inputs.contiguous().view(inputs.shape[0], -1))
        targets = targets.contiguous().view(targets.shape[0], -1)

        # Numerator and denominator of Dice coefficient
        num = (2 * (inputs * targets).sum(dim=1)) + self.smooth
        den = (inputs.pow(self.p).sum(dim=1) + targets.pow(self.p).sum(dim=1)) + self.smooth

        dice_score = num / den
        loss = 1 - dice_score

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
        


batch_size = 16
debug_interval = 200   
save_interval = 500     
n = 0
model = EncoderModel()

from datetime import datetime

project_folder = f"elasticdino-runs/openlabel"
current_datetime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
run_folder = os.path.join(project_folder, current_datetime)
os.makedirs(run_folder, exist_ok=True)
os.makedirs(f"{run_folder}/images", exist_ok=True)
os.makedirs(f"{run_folder}/checkpoints", exist_ok=True)

import bitsandbytes
optimizer = bitsandbytes.optim.AdamW8bit(
    [{"params": model.parameters(), "lr": 1e-5}], eps=1e-5, weight_decay=1e-4)

running_loss = None


kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
dynamo_backend = "inductor"
accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[kwargs], dynamo_backend=dynamo_backend)

dataloader = torch.utils.data.DataLoader(PhraseDataset(), batch_size=batch_size, shuffle=True)

checkpoint = None

edino, siglip, model, optimizer, dataloader = accelerator.prepare(edino, siglip, model, optimizer, dataloader)
if checkpoint is not None:
    accelerator.load_state(checkpoint)
for epoch in range(20):
    if accelerator.is_local_main_process:
        print("Epoch", epoch + 1)
    for images, mask, phrases in dataloader:
        n+=1
        with torch.no_grad():
            embeds = siglip(siglip_processor(text=phrases, padding=True,)["input_ids"].cuda()).squeeze()
        with accelerator.autocast():
            with torch.no_grad():
                features = edino(images)
            predicted = model(features, embeds)
            loss = BinaryDiceLoss()(predicted.reshape((-1, 1)), mask.to(dtype=torch.float32).reshape((-1, 1))) + nn.BCEWithLogitsLoss()(predicted.reshape((-1, 1)), mask.to(dtype=torch.float32).reshape((-1, 1)))
        
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if running_loss is None:
            running_loss = loss.detach()
        else:
            running_loss = 0.98 * running_loss + 0.02 * loss.detach()  
        if n == 1:
            print("First iteration done")
        if n % debug_interval == 0 and accelerator.is_local_main_process:
            model.eval()
            line = f"{n} {running_loss.item()}"
            print(line)
            with open(f"{run_folder}/training_loss.txt", "a+") as f:
                f.write(line + "\n")
            dbg = [
            make_saliency_map_flat(model, F.pil_to_tensor(Image.open("dog.jpeg").convert("RGB").resize((128, 128))).cuda() / 255.0, "grass"),
            make_saliency_map_flat(model, F.pil_to_tensor(Image.open("dog.jpeg").convert("RGB").resize((128, 128))).cuda() / 255.0, "dog"),
            make_saliency_map_flat(model, F.pil_to_tensor(Image.open("person.jpeg").convert("RGB").resize((128, 128))).cuda() / 255.0, "hair"),
            make_saliency_map_flat(model, F.pil_to_tensor(Image.open("person.jpeg").convert("RGB").resize((128, 128))).cuda() / 255.0, "shirt"),
            make_saliency_map_flat(model, F.pil_to_tensor(Image.open("car.jpeg").convert("RGB").resize((128, 128))).cuda() / 255.0, "window"),
            make_saliency_map_flat(model, F.pil_to_tensor(Image.open("car.jpeg").convert("RGB").resize((128, 128))).cuda() / 255.0, "car")
        ]

            Image.fromarray(np.vstack(dbg)).save(f"{run_folder}/images/{n}.png")

            model.train()
        
        if n % save_interval == 0:
          accelerator.save_state( f"{run_folder}/checkpoints/{n}")
        
        del images
        del features
        del mask
        del embeds
        del predicted
        del loss
        
