---
library_name: pytorch
---

# ElasticDino 3 32-S

This model uses DINOv3 ViT-S/16-plus features and produces a 256 x 256 dense
feature map.
The weights were duplicated from `articuno7/elasticdino-3-32-S`.

The ElasticDino repository is public, but the upstream
[DINOv3 backbone](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m)
is gated. Accept Meta's access terms and run `hf auth login`, or pass an
authorized `token` to `from_pretrained`.

## Usage

```bash
git clone https://github.com/odusseys/ElasticDino.git
cd ElasticDino
pip install torch torchvision huggingface_hub pillow 'transformers>=4.57,<5'
```

```python
import torch
from PIL import Image
from elasticdino.model.elasticdino import ElasticDino

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ElasticDino.from_pretrained("odusseys/elasticdino-3-32-S").to(device).eval()
image = Image.open("image.jpg").convert("RGB")
with torch.inference_mode():
    features = model(image)
print(features.shape)  # (1, 384, 256, 256)
```

An RGB tensor of shape `(3, H, W)` or `(B, 3, H, W)` is also accepted;
use `uint8` in `[0, 255]` or floating point in `[0, 1]`. Non-square inputs
are center cropped with a warning.
