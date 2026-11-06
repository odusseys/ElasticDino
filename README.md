# ElasticDino

## Install

Create and activate a Python virtual environment, then install the dependencies:

```bash
pip install torch torchvision huggingface_hub pillow
```

The DINOv2 backbone is downloaded through PyTorch Hub on first use.

## Usage

| Repository | Base model | Model size |
| --- | --- | --- |
| [elasticdino-3-32-S](https://huggingface.co/odusseys/elasticdino-3-32-S) | DINOv2 | Small |
| [elasticdino-32-S](https://huggingface.co/odusseys/elasticdino-32-S) | DINOv2 | Small |

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

Inputs can also be RGB PyTorch tensors of shape `(3, H, W)` or
`(B, 3, H, W)`: use `uint8` values in `[0, 255]` or floating-point values
in `[0, 1]`. Inputs are moved to the model's device. Non-square images are
center cropped with a warning. The output always retains a batch dimension.
