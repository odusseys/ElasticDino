# ElasticDino

## Install

Create and activate a Python virtual environment, then install the dependencies:

```bash
pip install torch torchvision huggingface_hub pillow 'transformers>=4.57,<5'
```

The DINOv2 backbone is downloaded through PyTorch Hub on first use. The
DINOv3 model also requires access to Meta's gated
[DINOv3 ViT-S/16-plus weights](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m).
Accept the upstream access terms and authenticate with `hf auth login` before
loading that model, or pass a Hugging Face token to `from_pretrained`.

## Usage

| Repository | Base model | Model size |
| --- | --- | --- |
| [elasticdino-3-32-S](https://huggingface.co/odusseys/elasticdino-3-32-S) | DINOv3 | Small |
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

## Experiments

Research entry points and notebooks are in `elasticdino/experiments`. Run an
entry point from the repository root as a module, for example:

```bash
python -m elasticdino.experiments.train_evaluate_ablations --dataset-root datasets/hypersim
```

Some older experiments and notebooks use placeholder paths and the previous
checkpoint-loading API; they require adaptation before running.
