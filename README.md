# ElasticDino

## Install

We recommend you create a virtual environment first: `python -m venv my-venv-name` and `source  my-venv-name/bin/activate`

```
pip install torch torchvision kornia

from elasticdino.model.elasticdino import ElasticDino

model = ElasticDino.from_pretrained("path/to/checkpoint")

```

## TODO

- [] Release semantic segmentation and open-vocabulary checkpoints
- [] Fix and document training and eval code

