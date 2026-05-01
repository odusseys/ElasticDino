# ElasticDino

## Install

We recommend you create a virtual environment first: `python -m venv my-venv-name` and `source  my-venv-name/bin/activate`

```
pip install torch torchvision kornia
```

## Usage

```

from elasticdino.model.elasticdino import ElasticDino

model = ElasticDino.from_pretrained("path/to/checkpoint")

```
