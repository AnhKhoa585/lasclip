# LAS-CLIP: A Lightweight Adapter Steering Approach for CLIP's Visual Encoder

---

## Overview

LAS-CLIP equips CLIP with region-level visual understanding while keeping all pre-trained backbone parameters frozen. A compact MaskAdapter generates per-head attention biases conditioned on both the input mask and visual tokens. These biases steer the self-attention layers toward the region of interest without altering pre-trained weights. When no mask is supplied, the framework seamlessly reverts to standard CLIP behavior, preserving foundational zero-shot capabilities.

## Key Highlights

- Requires only 116K to 145K trainable parameters trained on 100K samples.
- Matches or outperforms Alpha-CLIP on ImageNet-S classification and RefCOCO referring expression comprehension benchmarks.
- Keeps the entire vision encoder intact to prevent representational drift.
- Retains original image-level representation fidelity when operating without spatial masks.

## Installation

Clone the repository and install dependencies.

```bash
git clone https://github.com/AnhKhoa585/lasclip.git
cd lasclip
pip install -r requirements.txt
```

## Checkpoints

Pretrained adapter checkpoints are available in the `checkpoints` directory.

| Backbone | Adapter Layers | Checkpoint File |
| --- | --- | --- |
| ViT-B/16 | 3 | `checkpoints/lasclip_b16_3_layers.pt` |
| ViT-L/14 | 3 | `checkpoints/lasclip_l14_3_layers.pt` |

## Usage

Load the model with a pretrained checkpoint and extract region-guided image representations.

```python
import torch
from PIL import Image
from LAS_CLIP import LASCLIP

model = LASCLIP(
    clip_model="ViT-B/16",
    pretrained="checkpoints/lasclip_b16_3_layers.pt",
    device="cuda"
)

image = Image.open("example.jpg")
image_tensor = model.preprocess(image).to("cuda")

mask = torch.ones(1, 224, 224, device="cuda")

feature, attention = model.encode_image_single(image_tensor, mask)
```

## Citation

```bibtex
@inproceedings{dinh2026lasclip,
  title={LAS-CLIP: A Lightweight Adapter Steering Approach for CLIP's Visual Encoder},
  author={Dinh, Duc Anh Khoa and Dinh, Duc-Tai and Nguyen, Tam V. and Tran, Minh-Triet},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2026}
}
```