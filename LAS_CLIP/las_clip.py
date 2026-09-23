import os
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
try:
    import open_clip
except ImportError:
    open_clip = None

# Use local modules within the package
from . import clip  # Local modified clip
from .interpreter import Box

from .adapter import ContextAdapter, save_adapter, load_adapter

# Supported model configs: {model_name: (input_size, patch_size, vision_width, vision_layers, vision_head_width)}
# vision_head_width: width per attention head (64 for ViT-L, 80 for ViT-H)
MODEL_DEFAULTS = {
    # ViT-L/14 variants (OpenAI style)
    "ViT-L/14@336px": (336, 14, 1024, 24, 64),  # grid 24x24, 577 tokens
    "ViT-L/14":       (224, 14, 1024, 24, 64),  # grid 16x16, 257 tokens
    # ViT-B variants
    "ViT-B/16":       (224, 16, 768, 12, 64),
    "ViT-B/32":       (224, 32, 768, 12, 64),
}

class LASCLIP(nn.Module):
    """LASCLIP executor using CLIP ViT-L/14 and ViT-H/14 variants.
    
    Supports local OpenAI CLIP models and OpenCLIP/HuggingFace models.
    Features batch-parallel encoding and vectorized mask generation.
    Supports Trainable SegFALIP integration via `ContextAdapter`.
    """

    def __init__(self, clip_model: str = "ViT-L/14@336px", pretrained: str = None, 
                 device: str = "cuda", 
                 batch_size: int = 32, mask_layers: Optional[int] = 3,
                 mode: str = "eval", hidden_dim: int = 64):
        """Initialize LASCLIP.
        
        Args:
            clip_model: Model architecture name, e.g. "ViT-L/14", "ViT-L/14@336px", "ViT-H/14".
            pretrained: Pretrained weights source:
                - None: Load local OpenAI CLIP weights.
                - "hf-hub:org/model": Load from HuggingFace Hub (e.g. "hf-hub:apple/DFN2B-CLIP-ViT-L-14").
                - OpenCLIP tag: Load from OpenCLIP registry (e.g. "laion2b_s32b_b79k").
                - Local checkpoint path for SegFALIP eval.
            device: Torch device.
            batch_size: Batch size for encoding.
            mask_layers: Number of last transformer layers to inject the mask.
            mode: "train" or "eval".
        """
        super().__init__()
        self.device = device
        self.device_tensor = torch.device(device)
        self.clip_model_name = clip_model
        self.encode_batch_size = batch_size
        self.mask_layers = mask_layers
        self.mask_net = None
        self.mode = mode
        self.hidden_dim = hidden_dim

        # Determine if we use local CLIP loading or OpenCLIP
        use_open_clip = pretrained is not None and ('hf-hub:' in pretrained or '/' not in pretrained) and not pretrained.endswith('.pt')
        loading_checkpoint = pretrained is not None and pretrained.endswith('.pt')

        if use_open_clip:
            if open_clip is None:
                raise ImportError("open_clip is required to load custom weights. Install with `pip install open_clip_torch`.")
            self._init_open_clip(clip_model, pretrained)
        else:
            self._init_local_clip(clip_model)

        # Keep mask_layers as instantiated (default 3)
        if self.mode == "train":
            if hasattr(self, 'model') and hasattr(self.model, 'visual') and hasattr(self.model.visual, 'transformer'):
                self.model.visual.transformer.mask_layers = self.mask_layers

        if mode == "train":
            d_model, num_heads = self._get_vision_config()
            self.mask_net = ContextAdapter(input_size=self.input_size, patch_size=self.patch_size, mask_layers=self.mask_layers, d_model=d_model, num_heads=num_heads, hidden_dim=self.hidden_dim)
            self.mask_net.to(self.device_tensor)
            
            for param in self.model.parameters():
                param.requires_grad = False
                
            for param in self.mask_net.parameters():
                param.requires_grad = True
                
        elif mode == "eval":
            if loading_checkpoint:
                sd = torch.load(pretrained, map_location="cpu")
                if "state_dict" in sd:
                    sd = sd["state_dict"]
                has_mask_net = any(k.startswith("mask_net.") for k in sd.keys())
                
                if has_mask_net:
                    ckpt_mask_layers = self.mask_layers
                    if "mask_net.layer_embed.weight" in sd:
                        ckpt_mask_layers = sd["mask_net.layer_embed.weight"].shape[0]

                    d_model, num_heads = self._get_vision_config()
                    self.mask_net = ContextAdapter(input_size=self.input_size, patch_size=self.patch_size, mask_layers=ckpt_mask_layers, d_model=d_model, num_heads=num_heads, hidden_dim=self.hidden_dim)
                    self.mask_net.to(self.device_tensor)
                    
                    if hasattr(self, 'model') and hasattr(self.model, 'visual') and hasattr(self.model.visual, 'transformer'):
                         self.model.visual.transformer.mask_layers = ckpt_mask_layers
                    self.load_state_dict(sd, strict=False)
                else:
                    raise ValueError("Provided checkpoint does not contain SegFALIP trained weights (`mask_net`). Only SegFALIP checkpoints are permitted for eval.")
            else:
                d_model, num_heads = self._get_vision_config()
                self.mask_net = ContextAdapter(input_size=self.input_size, patch_size=self.patch_size, mask_layers=self.mask_layers, d_model=d_model, num_heads=num_heads, hidden_dim=self.hidden_dim)
                self.mask_net.to(self.device_tensor)
                
                if hasattr(self, 'model') and hasattr(self.model, 'visual') and hasattr(self.model.visual, 'transformer'):
                    self.model.visual.transformer.mask_layers = self.mask_layers
                
        self.model.to(self.device)
        self.model.eval()

    def _get_vision_config(self):
        """Extract d_model and num_heads from the loaded visual backbone."""
        if hasattr(self.model.visual, "transformer"):
            d_model = self.model.visual.transformer.width
            num_heads = self.model.visual.transformer.resblocks[0].attn.num_heads
        elif hasattr(self.model.visual, "attnpool"):
            d_model = self.model.visual.attnpool.c_proj.in_features
            num_heads = self.model.visual.attnpool.num_heads
        else:
            d_model = 768
            num_heads = 12
        return d_model, num_heads

    def _init_local_clip(self, model_name):
        if model_name.startswith("ViT-H") or model_name.startswith("ViT-h") or model_name.startswith("ViT-G") or model_name.startswith("ViT-g"):
            raise ValueError(f"{model_name} models require pretrained weights. Use OpenCLIP by specifying 'pretrained' parameter.")

        if model_name not in MODEL_DEFAULTS:
             raise ValueError(f"Unsupported local model '{model_name}'. "
                             f"Choose from: {list(MODEL_DEFAULTS.keys())} or specify 'pretrained' for OpenCLIP.")
        
        self.input_size, self.patch_size, self.vision_width, self.vision_layers, self.vision_head_width = MODEL_DEFAULTS[model_name]
        
        if getattr(self, "mask_layers", None) is None:
            self.mask_layers = 3
            
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False, mask_layers=self.mask_layers)
        self.preprocess.transforms[0] = T.Resize(
            (self.input_size, self.input_size),
            interpolation=T.InterpolationMode.BICUBIC
        )

    @staticmethod
    def _convert_model_name_to_open_clip(model_name: str) -> str:
        """Convert model name from OpenAI format to OpenCLIP format.
        
        Examples:
            'ViT-L/14' -> 'ViT-L-14'
            'ViT-L/14@336px' -> 'ViT-L-14-336'
            'ViT-H/14@378px' -> 'ViT-H-14-378'
        """
        # Replace '/' with '-'
        oc_name = model_name.replace('/', '-')
        # Handle resolution suffix: @336px -> -336
        if '@' in oc_name:
            oc_name = oc_name.replace('@', '-').replace('px', '')
        return oc_name

    def _init_open_clip(self, model_name, pretrained):
        """Initialize model using OpenCLIP.
        
        Args:
            model_name: Architecture name (e.g. "ViT-L/14", "ViT-H/14@378px")
            pretrained: Either "hf-hub:org/model" or OpenCLIP pretrained tag
        """
        # Convert model name to OpenCLIP format
        oc_model_name = self._convert_model_name_to_open_clip(model_name)
        vision_head_width = None
        
        # Infer vision_head_width from model name
        if 'ViT-H' in model_name or 'ViT-h' in model_name:
            vision_head_width = 80
        elif 'ViT-g' in model_name or 'ViT-G' in model_name:
            vision_head_width = 104
        else:
            vision_head_width = 64  # Default for ViT-L, ViT-B
            
        print(f"Loading weights via OpenCLIP: model={oc_model_name}, pretrained={pretrained}")
        
        # For HuggingFace Hub models, use create_model_from_pretrained
        if pretrained.startswith("hf-hub:"):
            oc_model, oc_preprocess = open_clip.create_model_from_pretrained(pretrained)
        else:
            oc_model, _, oc_preprocess = open_clip.create_model_and_transforms(
                oc_model_name, pretrained=pretrained, device="cpu"
            )
        sd = oc_model.state_dict()
        
        # Extract architecture parameters from state dict
        if 'visual.conv1.weight' in sd:
            vision_width = sd['visual.conv1.weight'].shape[0]
            patch_size = sd['visual.conv1.weight'].shape[-1]
        elif 'visual.patch_embed.proj.weight' in sd:  # timm style
            vision_width = sd['visual.patch_embed.proj.weight'].shape[0]
            patch_size = sd['visual.patch_embed.proj.weight'].shape[-1]
        else:
            raise ValueError("Cannot determine vision config from state dict")
            
        vision_layers = len([k for k in sd.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        if vision_layers == 0:
            vision_layers = len([k for k in sd.keys() if k.startswith("visual.transformer.resblocks.") and k.endswith(".attn.in_proj_weight")])
        
        if 'visual.positional_embedding' in sd:
            pos_embed_len = sd['visual.positional_embedding'].shape[0]
            grid_area = pos_embed_len - 1
            grid_dim = int(grid_area ** 0.5)
            image_resolution = grid_dim * patch_size
        else:
            image_resolution = 224
        
        if vision_head_width is None:
            if vision_width == 1280:
                vision_head_width = 80
            elif vision_width == 1664:
                vision_head_width = 104
            else:
                vision_head_width = 64
        
        vision_heads = vision_width // vision_head_width
        
        embed_dim = sd["text_projection"].shape[1]
        vocab_size = sd["token_embedding.weight"].shape[0]
        context_length = sd["positional_embedding"].shape[0]
        transformer_width = sd["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in sd.keys() if k.startswith("transformer.resblocks")))
        
        self.input_size = image_resolution
        self.patch_size = patch_size
        self.vision_width = vision_width
        self.vision_layers = vision_layers
        self.vision_head_width = vision_head_width
        
        if getattr(self, "mask_layers", None) is None:
            self.mask_layers = 3

        print(f"Detected config: Resolution={image_resolution}, Patch={patch_size}, "
              f"VisionWidth={vision_width}, VisionLayers={vision_layers}, VisionHeads={vision_heads}, "
              f"HeadWidth={vision_head_width}")

        from .clip.model import CLIP, convert_weights
        self.model = CLIP(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=patch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            vision_head_width=vision_head_width,
            mask_layers=self.mask_layers,
        )
        
        self._load_open_clip_state_dict(sd)
        convert_weights(self.model)
        
        mean = getattr(oc_model.visual, 'image_mean', (0.48145466, 0.4578275, 0.40821073))
        std = getattr(oc_model.visual, 'image_std', (0.26862954, 0.26130258, 0.27577711))
        
        self.preprocess = T.Compose([
            T.Resize((image_resolution, image_resolution), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def _load_open_clip_state_dict(self, sd):
        """Map OpenCLIP state dict to local format."""
        new_sd = {}
        prefix_map = {
            "visual.patch_embed.proj.": "visual.conv1.",
            "visual.blocks.": "visual.transformer.resblocks.",
            "visual.norm.": "visual.ln_post.",
            "transformer.resblocks.": "transformer.resblocks.",
        }
        
        for k, v in sd.items():
            new_k = k
            if new_k.startswith("module."):
                new_k = new_k[7:]
            
            for oc_pre, ai_pre in prefix_map.items():
                if new_k.startswith(oc_pre):
                    new_k = new_k.replace(oc_pre, ai_pre)
                    break
            
            if "attn.qkv.weight" in new_k or "attn.qkv.bias" in new_k:
                 new_k = new_k.replace("qkv", "in_proj")
            
            new_sd[new_k] = v

        msg = self.model.load_state_dict(new_sd, strict=False)
        print(f"Weights loaded: {msg}")
        if msg.missing_keys:
            critical = [k for k in msg.missing_keys if "visual" in k or "transformer" in k]
            if critical:
                print(f"WARNING: Critical keys missing: {critical[:5]}...")

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text(s) -> (N, D) normalized features."""
        if isinstance(texts, str):
            texts = [texts]
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        feats = self.model.encode_text(tokens, mask=None)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_image_single(self, image_tensor: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode one image -> (feature, attn)."""
        image_tensor = image_tensor.to(self.device).unsqueeze(0)
        if mask is not None:
            mask = mask.to(self.device_tensor)
            if mask.dim() == 3:
                 mask = mask.unsqueeze(1)
            mask = mask.to(dtype=next(self.mask_net.parameters()).dtype)
            f, a = self.model.encode_image(image_tensor, (self.mask_net, mask))
        else:
            f, a = self.model.encode_image(image_tensor, None)
        return f.squeeze(0), (a.squeeze(0) if a is not None else None)

    def encode_images_batch(
        self,
        image_tensors: torch.Tensor,
        masks: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch-encode images with parallel processing."""
        all_feats, all_attns = [], []
        n = image_tensors.shape[0]

        for start in range(0, n, self.encode_batch_size):
            end = min(start + self.encode_batch_size, n)
            batch_imgs = image_tensors[start:end].to(self.device)
            batch_masks = masks[start:end].to(self.device_tensor)
            
            if batch_masks.dim() == 3:
                 batch_masks = batch_masks.unsqueeze(1)
            batch_masks = batch_masks.to(dtype=next(self.mask_net.parameters()).dtype)
            
            feats, attns = self.model.encode_image(batch_imgs, (self.mask_net, batch_masks), need_weights=need_weights)
            
            all_feats.append(feats)
            if need_weights:
                all_attns.append(attns)

        features = F.normalize(torch.cat(all_feats, dim=0), dim=-1)
        attns = torch.cat(all_attns, dim=0) if need_weights and all_attns else None
        return features, attns

