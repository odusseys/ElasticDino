import math

import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize


class DinoV3(nn.Module):
    """DINOv3 patch features with the same interface as the DINOv2 adapter."""

    def __init__(self, repo_id, *, token=None, revision=None):
        super().__init__()
        from transformers import AutoModel

        self.dino_backbone = AutoModel.from_pretrained(repo_id, token=token, revision=revision)
        self.dino_backbone.requires_grad_(False).eval()
        self.feature_size = self.dino_backbone.config.hidden_size
        self.patch_size = self.dino_backbone.config.patch_size
        self.prefix_tokens = 1 + self.dino_backbone.config.num_register_tokens

    def _forward(self, images, *, hidden_states=False):
        images = normalize(images, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return self.dino_backbone(pixel_values=images, output_hidden_states=hidden_states)

    def _patch_features(self, tokens):
        patches = tokens[:, self.prefix_tokens:, :]
        size = math.isqrt(patches.shape[1])
        if size * size != patches.shape[1]:
            raise ValueError("DINOv3 returned a non-square patch grid")
        return patches.reshape(patches.shape[0], size, size, self.feature_size).permute(0, 3, 1, 2)

    @torch.no_grad()
    def get_features_for_tensor(self, images):
        return self._patch_features(self._forward(images).last_hidden_state)

    @torch.no_grad()
    def get_features_and_class_token_for_tensor(self, images):
        tokens = self._forward(images).last_hidden_state
        return self._patch_features(tokens), tokens[:, 0]

    @torch.no_grad()
    def get_intermediate_features_for_tensor(self, images, n):
        if not isinstance(n, int) or not 1 <= n <= self.dino_backbone.config.num_hidden_layers:
            raise ValueError("n_hidden_layers must be between 1 and the backbone layer count")
        states = self._forward(images, hidden_states=True).hidden_states[-n:]
        return [self._patch_features(self.dino_backbone.norm(state)) for state in states]
