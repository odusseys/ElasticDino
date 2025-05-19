import torch.nn as nn
import torch
from PIL import Image
import numpy as np

class PCA(nn.Module):
    def __init__(self, n_components, scale=True):
        super().__init__()
        self.n_components = n_components
        self.scale = scale

    def fit(self, X):
        b, n, d = X.shape
        self.register_buffer("mean_", X.mean(1, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh.transpose(1, 2)[:, :, :self.n_components]
        self.register_buffer("components_", Vt)
        std = S[:, :self.n_components].unsqueeze(1).sqrt()
        self.register_buffer("std_", std)
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        unscaled = torch.bmm(X - self.mean_, self.components_)
        if self.scale:
          scaled = unscaled / self.std_  # Scale for unit variance
        return scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        Y = Y * self.std_  # Unscale
        return  torch.bmm(Y, self.components_.transpose(1, 2)) + self.mean_


@torch.compile
def compute_pca(f1, n):
  f1_size = f1.shape[-1]
  batch_size = f1.shape[0]
  f1 = f1.reshape((batch_size, N_FEATURES, f1_size * f1_size)).transpose(1, 2)
  pca = PCA(n)
  f1_reduced = pca.fit_transform(f1)
  f1_reduced = f1_reduced.transpose(1, 2).reshape((batch_size, n, f1_size, f1_size))
  return f1_reduced, pca

@torch.compile
def apply_pca(pca, f1):
  f1_size = f1.shape[-1]
  batch_size = f1.shape[0]
  f1 = f1.reshape((batch_size, N_FEATURES, f1_size * f1_size)).transpose(1, 2)
  f1_reduced = pca.transform(f1)
  n = f1_reduced.shape[2]
  f1_reduced = f1_reduced.transpose(1, 2).reshape((batch_size, n, f1_size, f1_size))
  return f1_reduced

def reduce_features(f1, f2, n, scale=True):
  f1_size = f1.shape[-1]
  f2_size = f2.shape[-1]
  batch_size = f1.shape[0]
  f1 = f1.reshape((batch_size, N_FEATURES, f1_size * f1_size)).transpose(1, 2)
  f2 = f2.reshape((batch_size, N_FEATURES, f2_size * f2_size)).transpose(1, 2)
  pca = PCA(n, scale)
  f1_reduced = pca.fit_transform(f1)
  f2_reduced = pca.transform(f2)
  f1_reduced = f1_reduced.transpose(1, 2).reshape((batch_size, n, f1_size, f1_size))
  f2_reduced = f2_reduced.transpose(1, 2).reshape((batch_size, n, f2_size, f2_size))
  return f1_reduced, f2_reduced, pca

def reduce_dimension(f1, other_features, n, scale=True):
  batch_size = f1.shape[0]
  n_features = f1.shape[1]
  size_1 = f1.shape[2]
  other_sizes = [f2.shape[2] for f2 in other_features]
  f1 = f1.permute((0, 2, 3, 1)).reshape((batch_size, size_1 * size_1, n_features))
  other_features = [f2.permute((0, 2, 3, 1)).reshape((batch_size, f2.shape[2] * f2.shape[2], n_features)) for f2 in other_features]
  pca = PCA(n_components=n).fit(f1)
  f1 = pca.transform(f1).reshape((batch_size, size_1, size_1, n)).permute(0, 3, 1, 2)
  other_features = [pca.transform(f2).reshape((batch_size, size_2, size_2, n)).permute(0, 3, 1, 2) for f2, size_2 in zip(other_features, other_sizes)]
  m = min(torch.min(f1), *[torch.min(f2) for f2 in other_features]) if other_features else torch.min(f1)
  M = max(torch.max(f1), *[torch.max(f2) for f2 in other_features]) if other_features else torch.max(f1)
  f1 = (f1 - m) / (M - m)
  other_features = [(f2 - m) / (M - m) for f2 in other_features]
  return f1, other_features


def random_projection(features_list, n_projected_features, dtype=torch.float32):
  proj = torch.randn(features_list[0].shape[0],
                    features_list[0].shape[1],
                    n_projected_features, device="cuda", dtype=dtype, requires_grad=False)
  proj /= proj.square().sum(1, keepdim=True).sqrt()
  return [torch.einsum("bchw,bcd->bdhw", features, proj) for features in features_list]


def debug_features(f1, other_features=[], display_size=128):
  f1 = f1.to(dtype=torch.float32)
  other_features = [f2.to(dtype=torch.float32) for f2 in other_features]
  f1, other_features = reduce_dimension(f1.unsqueeze(0), [f2.unsqueeze(0) for f2 in other_features], 3)
  f1 = f1[0].permute(1, 2, 0).detach().cpu().float().numpy().squeeze() * 255
  images = [Image.fromarray(f1.astype(np.uint8)).resize((display_size, display_size), 0)]
  for f2 in other_features:
    f2 = f2[0].permute(1, 2, 0).detach().cpu().float().numpy().squeeze() * 255
    images.append(Image.fromarray(f2.astype(np.uint8)).resize((display_size, display_size), 0))
  return Image.fromarray(np.hstack(images).astype(np.uint8))

def debug_individual_features(features, size=128, clip=5):
  images = []
  for i in range(min(5, len(features))):
    f = np.clip(features[i], -clip, clip)
    f = (f + clip) / (2 * clip) * 255
    images.append(Image.fromarray(f.astype(np.uint8)).resize((size, size), 0))
  # make row of images and display
  return Image.fromarray(np.hstack(images).astype(np.uint8))
