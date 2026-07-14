# eraser/data_loader.py
"""
Data loader that yields (image, prompt, is_target) per sample.

- Supports:
  • target-only training (related path missing/empty)
  • mixed target/related training via a single DataLoader
  • approximate target fraction via WeightedRandomSampler (p_target)
  • variable aspect ratios with safe padding in collate_fn
"""

import os
import json
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from PIL import Image


# ----------------------------- Image helpers -----------------------------

def image_resize(img: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize so the longer side == max_size, preserving aspect ratio."""
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int(round((max_size / w) * h))
    else:
        new_h = max_size
        new_w = int(round((max_size / h) * w))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def crop_to_aspect_ratio(image: Image.Image, ratio: str = "16:9") -> Image.Image:
    """Center-crop to a given aspect ratio."""
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1),
    }
    if ratio not in ratio_map:
        return image
    target_w, target_h = ratio_map[ratio]
    target_ratio = target_w / target_h
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    return image.crop(crop_box)


# ----------------------------- Dataset classes -----------------------------

class CustomImageDataset(Dataset):
    """
    Reads images + sidecar captions (.json with {"caption": "..."} or .txt).
    Returns tensors scaled to [-1, 1], dtype float32, shape (C,H,W) with H,W multiple of 32.
    """
    def __init__(
        self,
        img_dir: Optional[str],
        img_size: int = 512,
        caption_type: str = "json",   # "json" or "txt"
        random_ratio: bool = False,
        ratio_choices: Optional[List[str]] = None,  # e.g., ["16:9","1:1","4:3","default"]
    ):
        self.img_dir = img_dir if img_dir and os.path.isdir(img_dir) else None
        self.img_size = int(img_size)
        self.caption_type = caption_type
        self.random_ratio = bool(random_ratio)
        self.ratio_choices = ratio_choices or ["16:9", "default", "1:1", "4:3"]

        if self.img_dir is None:
            self.images: List[str] = []
        else:
            exts = (".jpg", ".jpeg", ".png")
            self.images = [
                os.path.join(self.img_dir, f)
                for f in os.listdir(self.img_dir)
                if f.lower().endswith(exts)
            ]
            self.images.sort()

    def __len__(self) -> int:
        return len(self.images)

    def _read_caption(self, stem: str) -> str:
        cap_path = f"{stem}.{self.caption_type}"
        if self.caption_type.lower() == "json":
            with open(cap_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("caption", "")
        else:
            with open(cap_path, "r", encoding="utf-8") as f:
                return f.read().strip()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # If empty dataset, return a dummy (keeps pipeline robust)
        if len(self.images) == 0:
            return torch.zeros(3, 32, 32, dtype=torch.float32), ""

        path = self.images[idx]
        try:
            img = Image.open(path).convert("RGB")

            # optional aspect-ratio crop
            if self.random_ratio:
                ratio = random.choice(self.ratio_choices)
                if ratio != "default":
                    img = crop_to_aspect_ratio(img, ratio)

            # resize and then enforce multiples of 32
            # img = image_resize(img, self.img_size)
            # w, h = img.size
            # new_w = max(32, (w // 32) * 32)
            # new_h = max(32, (h // 32) * 32)
            # img = img.resize((new_w, new_h), resample=Image.BICUBIC)
            img = img.resize((self.img_size, self.img_size), resample=Image.BICUBIC)


            # to tensor in [-1, 1], float32
            arr = np.asarray(img, dtype=np.float32)  # [H,W,3], 0..255
            arr = (arr / 127.5) - 1.0
            img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]

            stem, _ = os.path.splitext(path)
            prompt = self._read_caption(stem)
            return img_t, prompt

        except Exception as e:
            print(f"[Dataset] {e} @ {path}")
            # Try another index
            return self.__getitem__(random.randint(0, len(self.images) - 1))


class WithLabel(Dataset):
    """Wraps a base dataset to return (img, prompt, is_target)."""
    def __init__(self, base: Dataset, is_target: bool):
        self.base = base
        self.flag = 1.0 if is_target else 0.0

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, float]:
        img, prompt = self.base[idx]
        return img, prompt, self.flag


# ----------------------------- Building datasets/loader -----------------------------

def _build_mixed_dataset(
    target_dir: Optional[str],
    related_dir: Optional[str],
    **ds_kwargs,
) -> Tuple[Dataset, int, int]:
    """Create a ConcatDataset of target/related (labeled). If related is empty, return target-only."""
    ds_t = CustomImageDataset(target_dir, **ds_kwargs)
    Lt = len(ds_t)

    has_related = bool(related_dir) and os.path.isdir(related_dir)
    ds_r = CustomImageDataset(related_dir, **ds_kwargs) if has_related else None
    Lr = 0 if ds_r is None else len(ds_r)

    if Lr == 0:
        return WithLabel(ds_t, True), Lt, 0

    return ConcatDataset([WithLabel(ds_t, True), WithLabel(ds_r, False)]), Lt, Lr


def _pad_to_max(batch_imgs: List[torch.Tensor], pad_value: float = -1.0) -> torch.Tensor:
    """Pad variable-sized CHW tensors to (C, Hmax, Wmax) and stack -> (B,C,Hmax,Wmax)."""
    C = batch_imgs[0].shape[0]
    Hmax = max(x.shape[1] for x in batch_imgs)
    Wmax = max(x.shape[2] for x in batch_imgs)
    out = []
    for x in batch_imgs:
        _, H, W = x.shape
        pad = (0, Wmax - W, 0, Hmax - H)  # (left, right, top, bottom)
        x = F.pad(x, pad, value=pad_value)
        out.append(x)
    return torch.stack(out, dim=0)


def collate_triplet(batch: List[Tuple[torch.Tensor, str, float]]) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Collate (img, prompt, is_target) with padding for variable sizes."""
    imgs, prompts, flags = zip(*batch)
    imgs_t = _pad_to_max(list(imgs), pad_value=-1.0)
    is_target = torch.tensor(flags, dtype=torch.float32)
    return imgs_t, list(prompts), is_target


def loader_triplet(
    target_dir: Optional[str],
    related_dir: Optional[str] = None,
    train_batch_size: int = 1,
    num_workers: int = 0,
    p_target: float = 0.5,
    use_weighted_sampling: bool = True,
    **ds_kwargs,
) -> DataLoader:
    """
    Returns a single DataLoader that yields (img, prompt, is_target).
    - If related_dir is empty/missing/has 0 images, falls back to target-only.
    - If use_weighted_sampling, approximates p_target fraction via WeightedRandomSampler.
    """
    dataset, Lt, Lr = _build_mixed_dataset(target_dir, related_dir, **ds_kwargs)

    # Target-only case
    if Lr == 0:
        return DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_triplet,
        )

    # Mixed set
    if not use_weighted_sampling:
        # Natural frequency given by sizes; simple shuffle each epoch
        return DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_triplet,
        )

    # Weighted sampler to match desired p_target in expectation
    N = Lt + Lr
    w_t = (p_target / max(Lt, 1))
    w_r = ((1.0 - p_target) / max(Lr, 1))
    weights = torch.tensor([w_t] * Lt + [w_r] * Lr, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=N,           # one pass worth; DataLoader re-seeds each epoch
        replacement=True,
    )

    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_triplet,
    )


# ----------------------------- (Optional) legacy API -----------------------------

def loader(train_batch_size: int, num_workers: int, **args) -> DataLoader:
    """
    Legacy single-dir loader (returns (img, prompt) only).
    Prefer `loader_triplet` for (img, prompt, is_target).
    """
    ds = CustomImageDataset(**args)
    return DataLoader(
        ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


__all__ = [
    "CustomImageDataset",
    "WithLabel",
    "loader_triplet",
    "loader",
    "collate_triplet",
]

