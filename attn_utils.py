import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention, FluxAttnProcessor2_0
from diffusers.models.embeddings import apply_rotary_emb
from tqdm import tqdm
import collections
from PIL import Image, ImageDraw, ImageFont, ImageOps
from typing import Optional, Dict, Any, List
import numpy as np
from einops import rearrange
import math
import os
# from attention_map_diffusers import process_token
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# --- HELPER FUNCTIONS ---
def set_attention_processors(pipeline, processor):
    """Set attention processor for all attention layers."""
    for name, module in pipeline.transformer.named_modules():
        if isinstance(module, Attention):
            module.processor = processor

def set_attention_processors_dit(dit, config):
    all_processors = {}
    for name, module in dit.named_modules():
        if isinstance(module, Attention):
            print(name)
            processor = FluxAttentionEraser(height=config['height'], save_all_tokens=True)
            all_processors[name] = processor
            module.processor = processor
    return all_processors


def get_token_index(tokenizer, prompt, token_word):
    """Finds the (first) index of a specific token in the tokenized prompt."""
    token_ids = tokenizer.encode(token_word, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Token '{token_word}' could not be encoded.")
    
    # token_id = token_ids[0]  # Use first token if word is split TODO: modify?
    tokenized_input = tokenizer(
        prompt, 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    )
    indices = [torch.where(tokenized_input.input_ids[0] == token_id)[0].item() for token_id in token_ids]
    if len(indices) == 0:
        raise ValueError(f"Token '{token_word}' not found in prompt '{prompt}'.")
    
    return indices

import re

import re

def get_token_indecies(tokenizer, prompt, token_word):
    """
    Return the list of token indices (contiguous) covering the FIRST occurrence
    of `token_word` in `prompt`, case-insensitive and ignoring spaces/hyphens/underscores.
    Indices are with respect to the tokenizer's tokens for the ORIGINAL prompt.
    Example: 'Spider-man' in prompt matches concept 'Spiderman' and might return [12,13].
    """
    # Tokenize the *original prompt* exactly once
    ids  = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)

    # Normalizers: strip tokenizer markers and collapse separators
    def norm_tok(t):
        t = t.replace("Ġ", "").replace("▁", "").replace("##", "")  # common BPE markers
        return re.sub(r"[ \-_]", "", t.lower())

    target = re.sub(r"[ \-_]", "", token_word.lower())

    cleaned = [norm_tok(t) for t in toks]

    # Sliding window over tokens; only count characters from cleaned tokens
    for i in range(len(cleaned)):
        acc = ""
        idxs = []
        for j in range(i, len(cleaned)):
            c = cleaned[j]
            # specials often become '', keep index space consistent but don't add chars
            if c:
                acc += c
                idxs.append(j)
            # Early exits
            if not target.startswith(acc):
                break
            if acc == target:
                return idxs
    return []



def combine_images(images, cols=2):
    """Combines a list of PIL images into a single grid image."""
    if not images:
        return None
    
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    
    return grid


def scaled_dot_product_attention_with_weights(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, calculate_original=False, eraser_dims=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    
    if calculate_original:
        attn_weight_orig = attn_weight[:, :, :, eraser_dims:]  # slice last dim
        attn_weight_orig = torch.softmax(attn_weight_orig, dim=-1)
        attn_weight_orig_drop = F.dropout(attn_weight_orig, p=dropout_p, training=True)
        original_out = attn_weight_orig_drop @ value[:, :, eraser_dims:, :]

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight_dropout = F.dropout(attn_weight, p=dropout_p, training=True)

    if not calculate_original:
        return attn_weight_dropout @ value, attn_weight
    else:
        return attn_weight_dropout @ value, attn_weight, original_out, attn_weight_orig

def get_inner_dim(pipeline):
    # TODO: maybe modify how extracted
    inner_dim = None 
    for _, module in pipeline.transformer.named_modules():
        if isinstance(module, Attention) and hasattr(module, 'add_k_proj'):
            inner_dim = module.add_k_proj.out_features
            break
    if inner_dim is None:
        raise ValueError("Could not find a cross-attention layer to determine the inner dimension.")
    print(f"Found attention inner dimension: {inner_dim}")
    return inner_dim

def reshape_value(value, last_dim=3072):
    if value.shape[-1] != last_dim:
        if (value.shape[1] == 24) and (value.shape[2] in [1,2,3,4,5,6,7]) and (value.shape[-1] == 128):
            value = value.permute(0,2,1,3)
            B, T, H, D = value.shape
            value = value.reshape(B, T, -1)
            if value.shape[-1] != last_dim:
                print(f'{value.shape}, last dim doesnt mathc {last_dim}')
                breakpoint
        elif value.shape == torch.Size([1, 24, 128]):
            value = value.reshape(1, 1, -1)
        else:
            print(value.shape, 'attention processor val shape incorrect')
            breakpoint()
    assert value.shape[-1] == last_dim
    if value.ndim == 3:
        value = value.mean(dim=1, keepdim=True) # assumes shape [Batch, Tokens, Heads*Dims(3072)]
    return value

### adapted from attentiion_map_diffusers ###
# def save_attention_image(attn_map, tokens, batch_dir, to_pil):
#     startofword = True
#     for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
#         token, startofword = process_token(token, startofword)
#         # to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}.png'))
#         to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}.jpg'), format='JPEG')



# -------- utility: pad tiles in a row to same height ----------
def _hpad_tiles_to_max_height(tiles: list[Image.Image], bg=(255, 255, 255), gutter: int = 0):
    if not tiles:
        return Image.new("RGB", (1, 1), bg)
    max_h = max(im.height for im in tiles)
    padded = [ImageOps.expand(im, border=(0, 0, 0, max_h - im.height), fill=bg) for im in tiles]

    w = sum(im.width for im in padded) + gutter * (len(padded) - 1)
    row = Image.new("RGB", (w, max_h), bg)
    x = 0
    for i, im in enumerate(padded):
        row.paste(im, (x, 0))
        x += im.width + (gutter if i < len(padded) - 1 else 0)
    return row


# Per-layer normalization bounds calculated from your data
# Each entry is {layer_index: (vmin, vmax)}
LAYER_NORM_BOUNDS = {
    0: (0.000148, 0.000308), 2: (0.000206, 0.001413),
    3: (0.000275, 0.002396), 4: (5.42e-05, 0.001939),
    5: (0.000140, 0.002354), 6: (0.000157, 0.001843),
    8: (0.000174, 0.013989), 9: (0.000136, 0.014026),
    10: (0.000168, 0.007425), 11: (0.000143, 0.018994),
    12: (4.88e-05, 0.04541), 13: (7.36e-05, 0.011505),
    14: (0.000259, 0.03595), 16: (3.56e-05, 0.011829),
    18: (4.13e-05, 0.070007), 19: (1.50e-05, 0.046875),
    20: (3.40e-05, 0.036865), 21: (1.95e-05, 0.030029),
    22: (2.95e-05, 0.038086), 23: (3.58e-05, 0.046326),
    24: (5.75e-05, 0.086304), 25: (3.61e-05, 0.065552),
    26: (2.60e-05, 0.036743), 27: (2.70e-05, 0.096558),
    28: (2.80e-05, 0.126709), 29: (1.00e-05, 0.049805),
    30: (1.11e-05, 0.057312), 31: (2.5e-06, 0.037354),
    32: (2.41e-06, 0.068726), 33: (3.3e-06, 0.011108),
    34: (4.8e-06, 0.010254), 35: (4.0e-06, 0.010864),
    36: (6.4e-06, 0.004551), 37: (1.19e-05, 0.009521),
    38: (1.80e-05, 0.011047), 39: (1.47e-05, 0.014587),
    41: (5.1e-06, 0.012115), 42: (8.8e-06, 0.01355),
    43: (6.5e-06, 0.012268), 44: (5.2e-06, 0.010925),
    45: (5.0e-05, 0.016968), 46: (4.2e-05, 0.012207),
    48: (0.000104, 0.00885), 49: (4.7e-05, 0.007629),
    50: (6.5e-05, 0.012756), 51: (2.4e-05, 0.014587),
    52: (8.2e-05, 0.011353), 53: (9.3e-05, 0.013855),
    54: (9.9e-05, 0.00824), 55: (0.000217, 0.002304),
    56: (0.00047, 0.002579)
}

# A fallback for any layer not in the dictionary, calculated from all data
FALLBACK_BOUNDS = (7.1e-05, 0.02)

######### TODO: remove after DEBUG ##########
def _sanitize_for_vis(t: torch.Tensor) -> torch.Tensor:
    """
    Make a tensor safe for ToPILImage:
    - detach, float32, cpu
    - replace NaN/±Inf with 0
    - normalize to [0,1] (robust to constant/degenerate maps)
    - accept HxW or 1xHxW or 3xHxW; returns the same shape (but float32 on CPU)
    """
    t = t.detach().float().cpu()
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    # If 1xHxW, drop the channel for grayscale PIL; keep 3xHxW as-is.
    if t.dim() == 3 and t.size(0) == 1:
        t = t[0]

    # Normalize to [0,1] safely
    t_min = float(t.min())
    t_max = float(t.max())
    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        # degenerate map -> clamp is fine
        t = t.clamp(0, 1)
    else:
        t = (t - t_min) / (t_max - t_min + 1e-8)
        t = t.clamp(0, 1)
    return t
############ TODO remove after debug ############

def prep_attention_map_for_vis(attn_map, head_index=-1, height=32, is_eraser=False, layer_ind=-1):
    to_pil = ToPILImage()
    if attn_map.ndim != 3 or attn_map.shape[-1] != 1024:
        raise ValueError("Expected attention maps of shape [1, num_heads, num_patches]")
    
    if head_index == -1: # avg across all attention heads
        single_map = attn_map.mean(dim=1)[0]  # [num_patches]
    else:
        single_map = attn_map[0, head_index]  # shape: [H]
    
    image = single_map.view(1, height, height)
    
    # This print statement is useful for collecting new data if distributions shift
    # print(f"min: {image.min()}, max: {image.max()}, mean: {image.mean()}, layer_ind: {layer_ind}, eraser? {is_eraser}")
   
    # Look up the correct bounds for the current layer
    vmin, vmax = LAYER_NORM_BOUNDS.get(layer_ind, FALLBACK_BOUNDS)

    # Clamp the image to the layer-specific range and normalize
    image = image.clamp(min=vmin, max=vmax)
    image = (image - vmin) / max(vmax - vmin, 1e-8)

    # optional gamma to boost mid/lows (helps “see” Spiderman in faint maps)
    gamma = 0.6  # tweak 0.4–0.8
    image = image.pow(gamma)
    
    pil = to_pil(image.to(torch.float32))
    # pil = to_pil(_sanitize_for_vis(image)) # TODO: remove after debug
  
    return pil

def _get_font(size: int):
    for p in ("DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

def add_label_with_bg(img: Image.Image, text: str, where: str = "below",
                      font_size: int = 16, pad_x: int = 8, pad_y: int = 4,
                      fixed_label_h: int | None = 26):
    """Add a white strip with text above/below image, using a fixed label height."""
    font = _get_font(font_size)

    # measure text; account for fonts with nonzero bbox top
    tmp = Image.new("RGB", (1, 1))
    dr = ImageDraw.Draw(tmp)
    l, t, r, b = dr.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t

    label_h = fixed_label_h if fixed_label_h is not None else (th + 2 * pad_y)
    W = img.width
    label = Image.new("RGB", (W, label_h), (255, 255, 255))
    d = ImageDraw.Draw(label)
    d.text(((W - tw) // 2, (label_h - th) // 2 - t), text, font=font, fill=(0, 0, 0))

    out = Image.new("RGB", (W, img.height + label_h), (255, 255, 255))
    if where == "above":
        out.paste(label, (0, 0))
        out.paste(img, (0, label_h))
    else:
        out.paste(img, (0, 0))
        out.paste(label, (0, img.height))
    return out

def add_row_header(row_img: Image.Image, header: str, font_size: int = 18, pad_x: int = 10):
    """Add a left header box before a horizontal row image, matched to row height."""
    font = _get_font(font_size)
    tmp = Image.new("RGB", (1, 1))
    dr = ImageDraw.Draw(tmp)
    l, t, r, b = dr.textbbox((0, 0), header, font=font)
    tw, th = r - l, b - t

    box_w = tw + 2 * pad_x
    box_h = row_img.height
    box = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    d = ImageDraw.Draw(box)
    d.text((pad_x, (box_h - th) // 2 - t), header, font=font, fill=(0, 0, 0))

    out = Image.new("RGB", (box_w + row_img.width, row_img.height), (255, 255, 255))
    out.paste(box, (0, 0))
    out.paste(row_img, (box_w, 0))
    return out

def _hpad_tiles_to_max_height(tiles: list[Image.Image], bg=(255, 255, 255), gutter: int = 0):
    if not tiles:
        return Image.new("RGB", (1, 1), bg)
    max_h = max(im.height for im in tiles)
    padded = [ImageOps.expand(im, border=(0, 0, 0, max_h - im.height), fill=bg) for im in tiles]

    w = sum(im.width for im in padded) + gutter * (len(padded) - 1)
    row = Image.new("RGB", (w, max_h), bg)
    x = 0
    for i, im in enumerate(padded):
        row.paste(im, (x, 0))
        x += im.width + (gutter if i < len(padded) - 1 else 0)
    return row

def save_side_by_side_attention_map(
    original_target_map: torch.Tensor,
    extended_target_map: torch.Tensor,
    original_eraser_map: torch.Tensor,
    eraser_map: torch.Tensor,
    extended_image_to_text_attention: torch.Tensor,
    original_img_to_text_attention: torch.Tensor,
    all_tokens: list,
    target_token_idx_lst: list,
    out_path: str,
    head_index: int = -1,
    height: int = 32,
    cmap: str = "inferno",
    layer_ind: int = -1,
    token_font_size: int = 10,
    row_font_size: int = 10
):
    """
    Top→bottom:
      1) eraser
      2) original target token(s)      [optional]
      3) extended target token(s)      [optional]
      4) ORIGINAL full prompt (per-token maps)
      5) EXTENDED/ERASED full prompt (per-token maps)
    """
    # if original_target_map is None or extended_target_map is None or target_token_idx_lst is None:
    #     breakpoint()
    # --- Row 1: eraser ---
    orig_era_pil = prep_attention_map_for_vis(original_eraser_map, head_index, height, is_eraser=False, layer_ind=layer_ind)
    eraser_pil   = prep_attention_map_for_vis(eraser_map, head_index, height, is_eraser=True,  layer_ind=layer_ind)

    orig_era_tile = add_label_with_bg(orig_era_pil, "original", where="below", font_size=6, fixed_label_h=14)
    eraser_tile   = add_label_with_bg(eraser_pil,   "eraser",   where="below", font_size=6, fixed_label_h=14)

    top_row = _hpad_tiles_to_max_height([orig_era_tile, eraser_tile], gutter=0)

    # --- Row 2: original target tokens (optional) ---
    orig_row = None
    if original_target_map is not None and target_token_idx_lst is not None:
        orig_target_tiles = []
        for i, idx in enumerate(target_token_idx_lst):
            token = all_tokens[idx]
            attn_map = original_target_map[:, :, :, i]
            img = prep_attention_map_for_vis(attn_map, head_index, height, layer_ind=layer_ind)
            orig_target_tiles.append(add_label_with_bg(img, f"original {token}", where="below", font_size=6, fixed_label_h=14))
        orig_row = _hpad_tiles_to_max_height(orig_target_tiles, gutter=0)

    # --- Row 3: extended target tokens (optional) ---
    ext_row = None
    if extended_target_map is not None and target_token_idx_lst is not None:
        ext_target_tiles = []
        for i, idx in enumerate(target_token_idx_lst):
            token = all_tokens[idx]
            attn_map = extended_target_map[:, :, :, i]
            img = prep_attention_map_for_vis(attn_map, head_index, height, layer_ind=layer_ind)
            ext_target_tiles.append(add_label_with_bg(img, f"ext {token}", where="below", font_size=6, fixed_label_h=14))
        ext_row = _hpad_tiles_to_max_height(ext_target_tiles, gutter=0)

    # --- Rows 4 & 5: full prompt (always present) ---
    original_tiles, extended_tiles = [], []
    for i, tok in enumerate(all_tokens):
        if tok == "</s>":
            continue
        original_tiles.append(
            add_label_with_bg(
                prep_attention_map_for_vis(original_img_to_text_attention[:, :, :, i], head_index, height, layer_ind=layer_ind),
                tok, where="above", font_size=token_font_size
            )
        )
        extended_tiles.append(
            add_label_with_bg(
                prep_attention_map_for_vis(extended_image_to_text_attention[:, :, :, i+1], head_index, height, layer_ind=layer_ind),
                tok, where="above", font_size=token_font_size
            )
        )

    original_row = _hpad_tiles_to_max_height(original_tiles, gutter=0)
    extended_row = _hpad_tiles_to_max_height(extended_tiles, gutter=0)

    original_row = add_label_with_bg(original_row, "original",
                                     where="above", font_size=row_font_size,
                                     fixed_label_h=20)
    extended_row = add_label_with_bg(extended_row, "erased",
                                     where="above", font_size=row_font_size,
                                     fixed_label_h=20)

    # --- Final stack ---
    rows = [top_row]
    if orig_row is not None:
        rows.append(orig_row)
    if ext_row is not None:
        rows.append(ext_row)
    rows.extend([original_row, extended_row])

    final_w = max(row.width for row in rows)
    final_h = sum(row.height for row in rows)

    final_img = Image.new("RGB", (final_w, final_h), (255, 255, 255))
    y = 0
    for row in rows:
        final_img.paste(row, (0, y))
        y += row.height

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_img.save(out_path, format='JPEG')

def normalize_val_tensor_shape(t: torch.Tensor, inner_dim: int):
    """
    Accept [H,D], [1,H,D], [inner_dim], or [1,1,inner_dim]; return [1,1,inner_dim].
    """
    # squeeze batch dim if present
    if t.ndim == 3 and t.shape[0] == 1:      # [1,H,D] or [1,1,inner_dim]
        t = t.squeeze(0)

    if t.ndim == 2:                           # [H,D]
        t = t.reshape(1, 1, -1)
    elif t.ndim == 1:                         # [inner_dim]
        t = t.reshape(1, 1, -1)
    elif t.ndim == 3 and t.shape[:2] == (1,1):  # [1,1,inner_dim]
        pass
    else:
        raise ValueError(f"Unexpected val tensor shape {tuple(t.shape)}; "
                         f"expected [H,D], [1,H,D], [inner_dim], or [1,1,inner_dim].")

    if t.shape[-1] != inner_dim:
        raise ValueError(f"Val tensor last dim {t.shape[-1]} != inner_dim {inner_dim}")

    return t

# def save_side_by_side_attention_map(
#     original_map: torch.Tensor,
#     extended_map: torch.Tensor,
#     original_eraser_map: torch.Tensor,
#     eraser_map: torch.Tensor,
#     extended_image_to_text_attention: torch.Tensor,
#     original_img_to_text_attention: torch.Tensor,
#     all_tokens: list,
#     target_token_idx_lst: list,
#     out_path: str,
#     head_index: int = -1,
#     height: int = 32,
#     cmap: str = "inferno",
#     layer_ind: int = -1,
#     token_font_size: int = 10,
#     row_font_size: int = 10
# ):
#     """
#     Top→bottom:
#       1) eraser
#       2) original target token(s)
#       3) extended target token(s)
#       4) ORIGINAL full prompt (per-token maps)
#       5) EXTENDED/ERASED full prompt (per-token maps)
#     """
#     # --- Row 1: eraser ---
#     # Prepare original + eraser attention maps (e.g., 1D maps per head)
#     orig_era_pil = prep_attention_map_for_vis(original_eraser_map, head_index, height, is_eraser=False, layer_ind=layer_ind)
#     eraser_pil = prep_attention_map_for_vis(eraser_map, head_index, height, is_eraser=True, layer_ind=layer_ind)

#     # Add small labels
#     orig_era_tile = add_label_with_bg(orig_era_pil, "original", where="below", font_size=6, fixed_label_h=14)
#     eraser_tile   = add_label_with_bg(eraser_pil, "eraser", where="below", font_size=6, fixed_label_h=14)

#     # Horizontally stitch: [original | eraser]
#     top_row = _hpad_tiles_to_max_height([orig_era_tile, eraser_tile], gutter=0)
    
#     # --- Row 2: original target tokens ---
#     orig_target_tiles = []
#     for i, idx in enumerate(target_token_idx_lst):
#         token = all_tokens[idx]
#         attn_map = original_map[:, :, :, i]  # [1, Hds, H]
#         img = prep_attention_map_for_vis(attn_map, head_index, height, layer_ind=layer_ind)
#         orig_target_tiles.append(add_label_with_bg(img, f"original {token}", where="below", font_size=6, fixed_label_h=14))
#     orig_row = _hpad_tiles_to_max_height(orig_target_tiles, gutter=0)

#     # --- Row 3: extended target tokens ---
#     ext_target_tiles = []
#     for i, idx in enumerate(target_token_idx_lst):
#         token = all_tokens[idx]
#         attn_map = extended_map[:, :, :, i]
#         img = prep_attention_map_for_vis(attn_map, head_index, height, layer_ind=layer_ind)
#         ext_target_tiles.append(add_label_with_bg(img, f"ext {token}", where="below", font_size=6, fixed_label_h=14))
#     ext_row = _hpad_tiles_to_max_height(ext_target_tiles, gutter=0)

#     # --- Rows 4 & 5: full prompt (ORIGINAL then EXTENDED) ---
#     kept_tokens = [tok for tok in all_tokens if tok != "</s>"]
#     original_tiles, extended_tiles = [], []
#     for i, tok in enumerate(all_tokens):
#         if tok == "</s>":
#             continue
#         original_tiles.append(
#             add_label_with_bg(
#                 prep_attention_map_for_vis(original_img_to_text_attention[:, :, :, i], head_index, height, layer_ind=layer_ind),
#                 tok, where="above", font_size=token_font_size
#             )
#         )
#         extended_tiles.append(
#             add_label_with_bg(
#                 prep_attention_map_for_vis(extended_image_to_text_attention[:, :, :, i+1], head_index, height, layer_ind=layer_ind),
#                 tok, where="above", font_size=token_font_size
#             )
#         )

#     original_row = _hpad_tiles_to_max_height(original_tiles, gutter=0)
#     extended_row = _hpad_tiles_to_max_height(extended_tiles, gutter=0)

#     # add row headers
#     original_row = add_label_with_bg(original_row, "original",
#                                  where="above", font_size=row_font_size,
#                                  fixed_label_h=20)
#     extended_row = add_label_with_bg(extended_row, "erased",
#                                  where="above", font_size=row_font_size,
#                                  fixed_label_h=20)

#     # --- Final stack ---
#     final_w = max(top_row.width, orig_row.width, ext_row.width, original_row.width, extended_row.width)
#     final_h = top_row.height + orig_row.height + ext_row.height + original_row.height + extended_row.height

#     final_img = Image.new("RGB", (final_w, final_h), (255, 255, 255))
#     y = 0
#     for row in (top_row, orig_row, ext_row, original_row, extended_row):
#         final_img.paste(row, (0, y))
#         y += row.height

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     final_img.save(out_path, format='JPEG')
