import os
import socket

def set_visible_gpus(*device_ids: int):
    """
    Sets the CUDA_VISIBLE_DEVICES environment variable and enforces server-specific rules.

    This function must be called at the very beginning of your script,
    **before** you import the torch library.

    Args:
        *device_ids: A variable number of integer GPU IDs to make visible.
                     For example: set_visible_gpus(0, 1, 4)

    Raises:
        AssertionError: If the script is running on a server with 'ilves' in its
                          hostname and GPU ID 3 is included in the device_ids.
    """
    # Get the current server's hostname to check for specific rules
    hostname = socket.gethostname()

    # --- Server-Specific Rule ---
    # Check if 'ilves' is part of the hostname (e.g., 'ilves.csail.mit.edu')
    if 'ilves' in hostname:
        # Assert that GPU 3 is NOT in the list of requested devices
        assert 3 not in device_ids, (
            f"\n\n>>> Execution HALTED on server '{hostname}' <<<\n"
            f"GPU 3 is reserved on this machine and cannot be used.\n"
            f"Please remove '3' from your list of selected GPUs and re-run.\n"
        )

    # Convert the integer device IDs into a comma-separated string
    # Example: (0, 1, 4) -> "0,1,4"
    visible_devices_str = ",".join(map(str, device_ids))

    # Set the environment variable for CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices_str

    print(f"✅ CUDA_VISIBLE_DEVICES set to: '{visible_devices_str}'")

import sys
import re
from diffusers import FluxPipeline
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from math import ceil
from enum import Enum
import shutil
from datetime import datetime
import inspect
import json
import socket
from transformers import CLIPModel, CLIPProcessor


def slugify(s: str) -> str:
    SAFE_CHARS = r"[^A-Za-z0-9_\-]"

    """Filesystem-friendly name (matches your dataset folders)."""
    s = s.strip()
    s = re.sub(r"\s+", "_", s)        # spaces -> underscores
    s = re.sub(SAFE_CHARS, "", s)     # remove other punct except _ and -
    return s.lower()

def get_info_from_metadata(concept, joint_args):
    train_data_root = joint.get("train_data", "")
    key_val_root = joint.get("key_val", "")
    train_prompts = joint.get("train_prompts", "")
    train_eval_prompts = joint.get("train_eval_prompts", "")


def load_eval_prompts(file_path, concept):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entry = data[concept]
    return entry.get("erase", []), entry.get("preserve", [])

def calc_clip_score(image, text_prompt, clip_model, clip_processor, device):
    image = image.convert("RGB")
    inputs = clip_processor(text=[text_prompt], images=image, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    score = logits_per_image.item()

    return score

    # # Second method for calculation (from clip_eval_flux_erase)
    # img_e = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    # txt_e = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
    # score_2 =  float((txt_e @ img_e.T).squeeze().item())
    # img_e = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    # txt_e = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    # return float((txt_e @ img_e.T).squeeze().item())
    # inputs = processor(
    #         text=[text_prompt],
    #         images=image,
    #         return_tensors="pt", # Return PyTorch tensors
    #         padding=True
    #     ).to(device)


# ------- print console to file -------- #
_TEE_FILE = None

import sys
import os

_TEE_FILE = None

def attach_console_logger(log_path: str):
    """
    Mirrors stdout to both the terminal and a file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    class _Tee(object):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    global _TEE_FILE
    if _TEE_FILE and not _TEE_FILE.closed:
        try:
            _TEE_FILE.close()
        except Exception:
            pass
    
    _TEE_FILE = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _TEE_FILE)


def freeze_flux_pipeline(pipeline):
   for attr_name in dir(pipeline):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(pipeline, attr_name)
        except AttributeError:
            continue
        if isinstance(attr, nn.Module):
            attr.eval()
            for param in attr.parameters():
                if param.requires_grad:
                    param.requires_grad = False

def count_pipeline_parameters(pipeline):
    trainable = 0
    frozen = 0
    for attr_name in dir(pipeline):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(pipeline, attr_name)
        except AttributeError:
            continue
        if isinstance(attr, torch.nn.Module):
            for param in attr.parameters():
                if param.requires_grad:
                    trainable += param.numel()
                else:
                    frozen += param.numel()
    print(f"Trainable parameters:     {trainable:,}")
    print(f"Frozen (non-trainable):   {frozen:,}")
    print(f"Total:                    {trainable + frozen:,}")
    return trainable, frozen

# def freeze_flux_pipeline(pipeline: FluxPipeline):
#     for name, module in pipeline.named_children():
#             if isinstance(module, nn.Module):
#                 module.eval()
#                 for param in module.parameters():
#                     param.requires_grad = False

def freeze_dit(dit):
     for n, param in dit.named_parameters():
        param.requires_grad = False

def set_seed(seed):
    print(f"Setting seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Add labels below each image
def add_label(image, text, font_size=16, padding=4):
    # Load a slightly larger font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    width, height = image.size
    label_height = font_size + padding * 2

    # Create canvas with extra height for the label
    canvas = Image.new("RGB", (width, height + label_height), (255, 255, 255))
    canvas.paste(image, (0, 0))

    # Draw label below image
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, height + padding), text, fill=(0, 0, 0), font=font)

    return canvas

def arrange_images_and_save(labeled_cells, img_name, cols = 2, padding = 20, format=None, return_without_save=False):
    # Arrange cells in 2 columns
    rows = ceil(len(labeled_cells) / cols)
    cell_width = labeled_cells[0].width
    cell_height = labeled_cells[0].height

    grid_width = cols * cell_width + (cols - 1) * padding
    grid_height = rows * cell_height + (rows - 1) * padding
    final_image = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

    # Paste each cell into its grid position
    for idx, cell in enumerate(labeled_cells):
        row = idx // cols
        col = idx % cols
        x = col * (cell_width + padding)
        y = row * (cell_height + padding)
        final_image.paste(cell, (x, y))

    if return_without_save:
        return final_image
    
    # Save result
    os.makedirs(os.path.dirname(img_name), exist_ok=True)
    
    print(f"saving image to: {img_name}")
    if format is not None:
        final_image.save(img_name, format=format)
    else:
        final_image.save(img_name)

# GPUs you want to avoid (e.g., bad, reserved, or flaky)

def get_best_gpus(n=2, exclude=[]):
    if not torch.cuda.is_available():
        return None

    server_name = socket.gethostname()
    device_count = torch.cuda.device_count()
    gpu_info = []
    if (server_name == 'ilves.csail.mit.edu') or ('ilves' in server_name):
        exclude.append(3)

    for i in range(device_count):
        if i in exclude:
            continue
        try:
            stats = torch.cuda.mem_get_info(i)
            free_mem = stats[0]  # free memory in bytes
            gpu_info.append((free_mem, i))
        except RuntimeError:
            continue  # skip if device isn't available

    gpu_info.sort(reverse=True)  # sort by free memory descending
    best_gpus = [idx for _, idx in gpu_info[:n]]

    return best_gpus if len(best_gpus) == n else None

def hinge_ge_loss(e: torch.Tensor, t: torch.Tensor, margin: float = 0.0, weight_by_target=False):
    """
    Penalize places where eraser < target - margin.
    If weight_by_target=True, larger target mass gets heavier penalty.
    """
    gap = t - e + margin            # positive when e is too small
    per_elem = F.relu(gap)          # hinge
    if weight_by_target:
        per_elem = per_elem * (t.detach())  # optional weighting
    return per_elem.mean()

def _to_serializable(obj):
    # Make common non-JSON types safe (numpy, torch scalar, enums, etc.)
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    try:
        import torch
        if isinstance(obj, (torch.Tensor,)):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (torch.device,)):
            return str(obj)
        if isinstance(obj, (torch.dtype,)):
            return str(obj)
    except Exception:
        pass
    # Enums or other objects
    if hasattr(obj, "name") and hasattr(obj, "value"):
        return obj.value
    return str(obj)

def save_run_config(config: dict, dest_dir: str, filename_prefix: str = "config"):
    """
    Saves config and command line into out/<unique_id>/config/<prefix>_<timestamp>.json
    Also writes out/<unique_id>/config/argv.txt with the exact CLI.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_dir = os.path.join(dest_dir, "config")
    os.makedirs(cfg_dir, exist_ok=True)

    cfg_path = os.path.join(cfg_dir, f"{filename_prefix}_{ts}.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=_to_serializable)
    print(f"[config] Saved config -> {cfg_path}")

    argv_path = os.path.join(cfg_dir, "argv.txt")
    with open(argv_path, "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")
    print(f"[config] Saved CLI args -> {argv_path}")


def autosave_scripts(files_to_backup, dest_dir, backup_suffix="_backup"):
    """
    Save the listed files into out/<unique_id>/code/<timestamp>/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_code_dir = os.path.join(dest_dir, "code", timestamp)
    os.makedirs(run_code_dir, exist_ok=True)

    for file_path in files_to_backup:
        # Resolve relative paths against the script's folder
        if not os.path.isabs(file_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(base_dir, file_path)
            if os.path.exists(candidate):
                abs_path = candidate
            else:
                # Fall back to trying to resolve via import (e.g., eraser.attn_processor)
                try:
                    module = __import__(file_path.replace(".py", "").replace("/", "."))
                    abs_path = inspect.getsourcefile(module)
                except Exception:
                    abs_path = os.path.abspath(file_path)
        else:
            abs_path = file_path

        if not os.path.exists(abs_path):
            print(f"[autosave] WARNING: {file_path} not found (resolved: {abs_path}). Skipping.")
            continue

        filename = os.path.basename(abs_path)
        name, ext = os.path.splitext(filename)
        backup_filename = f"{name}{backup_suffix}{ext}" if ext else f"{name}{backup_suffix}"
        backup_path = os.path.join(run_code_dir, backup_filename)
        
        shutil.copy2(abs_path, backup_path)  # copy2 preserves metadata
        print(f"[autosave] Saved {filename} -> {backup_path}")

def get_unique_id(cfg, time_str, concept=None):
        lr = cfg["learning_rate"]
        era = cfg["erasure_loss_scale"]
        red = cfg["redirection_loss_scale"]
        pre = cfg["preservation_loss_scale"]
        rec = cfg["reconstruction_loss_scale"]
        scl = cfg["init_key_scale"]
        seed = cfg["seed"]
        name = cfg["experiment_name"]
        nepochs = cfg["num_train_epochs"]
        unique_id = f'{time_str}'
        if name is not None:
            unique_id = f'{unique_id}_{name}'
        unique_id = f'{unique_id}_lr_{lr}_era_{era}_red_{red}_pre_{pre}_rec_{rec}_init_scale_{scl}_seed_{seed}_nepochs{nepochs}'
        if concept is not None:
            unique_id = os.path.join(concept, unique_id)

        return unique_id

def ensure_gpu(device: str):
    if "ilves" in socket.gethostname() and device.strip().endswith(":3"):
        raise RuntimeError("prep_data.py: device cuda:3 is disabled on host 'ilves'. Pick another GPU.")


def save_command(out_dir="out"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(out_dir, f'command_{timestamp}.txt')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    cmd = " ".join(sys.argv)
    with open(fname, "w") as f:
        f.write(cmd + "\n")
    print(f"Saved command to {fname}")



class Mode(Enum):
    TRAIN = "train"
    SAVE_MAP = "save_map"
    INFERENCE = "inference"
    INFERENCE_ORIGINAL = "inference_original"
    TRAIN_SKIP = "train_skip"
    TRAIN_NO_TARGET = "train_no_target"
    EVAL = "eval" # for evaluation with attention map saving