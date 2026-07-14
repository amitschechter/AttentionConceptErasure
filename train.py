import os
from utils import set_visible_gpus
import argparse

import socket
import gc
import random
import sys
import textwrap
from datetime import datetime
import pprint
import json


import cv2
import numpy as np
import pandas
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from torch import Generator
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention, FluxAttnProcessor2_0
from attn_processor import FluxAttentionEraser
from attn_utils import (
    combine_images,
    get_inner_dim,
    get_token_index,
    set_attention_processors,
    set_attention_processors_dit,
    get_token_indecies,
)
from data_loader import loader
from utils import (
    Mode,
    add_label,
    arrange_images_and_save,
    count_pipeline_parameters,
    freeze_flux_pipeline,
    get_best_gpus,
    autosave_scripts,
    save_run_config,
    get_unique_id,
    load_eval_prompts,
    attach_console_logger,
    slugify,
)

os.environ["WANDB_CONSOLE"] = "off"
import wandb

# Environment variable configurations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
DTYPE = torch.bfloat16
N_ATTN_LAYERS = 57


def set_seed(seed):
    """Sets the seed for reproducibility."""
    print(f"Setting seed: {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MinimalFluxTrainer:
    """A trainer for fine-tuning the FLUX model for concept erasure."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['main_device'] if torch.cuda.is_available() else "cpu")
        self.encoder_device = torch.device(self.config['second_device'] if torch.cuda.is_available() else "cpu")
        self.first_run = True # needs to first run with the target in prompt for initialization
        self.cache_base_img = {'erase': {}, 'preserve': {}}

        # Load pipeline and move components to their respective devices
        self.pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=DTYPE
        ).to(self.device)
        self.tokenizer_2 = self.pipeline.tokenizer_2

        # Move components to their respective devices
        self.pipeline.vae.to(self.encoder_device)
        self.pipeline.text_encoder.to(self.encoder_device)
        self.pipeline.text_encoder_2.to(self.encoder_device)

        print(f"VAE on: {next(self.pipeline.vae.parameters()).device}")
        print(f"Text Encoders on: {next(self.pipeline.text_encoder.parameters()).device}")
        print(f"Transformer on: {next(self.pipeline.transformer.parameters()).device}")

         # Initialize models, optimizer, and dataloader
        freeze_flux_pipeline(self.pipeline)
        self.all_processors = self.set_distinct_attention_processors()
        print(f"Number of attention processors: {len(self.all_processors)}")
        count_pipeline_parameters(self.pipeline)

        self.setup_optimizer()
        del self.config["data_config"]["img_dir_2"] # TODO: remove this line
        self.train_dataloader = loader(**self.config["data_config"])

    def setup_optimizer(self):
        """Sets up the AdamW optimizer for trainable parameters."""
        trainable_params = [p for p in self.pipeline.transformer.parameters() if p.requires_grad]
        print(f"Optimized params: {len(trainable_params)}")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.get("learning_rate", 1e-4),
            betas=(self.config.get("adam_beta1", 0.9), self.config.get("adam_beta2", 0.999),),
            weight_decay=self.config.get("adam_weight_decay", 0.01),
            eps=self.config.get("adam_epsilon", 1e-8),
        )

    def set_distinct_attention_processors(self):
        """Initializes and sets custom attention processors for the transformer."""
        all_processors = {}
        inner_dim = get_inner_dim(self.pipeline)
        attn_layer_ind = 0
        for name, module in self.pipeline.transformer.named_modules():
            if isinstance(module, Attention):
                processor = FluxAttentionEraser(
                    height=self.config["height"],
                    inner_dim=inner_dim,
                    layer_ind=attn_layer_ind,
                    scale_eraser_key_inference=self.config["scale_eraser_key_inference"],
                    config=self.config,
                ).to(self.device)

                all_processors[name] = processor
                module.processor = processor
                attn_layer_ind += 1
        return all_processors

    def set_layers_mode(self, mode, layers_to_skip=None):
        """Sets the operational mode for all attention processors."""
        for layer_ind, attn_processor in enumerate(self.all_processors.values()):
            if layers_to_skip is None or layer_ind not in layers_to_skip:
                attn_processor.mode = mode
            elif mode == Mode.TRAIN:
                attn_processor.mode = Mode.TRAIN_SKIP
        if mode == Mode.TRAIN and layers_to_skip:
            print(f"Set {len(layers_to_skip)} layers to skip training (for memory optimization): {layers_to_skip}")

    def train_step(self, batch):
        """Performs a single training step, calculating loss based on sample type."""
        img, prompts = batch
        prompts = list(prompts) if isinstance(prompts, tuple) else prompts
        assert isinstance(prompts[0], str), f"Prompt items must be strings, got {type(prompts[0])}"
        if self.first_run:
            print(f'\nFirst prompt (init): {prompts[0]}')
            print(f'Seed: {self.config["seed"]}')
            self.first_run = False
        batch_size = img.shape[0]

        # Prepare target token indices and context for attention processors
        target_token_idx_lst = []
        for prompt in prompts:
            idx = get_token_indecies(self.tokenizer_2, prompt, self.config["concept"])
            target_token_idx_lst.append(idx)
       
        token_ids = self.tokenizer_2(prompts[0])["input_ids"]
        all_tokens = self.tokenizer_2.convert_ids_to_tokens(token_ids)
        for attn_processor in self.all_processors.values():
            attn_processor.target_token_idx_lst = target_token_idx_lst
            attn_processor.all_tokens = all_tokens

        # --- Latent and Embedding Preparation ---
        with torch.no_grad():
            img = img.to(self.encoder_device, dtype=DTYPE)
            x_1 = self.pipeline.vae.encode(img).latent_dist.sample().to(self.device)
            
            img_ids = FluxPipeline._prepare_latent_image_ids(
                x_1.shape[0],
                x_1.shape[2] // 2,
                x_1.shape[3] // 2,
                self.device,
                DTYPE,
            )
            x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

            prompt_embeds, pooled_prompt_embeds, prompt_ids = self.pipeline.encode_prompt(
                prompt=prompts,
                prompt_2=prompts,
                device=self.encoder_device,
                num_images_per_prompt=1,
            )
            prompt_embeds = prompt_embeds.to(self.device, dtype=DTYPE)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.device, dtype=DTYPE)
            prompt_ids = prompt_ids.to(self.device)

        # --- Noise and Timestep Sampling ---
        # generator = Generator(device=self.device).manual_seed(self.config["seed"]) # TODO: remove
        # t = torch.sigmoid(torch.randn((batch_size,), generator=generator, device=self.device))
        t = torch.sigmoid(torch.randn((batch_size,), device=self.device))

        # generator = Generator(device=self.device).manual_seed(self.config["seed"]) # TODO: remove
        # x_0 = torch.randn(x_1.shape, device=self.device, dtype=x_1.dtype, generator=generator)
        x_0 = torch.randn(x_1.shape, device=self.device, dtype=x_1.dtype)

        x_t = (1 - t.view(-1, 1, 1)) * x_1 + t.view(-1, 1, 1) * x_0

        # --- Guidance ---
        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.tensor([self.config["guidance_scale"]], device=self.device)
            guidance = guidance.expand(x_1.shape[0])
        else:
            guidance = None

        # --- Forward Passes and Loss Calculation ---
        transformer_kwargs = {
            "hidden_states": x_t.to(DTYPE),
            "img_ids": img_ids.to(DTYPE),
            "encoder_hidden_states": prompt_embeds.to(DTYPE),
            "txt_ids": prompt_ids.to(DTYPE),
            "pooled_projections": pooled_prompt_embeds.to(DTYPE),
            "timestep": t.to(DTYPE),
            "guidance": guidance.to(DTYPE) if guidance is not None else None,
        }

        # First pass to save original attention maps
        with torch.no_grad():
            self.set_layers_mode(Mode.SAVE_MAP)
            self.pipeline.transformer(**transformer_kwargs)

        # Second pass for training
        self.set_layers_mode(Mode.TRAIN, layers_to_skip=random.sample(range(N_ATTN_LAYERS), self.config["n_layers_to_skip"]))
        output = self.pipeline.transformer(**transformer_kwargs)

        # --- Loss Aggregation ---
        loss = F.mse_loss(output.sample, (x_t - x_0).to(DTYPE), reduction="mean")
        reconstruction_loss = loss.detach().cpu()
        loss = loss * self.config["reconstruction_loss_scale"]
        
        redirection_loss_arr, erasure_loss_arr, preservation_loss_arr = [], [], []
        for attn_processor in self.all_processors.values():
            redirection_loss_arr.append(attn_processor.layer_loss["redirection_loss"])
            erasure_loss_arr.append(attn_processor.layer_loss["erasure_loss"])
            preservation_loss_arr.append(attn_processor.layer_loss["preservation_loss"])

        total_redirection_loss = torch.mean(torch.stack(redirection_loss_arr))
        total_erasure_loss = torch.mean(torch.stack(erasure_loss_arr))
        total_preservation_loss = torch.mean(torch.stack(preservation_loss_arr))
        
        loss += (
            self.config["redirection_loss_scale"] * total_redirection_loss
            + self.config["erasure_loss_scale"] * total_erasure_loss
            + self.config["preservation_loss_scale"] * total_preservation_loss
        )
        
        if self.config["debug_print"]:
            print(f"Redirection loss arr: {[l.item() for l in redirection_loss_arr]}")
            print(f"Erasure loss arr: {[l.item() for l in erasure_loss_arr]}")
            print(f"Preservation loss arr: {[l.item() for l in preservation_loss_arr]}")

        print(
            f"redirection_loss: {total_redirection_loss.item():.5f}, "
            f"erasure_loss: {total_erasure_loss.item():.5f}, "
            f"preservation_loss: {total_preservation_loss.item():.5f}, "
            f"reconstruction_loss: {reconstruction_loss.item():.5f}, "
            f"loss: {loss.item():.5f}"
        )

        return loss.to(self.device)

    def train(self):
        """Main training loop."""
        if self.config["seed"] is not None:
            set_seed(self.config["seed"])

        num_epochs = self.config.get("num_train_epochs", 10)
        max_train_steps = self.config.get("max_train_steps")
        save_steps = self.config.get("save_steps", 1000)

        ckpt_dir = os.path.join(self.config["out_dir"], "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)

        self.set_layers_mode(mode=Mode.TRAIN)
        self.pipeline.transformer.train()

        global_step, epoch = 0, 0
        while True:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0.0
            progress_bar = tqdm(
                self.train_dataloader, desc=f"Training Epoch {epoch + 1}"
            )

            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{epoch_loss / (step + 1):.4f}",
                    }
                )
                if global_step <= 1:
                    self.test(global_step)

                if global_step % save_steps == 0:
                    self.save_checkpoint(ckpt_dir, global_step)

                if max_train_steps and global_step >= max_train_steps:
                    print(f"Reached maximum training steps: {max_train_steps}")
                    self.save_checkpoint(ckpt_dir, global_step)
                    return

            if global_step % self.config["inference_every_n_steps"] == 0 or global_step == 1:
                self.test(global_step)
                self.set_layers_mode(Mode.TRAIN)

            print(
                f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(self.train_dataloader):.4f}"
            )
            epoch += 1

        self.save_checkpoint(ckpt_dir, global_step)
        print("Training completed!")

    def save_checkpoint(self, ckpt_dir, global_step):
        """Saves the model checkpoint."""
        checkpoint_path = os.path.join(ckpt_dir, f"checkpoint-{global_step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f"Saving checkpoints in: {checkpoint_path}")

        if self.config["save_full_model"]:
            torch.save(
                self.pipeline.transformer.state_dict(),
                os.path.join(checkpoint_path, "dit.bin"),
            )
            torch.save(
                self.optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.bin")
            )

        eraser_state = {
            layer_name: {
                "key_addition": proc.key_addition.detach().cpu(),
                "value_addition": proc.value_addition.detach().cpu(),
            }
            for layer_name, proc in self.all_processors.items()
            if isinstance(proc, FluxAttentionEraser)
        }
        
        lr = self.config["learning_rate"]
        key_val_out_file = os.path.join(checkpoint_path, f"eraser_kv_{lr}.pt")
        torch.save(eraser_state, key_val_out_file)
        print(
            f"Checkpoint saved at step {global_step}. Eraser state dict saved at: {key_val_out_file}"
        )

    def load_eraser_weights(self, eraser_kv_path):
        """Loads eraser key/value weights from a checkpoint."""
        if not os.path.exists(eraser_kv_path):
            print(f"No eraser key/value file found at: {eraser_kv_path}")
            return
        eraser_state = torch.load(eraser_kv_path, map_location=self.device)
        for layer_name, state in eraser_state.items():
            if layer_name in self.all_processors:
                processor = self.all_processors[layer_name]
                if isinstance(processor, FluxAttentionEraser):
                    processor.key_addition.data = state["key_addition"].to(self.device)
                    processor.value_addition.data = state["value_addition"].to(self.device)
        print("Loaded eraser key/value weights from checkpoint.")

    def generate_image(self, prompt, with_erasure=False):
        """Generates a single image with or without concept erasure."""
        self.set_layers_mode(Mode.INFERENCE if with_erasure else Mode.INFERENCE_ORIGINAL)

        with torch.no_grad():
            # Temporarily move encoders to main device for generation
            self.pipeline.text_encoder.to(self.device)
            self.pipeline.text_encoder_2.to(self.device)
            self.pipeline.vae.to(self.device)

            generator = Generator(device=self.device).manual_seed(self.config.get("eval_seed", 42))
            image = self.pipeline(
                prompt=prompt,
                prompt_2=prompt,
                num_inference_steps=self.config["num_inference_steps"],
                guidance_scale=self.config["guidance_scale"],
                height=self.config["data_config"]["img_size"],
                width=self.config["data_config"]["img_size"],
                generator=generator,
            ).images[0]

            # Move encoders back to the secondary device
            self.pipeline.text_encoder.to(self.encoder_device)
            self.pipeline.text_encoder_2.to(self.encoder_device)
            self.pipeline.vae.to(self.encoder_device)
        return image

    def generate_for_input_prompt(self):
        """Interactive loop to generate images from user input."""
        # if self.config["seed"] is not None:
        #     set_seed(self.config["seed"])
        self.set_layers_mode(Mode.INFERENCE)

        while True:
            prompt = input("Enter your prompt (or press Enter to quit): ")
            if not prompt:
                print("Empty prompt - exiting interactive mode.")
                return

            print(f"You typed: {prompt}")

            # Set tokens for all attention processors
            token_ids = self.tokenizer_2(prompt)['input_ids']
            all_tokens = self.tokenizer_2.convert_ids_to_tokens(token_ids)
            for attn_processor in self.all_processors.values():
                attn_processor.all_tokens = all_tokens

            print(f"Generating images for: '{prompt}'")

            if prompt in self.cache_base_img['preserve']:
                baseline_img = self.cache_base_img['preserve'][prompt]
            else:
                baseline_img = self.generate_image(prompt, with_erasure=False)
                self.cache_base_img['preserve'][prompt] = baseline_img
                print(self.cache_base_img['preserve'])
                

            erased_img = self.generate_image(prompt, with_erasure=True)

            baseline_labeled = add_label(baseline_img, f"Prompt: {prompt[:50]}... | No Erasure")
            erased_labeled = add_label(
                erased_img,
                f"Prompt: {prompt[:50]}... | With Erasure, scale: {self.config['scale_eraser_key_inference']}",
            )
            
            cell = combine_images([baseline_labeled, erased_labeled])
            
            safe_prompt = prompt[:100].replace(" ", "_").replace("/", "_")
            img_name = os.path.join(self.config["out_dir"], f"img_{safe_prompt}_{self.config['unique_id']}.jpeg")
            
            os.makedirs(os.path.dirname(img_name), exist_ok=True)
            cell.save(img_name, format="JPEG")
            print(f"Saved image in {img_name}")

    def test_for_prompt_lst(self, test_prompts, image_prefix, image_suffix="", step=None):
        """Generates and saves a grid of images for a list of prompts."""
        labeled_cells = []

        for prompt in test_prompts:
            token_ids = self.tokenizer_2(prompt)['input_ids']
            all_tokens = self.tokenizer_2.convert_ids_to_tokens(token_ids)
            for attn_processor in self.all_processors.values():
                attn_processor.all_tokens = all_tokens
                attn_processor.step = step
                
            print(f"Generating images for: '{prompt}'")
            if prompt in self.cache_base_img['preserve']:
                baseline_img = self.cache_base_img['preserve'][prompt]
            else:
                baseline_img = self.generate_image(prompt, with_erasure=False)
                self.cache_base_img['preserve'][prompt] = baseline_img

            erased_img = self.generate_image(prompt, with_erasure=True)

            baseline_labeled = add_label(baseline_img, f"Prompt: {prompt[:50]}... | No Erasure")
            erased_labeled = add_label(
                erased_img,
                f"Prompt: {prompt[:50]}... | With Erasure, scale: {self.config['scale_eraser_key_inference']}",
            )

            cell = combine_images([baseline_labeled, erased_labeled])
            labeled_cells.append(cell)
        
        img_name_parts = [
            image_prefix,
            self.config['unique_id'],
            image_suffix,
            str(step) if step is not None else None,
        ]
        img_filename = "_".join(filter(None, img_name_parts)) + ".jpeg"
        
        # out_dir = os.path.join('out', self.config["unique_id"])
        img_path = os.path.join(self.config["out_dir"], img_filename)

        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        arrange_images_and_save(labeled_cells, img_path, format="JPEG")

        print("\n--- Results ---")
        print("Left: Baseline generations | Right: Generations with concept erasure")
        print(f"Saved comparison to '{img_path}'")

    def test(self, step=None):
        """Runs a standard test with predefined prompts."""
        # if self.config["seed"] is not None:
        #     set_seed(self.config["seed"])
        self.set_layers_mode(Mode.INFERENCE)

        erase_test_prompts, preserve_test_prompts = load_eval_prompts(self.config["eval_prompts_path"], slugify(self.config["concept"]))
        print(f'Evaluating for {len(erase_test_prompts)} erase prompts and {len(preserve_test_prompts)} preserve prompts')

        self.test_for_prompt_lst(erase_test_prompts, image_prefix='results', image_suffix='erase', step=step)
        self.test_for_prompt_lst(preserve_test_prompts, image_prefix='results', image_suffix='preserve', step=step)

        # if step is not None and step % 50 == 0:
        #     self.generate_for_input_prompt()


def main():
    parser = argparse.ArgumentParser()
    # Add arguments for each of the config parameters you want to pass via the command line
    parser.add_argument("--learning_rate", type=float, nargs='+', default=[0.003], help="One or more learning rates. Example: --learning_rate 1e-4 3e-4")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Number of training epochs. Default is 1000.")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Max number of training steps. Default is None.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Steps to save checkpoints. Default is 1000.")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale for generation. Default is 3.5.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps for text. Default is 20.")
    parser.add_argument("--init_key_scale", type=float, default=1.5, help="Scaling factor for key during init")
    
    parser.add_argument("--erasure_loss_scale", type=float, default=1.0, help="Scale for erasure loss. Default is 1.0.")
    parser.add_argument("--redirection_loss_scale", type=float, default=0.0, help="Scale for redirection loss. Default is 1.0.")
    parser.add_argument("--preservation_loss_scale", type=float, default=1.0, help="Scale for preservation loss. Default is 1.0.")
    parser.add_argument("--reconstruction_loss_scale", type=float, default=1.0, help="Scale for reconstruction loss. Default is 1.0.")
    
    parser.add_argument("--concept", type=str, required=True, help="Concept to erase e.g. spiderman")

    parser.add_argument("--metadate_path", type=str, default='data/metadata.json', help="Path to json file mapping concepts to their info")

    ####### ------------------- optional, if not passed- extracted from metadata ----------------------- ######
    parser.add_argument("--img_dir", type=str, default='data/train_data/Spiderman/Spiderman_data', help="Path to the image directory for prompts and imgs of the target concept.")
    parser.add_argument("--img_dir_2", type=str, default='data/train_data/Spiderman/no_Spiderman_data', help="Path to the image directory for prompts and imgs of other concepts.")
    parser.add_argument("--target_val_path", type=str, default='data/val_files/Spiderman_val', help="Path to the image directory for prompts and imgs of other concepts.")
    parser.add_argument("--neutral_val_path", type=str, default='data/val_files/man_val', help="Path to the image directory for prompts and imgs of other concepts.")
    parser.add_argument("--eval_prompts_path", type=str, default='data/train_data/train_eval_prompts.json', help="Path to json file mapping concepts to evaluation prompts")
    parser.add_argument("--load_existing_keys", type=str, default=None, help="Path to keys directory. Default None - the key is taken from the first target example")
    
    parser.add_argument("--main_device", type=str, default='cuda:0')
    parser.add_argument("--second_device", type=str, default=None)

    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training. Default 1.")
    parser.add_argument("--experiment_name", type=str, default='.', help="Experiment name")
    parser.add_argument("--scale_eraser_key_inference", type=float, default=1.0, help="Scale for eraser key. Default is 1.0.")
    parser.add_argument("--prompt_every_n", type=int, default=None, help="Number of steps before it asks for input prompts for inference")
    parser.add_argument("--inference_every_n_steps", type=int, default=40, help="Number of training epochs running inference")
    parser.add_argument("--save_full_model", action='store_true', help="Whether to save the full model. Default is False.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default is 42.")
    parser.add_argument("--eval_seed", type=int, default=42, help="Deterministic eval seed.")

    parser.add_argument("--n_layers_to_skip", type=int, default=6, help="Number of layers to skip for memory saving")

    parser.add_argument("--debug_print", action='store_true', help="Whether to print debug information. Default is False.")
    parser.add_argument("--visualize_attention", action='store_true', help="Whether to save attention maps.")
    parser.add_argument("--save_keys_in_inference", action='store_false', help="Whether to save keys in inference.")

    parser.add_argument("--optional_dir", type=str, default=None)
    parser.add_argument("--save_keys_every_n_steps", type=int, default=10)


    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments to populate the config dictionary
    config = {
        'model_name': 'flux-dev',  # or 'flux-schnell'
        'learning_rate': args.learning_rate,
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_weight_decay': 0.01,
        'adam_epsilon': 1e-8,
        'num_train_epochs': args.num_train_epochs,
        'max_train_steps': args.max_train_steps,
        'save_steps': args.save_steps,
        'guidance_scale': args.guidance_scale,
        'num_inference_steps': args.num_inference_steps,
        'prompt_every_n': args.prompt_every_n,
        'inference_every_n_steps': args.inference_every_n_steps,
        'init_key_scale': args.init_key_scale,
        
        "erasure_loss_scale": args.erasure_loss_scale,
        "redirection_loss_scale": args.redirection_loss_scale,
        "preservation_loss_scale": args.preservation_loss_scale,
        "reconstruction_loss_scale": args.reconstruction_loss_scale,

        "save_full_model": args.save_full_model,
        "seed": args.seed,
        "eval_seed": args.eval_seed,

        'data_config': {
            'train_batch_size': args.train_batch_size,
            'num_workers': 1,
            'img_dir': args.img_dir,
            'img_dir_2': args.img_dir_2,
            'img_size': 512,
        },

        'concept': args.concept,
        'target_val_path': args.target_val_path,
        'neutral_val_path': args.neutral_val_path,
        'eval_prompts_path': args.eval_prompts_path,
        'load_existing_keys': args.load_existing_keys,

        'optional_dir': args.optional_dir,
        'res_out_dir': './result_gen_out',
        'experiment_name': args.experiment_name,
        'scale_eraser_key_inference': args.scale_eraser_key_inference,
        'debug_print': args.debug_print,
        'visualize_attention': args.visualize_attention,
        'save_keys_every_n_steps': args.save_keys_every_n_steps,
        'save_keys_in_inference': args.save_keys_in_inference,
        'n_layers_to_skip': args.n_layers_to_skip,
        'metadate_path': args.metadate_path,

        'main_device': args.main_device,
        'second_device': args.second_device,
    }

    metadata_dict = json.loads(Path(args.metadate_path).read_text())
    # joint_args = meta.get("joint", {})
    # get_info_from_metadata(slugify(config["concept"]))
    assert os.path.isdir(config['data_config']['img_dir']), f"{config['data_config']['img_dir']} not found"
    assert os.path.isdir(config['data_config']['img_dir_2'])
    assert config["redirection_loss_scale"] == 0, "Change the attention processor to use non-zero rediorection loss"

    assert config["data_config"]["train_batch_size"] == 1, f'Batch size > 1: {config["data_config"]["train_batch_size"]}, need to adjust code'
    config["height"] = config["data_config"]["img_size"] // 16
    print(f"Image Size: {config['data_config']['img_size']}, Latent Height: {config['height']}")

    if 'ilves' in socket.gethostname():
        assert "3" not in config["main_device"] and "3" not in config["second_device"]
        assert config["main_device"] != "cuda:3" and config["second_device"] != "cuda:3"

    if config["second_device"] is None:
        config["second_device"] = config["main_device"]

    time_str = datetime.now().strftime("%m-%d_%H-%M-%S")

    # Train
    if len(config["learning_rate"]) > 0: # sweep
        sweep_learning_rates = config["learning_rate"] #[1e-7, 3e-7, 1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3]

        for learning_rate in sweep_learning_rates:
            # update the config with the learning rate etc.
            config["learning_rate"] = learning_rate
            config["unique_id"] = get_unique_id(config, time_str)
            pprint.pprint(config, indent=4)

            if config["optional_dir"]:
                run_out_dir = os.path.join('out', config["optional_dir"], config['concept'], config["unique_id"])    
            else:
                run_out_dir = os.path.join('out', config['concept'], config["unique_id"])
            config["out_dir"] = run_out_dir

            # save code & config for reproducability
            autosave_scripts(
                files_to_backup=[
                    __file__,            # the script you’re running
                    "attn_processor.py", # sibling file; if not present, warning is printed
                    "data_loader.py",
                    "attn_utils.py",
                    "utils.py",
                ],
                dest_dir=run_out_dir
            )
            save_run_config(config, run_out_dir)
            attach_console_logger(os.path.join(run_out_dir, "console.log"))

            # run the training & evaluation code
            trainer = MinimalFluxTrainer(config)
            trainer.train()
            trainer.test()

            # delete and free up space, the next run starts clean
            del trainer
            torch.cuda.empty_cache()
            gc.collect()



if __name__ == "__main__":
    main()