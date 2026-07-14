import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention, FluxAttnProcessor2_0
from diffusers.models.embeddings import apply_rotary_emb
from tqdm import tqdm
import collections
from PIL import Image
from typing import Optional, Dict, Any
import numpy as np
from einops import rearrange
import math

# from unused_files.flux_pipeline_custom import FluxPipelineWithGrad
from attn_utils import get_token_index, combine_images, scaled_dot_product_attention_with_weights, save_side_by_side_attention_map, reshape_value
from utils import Mode

def preprocess_attn_q_k_v(attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        image_rotary_emb: Optional[torch.Tensor] = None
        ):
    batch_size, sequence_length, inner_dim = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    head_dim = inner_dim // attn.heads
    encoder_output_length = None
    
    # Sample projections
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    
    assert inner_dim == key.shape[-1] # TODO: remove once checked

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # Context projections
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
        
        encoder_output_length = encoder_hidden_states_query_proj.shape[2]

        # Concatenate for joint attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    # Apply rotary embeddings if provided
    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)
    
    return query, key, value, batch_size, head_dim, hidden_states, encoder_output_length


class FluxAttentionEraserWithScale(nn.Module):
    """Extracts cross attention maps for a specific token from Flux model with trainable scale parameter."""
    
    def __init__(self, height: int = 64, inner_dim=None, dtype=torch.bfloat16, layer_ind: int = 0, scale_eraser_key_inference=1.0, train_scale=False, shared_key_scale=None, mode=Mode.TRAIN, config=None):
        super().__init__()

        # static variables - not changing
        self.text_length = 512 # TODO: extract from input
        self.height = height
        self.inner_dim = inner_dim
        self.layer_ind = layer_ind
        self.scale_eraser_key_inference = scale_eraser_key_inference
        self.train_scale = train_scale
        self.shared_key_scale = shared_key_scale
        self.config = config
        self.step = 0

        # state indicators
        self.mode = mode
        self.key_initialized = False 

        # counters and changing variables or ones that are initialized later
        self.time_step = 0
        self.all_tokens = [] # list of tokens (workds) for a sentence, initialized from outside
        self.target_token_idx_lst = None # list of size batch_size, where each item is a list of size num_tokens
        self.original_map = None # original attention map, saved/updated on save only runs
        self.original_eraser_map = None # original attention map for the eraser token (for vis/debugging)
        
        # key and value tensors (key is trainable and initialized on the first call using key_initialized)
        self.key_addition = nn.Parameter(torch.zeros(1, 1, self.inner_dim, dtype=dtype))
        self.get_shared_key_scale = None
        # Use shared scale parameter if provided, otherwise create individual one
        # if self.shared_key_scale is not None:
        #     self.scale_eraser_key = self.shared_key_scale
        # elif self.train_scale:
        #     self.scale_eraser_key = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        #     print(f"Initialized individual trainable scale parameter for layer {layer_ind}")
        # else:
        #     self.register_buffer('scale_eraser_key', torch.tensor(1.0, dtype=torch.float32))
            
        self.register_buffer('value_addition', torch.randn(1, 1, self.inner_dim, dtype=dtype) * 0.01) # Not trainable, currently random

        target_val_path, neutral_val_path = self.config['target_val_path'], self.config['neutral_val_path']

        # load neutral and target values
        if neutral_val_path is not None:
            neutral_val_path = os.path.join(neutral_val_path, f'{layer_ind}.pt')
            neutral_val = torch.load(neutral_val_path, map_location=self.value_addition.device)
            self.neutral_val = reshape_value(neutral_val)
            # self.neutral_val = torch.load(neutral_path).view(1, 1, -1)
            assert self.neutral_val.shape == self.value_addition.shape, f"neutral_val: {self.neutral_val.shape} != {self.value_addition.shape}"
        else:
            print(f'Neutral val not passed. No value to load.')
    
        if target_val_path is not None:
            target_path = os.path.join(target_val_path, f'{layer_ind}.pt')
            target_val = torch.load(target_path, map_location=self.value_addition.device)
            self.target_val = reshape_value(target_val)
            assert self.target_val.shape == self.value_addition.shape, f"target val: {self.target_val.shape} != {self.value_addition.shape}"
        else:
            print(f'Target val not passed. No value to load.')

        # self.neutral_val = torch.load(os.path.join(neutral_val_path, f'{self.layer_ind}.pt'), map_location=self.value_addition.device).view(1, 1, -1)
        # self.target_val = (torch.load(os.path.join(target_val_path, f'{self.layer_ind}.pt'), map_location=self.value_addition.device).mean(dim=2)).view(1, 1, -1)
    
    @torch.enable_grad()
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        height: int = None,
        **kwargs
    ) -> torch.FloatTensor:

        query, key, value, batch_size, head_dim, hidden_states, encoder_output_length = preprocess_attn_q_k_v(
            attn, hidden_states, encoder_hidden_states, image_rotary_emb)


        # self.man_val = torch.load(os.path.join('man_val', f'{self.layer_ind}.pt'), map_location=self.value_addition.device).view(1, 1, -1)
        # self.spiderman_val = (torch.load(os.path.join('spiderman_val', f'{self.layer_ind}.pt'), map_location=self.value_addition.device).mean(dim=2)).reshape_as(self.value_addition) # avg across target (e.g. Spider-Man) tokens
        # torch.save(eraser_value, os.path.join('spiderman_val', f'{self.layer_ind}.pt'))
        
        if self.mode in {Mode.TRAIN, Mode.TRAIN_SKIP, Mode.TRAIN_NO_TARGET}:
            self.time_step = self.time_step + 1
        
        if self.mode in {Mode.TRAIN, Mode.TRAIN_SKIP, Mode.SAVE_MAP}:
            if not self.key_initialized: # on first run - iniitialize the key from the target token
                if self.config["load_existing_keys"] is not None: # load keys from file
                    keys_path = os.path.join(self.config["load_existing_keys"], f'{self.layer_ind}.pt')
                    loaded_key = torch.load(keys_path, map_location=self.key_addition.device)
                    reshaped_token_key = loaded_key.reshape_as(self.key_addition)
                    with torch.no_grad():
                        self.key_addition.data.copy_(reshaped_token_key.to(self.key_addition.dtype))

                else: # initialize keys as the first example's target token
                    self._initialize_key_addition(key, self.target_token_idx_lst[0], scale=self.config["init_key_scale"]) # initialize the key_addition from the key of the target token (first example in the batch)
                self.key_initialized = True
                
                # with torch.no_grad():
                #     self.value_addition.copy_(eraser_val_tensor.view(1, 1, -1).to(self.value_addition.dtype))

        if self.mode in {Mode.TRAIN}: 
            assert len(self.target_token_idx_lst) == batch_size # in inference we don't have target_token_idx_lst
            if len(self.target_token_idx_lst) > 1:
                print(f"#########Note: the batch size {batch_size} >1. Adjust FluxAttentionEraserWithScale call func")
            
        self.layer_loss = {'erasure_loss': torch.tensor(0.0, device=key.device, dtype=key.dtype),
                           'redirection_loss': torch.tensor(0.0, device=key.device, dtype=key.dtype),
                           'preservation_loss': torch.tensor(0.0, device=key.device, dtype=key.dtype)}

        # inference without additional key/val. used for inference without the eraser (for baseline)
        # and in training for randomly selected layers, don't add eraser (memory optimization)
        if self.mode in {Mode.INFERENCE_ORIGINAL}:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
    
        # run the attention without the added key and save the original attention maps
        elif self.mode == Mode.SAVE_MAP:
            hidden_states, attention_probs = scaled_dot_product_attention_with_weights(
                query, key, value, attention_mask, dropout_p=0.0, is_causal=False, calculate_original=False, eraser_dims=0)
            original_img_to_text_attention = attention_probs[:, :, self.text_length:, :self.text_length] # Shape [1, 24, 1024, 512]
            if self.target_token_idx_lst and len(self.target_token_idx_lst) == batch_size:
                original_map_target_token = original_img_to_text_attention[:,:,:,self.target_token_idx_lst[0]]
                self.original_map = original_map_target_token.detach().cpu()
            else:
                self.original_map = None
            
            self.original_img_to_text_attention = original_img_to_text_attention.detach().cpu()
            # eraser_value = (value[:,:,self.target_token_idx_lst[0],:])#.mean(dim=2)).reshape_as(self.value_addition) # avg across target (e.g. Spider-Man) tokens
            # torch.save(eraser_value, os.path.join('spiderman_val', f'{self.layer_ind}.pt'))
        
        # - INJECT ERASER KEY/VALUE RIGHT BEFORE ATTENTION, only in train or inference mode -
        elif self.mode in {Mode.TRAIN, Mode.INFERENCE, Mode.TRAIN_SKIP, Mode.TRAIN_NO_TARGET, Mode.EVAL}:

            # For inference - set eraser value to man
            if self.mode in {Mode.INFERENCE, Mode.EVAL}:
                with torch.no_grad():
                    self.value_addition.data.copy_(self.neutral_val.to(device=self.value_addition.device, dtype=self.value_addition.dtype))
            # For training - set eraser value to target token value
            else:
                with torch.no_grad():
                    self.value_addition.data.copy_(self.target_val.to(device=self.value_addition.device, dtype=self.value_addition.dtype))
                # eraser_value = (value[:,:,self.target_token_idx_lst[0],:].mean(dim=2)).reshape_as(self.key_addition) # avg across target (e.g. Spider-Man) tokens
                # torch.save(eraser_value, os.path.join('spiderman_val', f'{self.layer_ind}.pt'))
                
            p = self.get_shared_key_scale() if callable(getattr(self, "get_shared_key_scale", None)) else None
            key_scale = p.to(dtype=key.dtype, device=key.device) if p is not None else key.new_tensor(1.0)
            scaled_key_addition = self.key_addition * key_scale
            # Expand the learnable K/V pair for the current batch size and heads Shape: (batch_size, 1, inner_dim) -> (batch_size, heads, 1, head_dim)
            eraser_key_batch = scaled_key_addition.expand(batch_size, -1, -1)
            eraser_value_batch = self.value_addition.expand(batch_size, -1, -1)
                
            # Reshape eraser tensors for multi-head attention
            eraser_key_batch = eraser_key_batch.view(batch_size, 1, attn.heads, head_dim).transpose(1, 2)
            eraser_value_batch = eraser_value_batch.view(batch_size, 1, attn.heads, head_dim).transpose(1, 2)
            
            # # Concatenate eraser key/value to the sequence (at the beginning). New order: [eraser, text, image]
            e_key_val_len_ = eraser_key_batch.shape[2]

            # Apply trainable scale parameter
            # if self.train_scale:
            #     # Use the trainable scale parameter
            #     scale_factor = self.scale_eraser_key
            # else:
            #     # Use the inference scale parameter
            #     scale_factor = self.scale_eraser_key_inference
            
            # during inference we might increase/decrease the eraser key scale
            # if (scale_factor is not None) and (self.mode in {Mode.INFERENCE, Mode.EVAL}):
            #     eraser_key_batch = eraser_key_batch * scale_factor
            
            key = torch.cat([eraser_key_batch, key], dim=2)
            value = torch.cat([eraser_value_batch, value], dim=2)
            
            if self.mode in {Mode.INFERENCE, Mode.TRAIN_SKIP, Mode.TRAIN_NO_TARGET}:
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                if self.mode == Mode.INFERENCE and self.config is not None and self.config["save_keys_in_inference"]:
                    ckpt_path = os.path.join(self.config["out_dir"], 'ckpt', 'keys', 'inference', f'step_{self.step}', f'layer_{self.layer_ind}_key.pt')
                    if not os.path.isfile(ckpt_path): # during inference only save the first
                        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                        # Save the scaled key
                        # scaled_key = self.key_addition * scale_factor
                        torch.save(scaled_key_addition.detach().cpu(), ckpt_path)
            
            elif self.mode == Mode.TRAIN:
                # # Extract the attention weights and attention score (hidden states) both with and without the new key/value tokens
                hidden_states, attention_probs, orig_hidden_states, orig_attention_probs = scaled_dot_product_attention_with_weights(
                    query, key, value, attention_mask, dropout_p=0.0, is_causal=False, calculate_original=True, eraser_dims=e_key_val_len_)

                # # Extract attention from image tokens to text tokens
                # # (1,24,4608,4608) -> (1,24,4096,512)
                extended_text_length = self.text_length + 1
                extended_image_to_text_attention = attention_probs[:, :, self.text_length:, :extended_text_length] # Shape [1, 24, 1024, 513]
                original_img_to_text_attention = orig_attention_probs[:, :, self.text_length:, :self.text_length] # Shape [1, 24, 1024, 512]

                curr_target_token_idx_lst = self.target_token_idx_lst[0]
                curr_extended_target_token_idx_lst = [curr_token_idx + 1 for curr_token_idx in curr_target_token_idx_lst]
                original_map_target_token = original_img_to_text_attention[:,:,:,curr_target_token_idx_lst]
                extended_map_target_token = extended_image_to_text_attention[:,:,:,curr_extended_target_token_idx_lst]
                extended_map_eraser_token = extended_image_to_text_attention[:,:,:,0] # first token - eraser

                if self.original_eraser_map is None: # save the first eraser map for vis
                    self.original_eraser_map = extended_map_eraser_token.detach().cpu()
                    self.const_scale = self.original_eraser_map.norm()
                
                # calculations for preservation loss - only using prompt tokens
                # n_tokens = len(self.all_tokens)
                # # Include only the first n_tokens excluding the target & eraser tokens
                # include_ind_orig = sorted(set(range(n_tokens)) - set(curr_target_token_idx_lst))
                # include_ind_extend = sorted(set(range(n_tokens + 1)) - set([0] + [idx + 1 for idx in curr_target_token_idx_lst]))
                # mask_orig = torch.zeros(self.original_img_to_text_attention.shape[-1], dtype=torch.bool)
                # mask_extend = torch.zeros(extended_image_to_text_attention.shape[-1], dtype=torch.bool)
                # mask_orig[include_ind_orig] = True
                # mask_extend[include_ind_extend] = True
                # original_img_to_text_attention_filtered = self.original_img_to_text_attention[..., mask_orig].to(extended_image_to_text_attention.device)
                # extended_image_to_text_attention_filtered = extended_image_to_text_attention[..., mask_extend]

                # calculations for preservation loss - using all tokens but target/eraser
                exclude_ind_orig, exclude_ind_extend = curr_target_token_idx_lst, [0] + [idx + 1 for idx in curr_target_token_idx_lst]
                mask_orig = torch.ones(self.original_img_to_text_attention.shape[-1], dtype=torch.bool)
                mask_extend = torch.ones(extended_image_to_text_attention.shape[-1], dtype=torch.bool)
                mask_orig[exclude_ind_orig] = False
                mask_extend[exclude_ind_extend] = False
                original_img_to_text_attention_filtered = self.original_img_to_text_attention[..., mask_orig].to(extended_image_to_text_attention.device)
                extended_image_to_text_attention_filtered = extended_image_to_text_attention[..., mask_extend]

                # eraser_map_shape_loss = (1 - F.cosine_similarity(extended_map_eraser_token, original_map_target_token.sum(dim=3), dim=-1).mean())
                # eraser_map_mag_loss = (1 - F.tanh(extended_map_eraser_token.norm() / self.const_scale))
                # if self.time_step <= 1:
                #     print(eraser_map_shape_loss, eraser_map_mag_loss, extended_map_eraser_token.norm(), self.const_scale)

                # redirection_loss =  F.mse_loss(extended_map_eraser_token, original_map_target_token.sum(dim=3), reduction="mean"),
                self.layer_loss = {
                    'erasure_loss': extended_map_target_token.mean(),
                    'redirection_loss':  torch.tensor(0.0, device=key.device, dtype=key.dtype), #eraser_map_shape_loss + eraser_map_mag_loss,
                    'preservation_loss': F.mse_loss(extended_image_to_text_attention_filtered, original_img_to_text_attention_filtered, reduction="mean")
                }
                if ((self.time_step - 1) % 7 == 0) and (self.config["visualize_attention"]):
                    if self.layer_ind == 0:
                        print(f'Saving attention maps')
                    if (self.layer_ind % 2 == 0) or (self.layer_ind < 5 or self.layer_ind > 52): #(self.layer_ind % 5 == 0) and 
                        folder = os.path.join(self.config["out_dir"], 'attention_maps', f'layer_{self.layer_ind}')
                        os.makedirs(folder, exist_ok=True)

                        save_side_by_side_attention_map(self.original_map, extended_map_target_token,
                            self.original_eraser_map ,extended_map_eraser_token,
                            extended_image_to_text_attention, self.original_img_to_text_attention,
                            self.all_tokens, curr_target_token_idx_lst,
                            out_path=os.path.join(folder, f"t_{self.time_step}_l_{self.layer_ind}_loss_e_{self.layer_loss['erasure_loss']:.6f}_r_{self.layer_loss['redirection_loss']:.6f}_p_{self.layer_loss['preservation_loss']}_attn.jpeg"),
                            head_index=-1,  # average across heads
                            height=self.height,
                            layer_ind=self.layer_ind,
                            cmap="turbo",          # or "inferno"/"viridis"
                            color_mode="heatmap",
                            tile_px=512,           # nice for papers
                            blur_px=1,
                        )
                if ((self.time_step - 1) % self.config["save_keys_every_n_steps"] == 0):
                    if self.layer_ind == 0:
                        print(f'Saving keys')
                    ckpt_path = os.path.join(self.config["out_dir"], 'ckpt', 'keys', f't_{self.time_step}', f'layer_{self.layer_ind}_key.pt')      
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    # Save the scaled key
                    # scaled_key = self.key_addition * scale_factor
                    torch.save(scaled_key_addition.detach().cpu(), ckpt_path)

                # if ((self.time_step - 1) % 5 == 0) and (self.layer_ind % 2 == 0) or (self.layer_ind < 5 or self.layer_ind > 52): #(self.layer_ind % 5 == 0) and 
                #         folder = os.path.join('out', self.unique_id, f'layer_{self.layer_ind}')
                #         os.makedirs(folder, exist_ok=True)
                #         save_side_by_side_attention_map(self.original_map, extended_map_target_token,
                #             self.original_eraser_map ,extended_map_eraser_token,
                #             extended_image_to_text_attention, self.original_img_to_text_attention,
                #             self.all_tokens, curr_target_token_idx_lst,
                #             out_path=os.path.join(folder, f"t_{self.time_step}_l_{self.layer_ind}_loss_e_{self.layer_loss['erasure_loss']:.3f}_r_{self.layer_loss['redirection_loss']:.3f}_attn.jpeg"),
                #             head_index=-1,  # average across heads
                #             height=self.height,
                #             layer_ind=self.layer_ind,
                #         )
            elif self.mode in {Mode.EVAL}:
                # Must compute with weights, like TRAIN, but without grads/loss
                hidden_states, attention_probs, orig_hidden_states, orig_attention_probs = scaled_dot_product_attention_with_weights(
                    query, key, value, attention_mask, dropout_p=0.0, is_causal=False, calculate_original=True, eraser_dims=e_key_val_len_
                )
                # Slice same tensors as in TRAIN:
                extended_text_length = self.text_length + 1
                extended_image_to_text_attention = attention_probs[:, :, self.text_length:, :extended_text_length]
                original_img_to_text_attention = orig_attention_probs[:, :, self.text_length:, :self.text_length]

                # Build the same views: original_map_target_token, extended_map_target_token, extended_map_eraser_token
                curr_target_token_idx_lst = self.target_token_idx_lst[0]
                curr_extended_target_token_idx_lst = [i + 1 for i in curr_target_token_idx_lst]
                original_map_target_token = original_img_to_text_attention[:,:,:,curr_target_token_idx_lst]
                extended_map_target_token = extended_image_to_text_attention[:,:,:,curr_extended_target_token_idx_lst]
                extended_map_eraser_token = extended_image_to_text_attention[:,:,:,0]

                if self.original_eraser_map is None: # save the first eraser map for vis
                    self.original_eraser_map = extended_map_eraser_token.detach().cpu()
                    self.const_scale = self.original_eraser_map.norm()

                # SAVE (eval variant)
                if self.inference_step % 3 == 0:
                    folder = os.path.join(self.eval_base_dir, f'layer_{self.layer_ind}')
                    os.makedirs(folder, exist_ok=True)
                    save_side_by_side_attention_map(
                        original_target_map=None,
                        extended_target_map=None,
                        original_eraser_map=self.original_eraser_map,
                        eraser_map=extended_map_eraser_token,
                        extended_image_to_text_attention=extended_image_to_text_attention,
                        original_img_to_text_attention=self.original_img_to_text_attention,
                        all_tokens=self.all_tokens,
                        target_token_idx_lst=None,
                        out_path=os.path.join(folder, f"t_{self.inference_step}_l_{self.layer_ind}_eval.jpeg"),
                        head_index=-1,
                        height=self.height,
                        layer_ind=self.layer_ind
                    )
                self.inference_step += 1

        # Reshape output
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)

        # Handle encoder outputs
        if encoder_hidden_states is not None:
            encoder_hidden_states_out, hidden_states_out = (
                hidden_states[:, :encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )

            # Apply output projections
            hidden_states = attn.to_out[0](hidden_states_out)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states_out)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
    
    def _initialize_key_addition(self, key: torch.Tensor, target_idx: int, scale=1.0):
        """Initialize key_addition with the target token's key values (detached)."""
        batch_size, num_heads, seq_len, head_dim = key.shape
        token_key = key[:, :, target_idx, :]  # shape: [B, H, N_TOKENS, D]
        if token_key.ndim == 4:
            token_key = token_key.mean(dim=2) * scale # shape: [B, H, N_TOKENS, D] -> shape: [B, H, D] # TODO: maybe mean
        reshaped_token_key = token_key.reshape_as(self.key_addition) # [1, 1, H*D] = [1, 1, inner_dim]
        
        # Replace the dummy parameter data with the target token values
        with torch.no_grad():
            self.key_addition.data.copy_(reshaped_token_key.to(self.key_addition.dtype))
        
