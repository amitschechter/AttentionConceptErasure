# ARCE: Attention Redirection for Concept Erasure

Official implementation of **"Concept Erasure via Attention Redirection"** (CVPR Findings 2026).

[[Paper]](https://openaccess.thecvf.com/content/CVPR2026F/papers/Schechter_Concept_Erasure_via_Attention_Redirection_CVPRF_2026_paper.pdf) · [[Code]](https://github.com/amitschechter/AttentionConceptErasure)

---

## Overview

ARCE removes a target concept (e.g. a character, object, or style) from a pretrained text-to-image diffusion model without retraining the whole network and without materially degrading the model's ability to generate everything else.

Instead of editing existing weights, ARCE **injects a single learnable "eraser" key/value pair into every cross-attention layer** of the model. This eraser token acts as a diversion: attention that a target-concept token would normally receive is redirected toward the eraser token instead, while the eraser's *value* is set to a neutral / non-target representation. The result is that prompting the model with the target concept (or close paraphrases of it) no longer reproduces that concept in the generated image, while prompts unrelated to the concept are left largely untouched.

The current implementation targets **FLUX.1-dev** (via 🤗 `diffusers`), and the erasure mechanism is implemented as a custom `AttnProcessor` that wraps every attention block in the transformer.

### Key ideas

- **Learnable eraser key**: a single key vector per attention layer, initialized from the key of the target concept's token (e.g. the token for "Spider-Man") and refined during training.
- **Neutral eraser value**: the value associated with the eraser key is drawn from a neutral/anchor concept (e.g. "man") rather than the target concept, so that attention diverted to the eraser token contributes neutral content instead of the erased concept.
- **Three complementary training losses**, computed directly on attention maps and combined with the standard flow-matching reconstruction loss:
  - **Erasure loss** — pushes attention mass away from the target concept's token.
  - **Preservation loss** — keeps the attention distribution over all *other* tokens close to the original (unedited) model's attention, to minimize collateral changes to unrelated content.
  - **Redirection loss** — shapes the attention captured by the eraser token to mirror the attention pattern the target token originally had (present in the formulation; disabled by default in this codebase, see `redirection_loss_scale`).
- Only the eraser key (and a small number of auxiliary parameters) are trained — the base model weights stay frozen, making the edit lightweight and easy to toggle or scale at inference time.

---

## Repository structure

```
.
├── train.py                    # Training entry point / CLI
├── train_with_scale.py         # Variant of the training loop with additional eraser-scale handling
├── attn_processor.py           # Core FluxAttentionEraser attention processor (the ARCE mechanism)
├── attn_processor_with_scale.py# Variant of the attention processor with scale support
├── attn_utils.py                # Attention-map helpers (weighted SDPA, token indices, visualization)
├── data_loader.py               # Dataset / DataLoader for target + related (preserve) image sets
└── utils.py                     # Misc utilities (GPU selection, logging, checkpoint helpers, etc.)
```

At a high level:

- `attn_processor.py` defines `FluxAttentionEraser`, an `nn.Module` that replaces the default `FluxAttnProcessor2_0` on every attention layer of the FLUX transformer. It holds the trainable eraser key, the (non-trainable) eraser value, and computes the erasure / preservation / redirection losses from the attention weights during training.
- `train.py` wires everything together: it loads FLUX.1-dev, freezes the base model, attaches an eraser processor to every attention layer, and runs a training loop that combines the diffusion reconstruction loss with the attention-based losses above.
- `data_loader.py` builds a `DataLoader` over a directory of target-concept images (with sidecar `.json`/`.txt` captions) and, optionally, a second directory of unrelated/"preserve" images, for training and evaluating preservation behavior.
- `attn_utils.py` contains the modified scaled-dot-product-attention that also returns attention weights (needed for the loss terms), token-index lookup utilities, and attention-map visualization/saving helpers.

---

## Requirements

- Python 3.10+
- PyTorch (with CUDA)
- [`diffusers`](https://github.com/huggingface/diffusers) (with FLUX support)
- `transformers`, `accelerate`
- `einops`, `opencv-python`, `numpy`, `pandas`, `Pillow`, `tqdm`
- `wandb` (used for logging; can be run in offline/disabled mode)
- Access to **FLUX.1-dev** weights (`black-forest-labs/FLUX.1-dev`) on the Hugging Face Hub

```bash
pip install torch diffusers transformers accelerate einops opencv-python numpy pandas pillow tqdm wandb
```

A machine with a CUDA GPU (or two — the pipeline supports splitting the transformer and the text/VAE encoders across separate devices) is strongly recommended given the size of FLUX.1-dev.

---

## Data preparation

Training expects, per concept to erase:

- **Target images**: a folder of images depicting the concept to erase, each with a sidecar caption file (`<name>.json` with a `"caption"` field, or `<name>.txt`) containing the prompt used to generate/describe that image.
- **Related / "preserve" images** *(optional)*: a folder of images that should remain unaffected by the erasure (e.g. images of similar but distinct concepts), used to compute the preservation loss and evaluate side effects.
- **Cached neutral/target value tensors**: per-layer `.pt` tensors (`{layer_ind}.pt`) representing the neutral anchor's and the target concept's attention *value*, referenced via `--neutral_val_path` and `--target_val_path`.
- **A metadata file** (default `data/metadata.json`) mapping concept names to their associated paths/settings.
- **An evaluation-prompts file** (default `data/train_data/train_eval_prompts.json`) listing prompts used to periodically render erase/preserve comparison grids during training.

---

## Training

```bash
python train.py \
  --concept "Spiderman" \
  --img_dir data/train_data/Spiderman/Spiderman_data \
  --img_dir_2 data/train_data/Spiderman/no_Spiderman_data \
  --target_val_path data/val_files/Spiderman_val \
  --neutral_val_path data/val_files/man_val \
  --learning_rate 3e-3 \
  --max_train_steps 1000 \
  --main_device cuda:0
```

Notable CLI options (see `train.py` for the full list):

| Flag | Description |
|---|---|
| `--concept` | Concept to erase, e.g. `"Spiderman"` (required). |
| `--img_dir` / `--img_dir_2` | Image directories for the target concept and for "other"/related images. |
| `--target_val_path` / `--neutral_val_path` | Directories of cached per-layer value tensors for the target concept and the neutral anchor. |
| `--load_existing_keys` | Path to a directory of previously trained eraser keys to resume/reuse instead of re-initializing from the target token. |
| `--erasure_loss_scale`, `--preservation_loss_scale`, `--redirection_loss_scale`, `--reconstruction_loss_scale` | Relative weights of each loss term. |
| `--init_key_scale` | Scale applied when initializing the eraser key from the target token's key. |
| `--scale_eraser_key_inference` | Multiplies the eraser key at inference time to strengthen/weaken erasure without retraining. |
| `--n_layers_to_skip` | Number of attention layers to randomly skip training on each step, for memory savings. |
| `--visualize_attention` | Save side-by-side attention-map visualizations (original vs. erased) during training. |
| `--save_full_model` | Also checkpoint the full transformer state dict, not just the eraser parameters. |
| `--main_device` / `--second_device` | Devices for the transformer vs. the text encoders/VAE (can be the same). |

Checkpoints are written under `out/<concept>/<run_id>/ckpt/`, including a lightweight `eraser_kv_<lr>.pt` file containing just the learned per-layer key/value additions — the artifact you need to reproduce or ship the erasure without the rest of the model.

`train_with_scale.py` and `attn_processor_with_scale.py` provide a variant of the pipeline with additional support for scaling the eraser's influence.

---

## Inference / evaluation

During training, `test()` periodically renders a grid comparing generations **with** and **without** the eraser active (`Mode.INFERENCE` vs. `Mode.INFERENCE_ORIGINAL`) for a fixed set of erase/preserve prompts, saved as labeled side-by-side JPEGs under the run's output directory.

There is also an interactive mode (`generate_for_input_prompt`) for typing arbitrary prompts and comparing baseline vs. erased generations on the fly, and a `scale_eraser_key_inference` knob to dial erasure strength up or down at inference time without retraining.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{schechter2026concept,
  title={Concept Erasure via Attention Redirection},
  author={Schechter, Amit and Gal, Rinon and Kedem, Ofir and Chechik, Gal and Cohen-Or, Daniel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4572--4581},
  year={2026}
}
```
---

## Acknowledgements

Built on top of [🤗 diffusers](https://github.com/huggingface/diffusers) and the FLUX.1 model family from Black Forest Labs.
