# Multi-GPU ComfyUI Launcher

**Run multiple ComfyUI instances without melting your RAM**

## The Problem

You've got 2 GPUs. You want 2 ComfyUI instances. But that 20GB model? It loads twice. Your 64GB RAM starts sweating. 💦

## The Solution

This launcher loads your big models **once** into shared memory, then spawns multiple ComfyUI workers that all read from the same place. 

Think of it like a library - one copy of the book, multiple people reading it.

## Quick Start

```bash
python multi_gpu_launcher_v4.py \
    --gpus 0,1 \
    --listen 0.0.0.0 \
    --unet /path/to/your/unet.safetensors \
    --clip /path/to/your/clip.safetensors \
    --vae /path/to/your/vae.safetensors
```

This gives you:
- **Port 8188** → GPU 0
- **Port 8189** → GPU 1

## Options

| Flag | What it does |
|------|-------------|
| `--gpus 0,1,2` | Which GPUs to use |
| `--base-port 8188` | Starting port number |
| `--unet` | Your diffusion model (the big one!) |
| `--clip` | Text encoder model(s) - can use multiple times |
| `--vae` | VAE model |
| `--weight-dtype fp8_e4m3fn` | For FP8 quantized models |
| `--listen 0.0.0.0` | Listen on all interfaces |

## What Gets Shared?

✅ **Share these** (they're huge):
- UNet/Diffusion model (~10-20GB)
- CLIP/Text encoders (~5-15GB)
- VAE (~1-2GB)

❌ **Don't bother sharing**:
- LoRAs (tiny, ~100-500MB)
- ControlNets (loaded on-demand anyway)

## RAM Savings Example

| Setup | RAM Usage |
|-------|-----------|
| 2x ComfyUI (normal) | ~60GB 😰 |
| 2x ComfyUI (shared) | ~32GB 😎 |

## How It Works (the short version)

1. Parent process loads model tensors into OS shared memory
2. Spawns worker processes (one per GPU)
3. Workers use the same tensors - no copying!
4. Each worker runs its own ComfyUI server

## Troubleshooting

**"CUDA out of memory"** - Try adding `--lowvram` to extra args:
```bash
python multi_gpu_launcher_v4.py --gpus 0,1 --unet /path/to/model -- --lowvram
```

**Workers crash on startup** - Check that model paths are correct and accessible.

**Different RAM than expected** - Use `smem -p -k | grep python` for accurate shared memory stats. Regular `top` lies about shared memory.

---

*Made with a lot of pain and too many failed attempts at fork() + CUDA*
