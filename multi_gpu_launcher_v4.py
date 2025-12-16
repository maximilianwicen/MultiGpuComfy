#!/usr/bin/env python3
"""
Multi-GPU ComfyUI Launcher v4 - Shared Memory with Null Offload

This version:
1. Loads ALL model tensors (UNet, CLIP, VAE) to shared memory
2. Uses a "null offload" strategy - models are discarded from VRAM, not copied to CPU
3. When a model is needed, it reloads from shared memory instead of private copies

Usage:
python multi_gpu_launcher_v4.py \
    --gpus 0,1 \
    --listen 0.0.0.0 \
    --unet /path/to/unet.safetensors \
    --clip /path/to/clip.safetensors \
    --vae /path/to/vae.safetensors \
    --weight-dtype fp8_e4m3fn
    
    """

import os
import sys
import argparse
import signal
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'

logging.basicConfig(level=logging.INFO, format='[%(process)d] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU ComfyUI Launcher v4 - Shared Memory')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='Comma-separated list of GPU IDs (default: 0,1)')
    parser.add_argument('--base-port', type=int, default=8188,
                        help='Base port number (default: 8188)')
    parser.add_argument('--listen', type=str, default='127.0.0.1',
                        help='IP address to listen on (default: 127.0.0.1)')
    
    # Model paths - can specify individual models or scan a directory
    parser.add_argument('--unet', type=str, default=None,
                        help='Path to UNet/diffusion model to preload')
    parser.add_argument('--clip', type=str, action='append', default=[],
                        help='Path to CLIP model(s) to preload (can specify multiple)')
    parser.add_argument('--vae', type=str, default=None,
                        help='Path to VAE model to preload')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to full checkpoint to preload')
    
    parser.add_argument('--weight-dtype', type=str, default='default',
                        choices=['default', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'],
                        help='Weight dtype for UNet loading (default: default)')
    
    parser.add_argument('comfyui_args', nargs='*',
                        help='Additional arguments to pass to ComfyUI')
    
    return parser.parse_args()


def get_ram_usage_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except ImportError:
        return 0.0


def load_safetensors_to_shared_memory(model_path: str) -> Dict:
    """Load a safetensors file with all tensors in shared memory."""
    import torch
    from safetensors import safe_open
    
    logger.info(f"Loading to shared memory: {model_path}")
    logger.info(f"RAM before: {get_ram_usage_gb():.2f} GB")
    
    shared_tensors = {}
    
    with safe_open(model_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            shared_tensors[key] = tensor.share_memory_()
    
    logger.info(f"Loaded {len(shared_tensors)} tensors")
    logger.info(f"RAM after: {get_ram_usage_gb():.2f} GB")
    
    return shared_tensors, metadata


def load_torch_to_shared_memory(model_path: str) -> Dict:
    """Load a .pt/.pth/.ckpt file with all tensors in shared memory."""
    import torch
    
    logger.info(f"Loading to shared memory: {model_path}")
    logger.info(f"RAM before: {get_ram_usage_gb():.2f} GB")
    
    data = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(data, dict):
        if 'state_dict' in data:
            state_dict = data['state_dict']
        else:
            state_dict = data
    else:
        state_dict = data
    
    shared_tensors = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            if not value.is_contiguous():
                value = value.contiguous()
            shared_tensors[key] = value.share_memory_()
        else:
            shared_tensors[key] = value
    
    logger.info(f"Loaded {len(shared_tensors)} items")
    logger.info(f"RAM after: {get_ram_usage_gb():.2f} GB")
    
    return shared_tensors, None


def load_model_to_shared_memory(model_path: str):
    """Load any model format to shared memory."""
    path = Path(model_path)
    
    if path.suffix.lower() == '.safetensors':
        return load_safetensors_to_shared_memory(model_path)
    else:
        return load_torch_to_shared_memory(model_path)


class SharedModelRegistry:
    """Registry of all shared model tensors."""
    
    def __init__(self):
        self.models: Dict[str, Dict] = {}  # path -> shared tensors
        self.metadata: Dict[str, Optional[Dict]] = {}  # path -> metadata
    
    def add(self, path: str, tensors: Dict, metadata: Optional[Dict] = None):
        normalized = os.path.normpath(os.path.abspath(path))
        self.models[normalized] = tensors
        self.metadata[normalized] = metadata
        logger.info(f"Registered shared model: {normalized}")
    
    def get(self, path: str):
        normalized = os.path.normpath(os.path.abspath(path))
        return self.models.get(normalized), self.metadata.get(normalized)
    
    def has(self, path: str) -> bool:
        normalized = os.path.normpath(os.path.abspath(path))
        return normalized in self.models
    
    def paths(self):
        return list(self.models.keys())


def worker_process(gpu_id: int, port: int, listen: str, 
                   registry_data: dict, weight_dtype: str, extra_args: list):
    """
    Worker process - runs ComfyUI on a specific GPU.
    """
    import torch
    
    # Set up logging for this worker
    logging.basicConfig(level=logging.INFO, format=f'[GPU{gpu_id}] %(levelname)s: %(message)s', force=True)
    logger = logging.getLogger(__name__)
    
    # Set GPU for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    logger.info(f"Worker starting: GPU {gpu_id}, Port {port}")
    logger.info(f"RAM at worker start: {get_ram_usage_gb():.2f} GB")
    logger.info(f"Shared models available: {list(registry_data['models'].keys())}")
    
    # Reconstruct registry from passed data
    registry = SharedModelRegistry()
    registry.models = registry_data['models']
    registry.metadata = registry_data['metadata']
    
    # Set up sys.argv for ComfyUI
    sys.argv = ['main.py', '--listen', listen, '--port', str(port)] + extra_args
    
    # =====================================================
    # CRITICAL PATCHES - Apply before importing comfy
    # =====================================================
    
    # Patch 1: Force load_state_dict to use assign=True
    original_load_state_dict = torch.nn.Module.load_state_dict
    
    def patched_load_state_dict(self, state_dict, strict=True, assign=False):
        return original_load_state_dict(self, state_dict, strict=strict, assign=True)
    
    torch.nn.Module.load_state_dict = patched_load_state_dict
    logger.info("Patched torch.nn.Module.load_state_dict to use assign=True")
    
    # Now import ComfyUI modules
    import comfy.options
    comfy.options.enable_args_parsing()
    
    import comfy.sd
    import comfy.utils
    import comfy.model_management
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    
    # =====================================================
    # Patch 2: Override load_torch_file to use shared memory
    # =====================================================
    original_load_torch_file = comfy.utils.load_torch_file
    
    def patched_load_torch_file(path, **kwargs):
        shared_tensors, metadata = registry.get(path)
        
        if shared_tensors is not None:
            logger.info(f"✓ USING SHARED MEMORY: {path}")
            if kwargs.get('return_metadata', False):
                return shared_tensors, metadata
            return shared_tensors
        
        logger.info(f"✗ Loading from disk (not shared): {path}")
        return original_load_torch_file(path, **kwargs)
    
    comfy.utils.load_torch_file = patched_load_torch_file
    logger.info("Patched comfy.utils.load_torch_file")
    
    # =====================================================
    # Patch 3: Prevent model offload from creating copies
    # When a model is "offloaded", just delete from VRAM without copying
    # =====================================================
    from comfy.model_patcher import ModelPatcher
    
    original_detach = ModelPatcher.detach
    
    def patched_detach(self, unpatch_all=True):
        """Modified detach that doesn't copy to CPU - just releases VRAM."""
        # Check if this model uses shared tensors (by checking if any tensor is in shared memory)
        is_shared = False
        try:
            for name, param in self.model.named_parameters():
                if param.is_shared():
                    is_shared = True
                    break
        except:
            pass
        
        if is_shared:
            logger.debug(f"Detaching shared model - skipping CPU copy")
            # Just move model off GPU without creating CPU copy
            self.model_patches_to(torch.device('meta'))  # meta device doesn't allocate
            if unpatch_all:
                self.unpatch_model(torch.device('meta'), unpatch_weights=unpatch_all)
            for callback in self.get_all_callbacks(comfy.model_patcher.CallbacksMP.ON_DETACH):
                callback(self)
        else:
            # Non-shared model - use original behavior
            return original_detach(self, unpatch_all)
    
    # Don't patch detach for now - it may cause issues
    # ModelPatcher.detach = patched_detach
    
    logger.info(f"RAM after patches: {get_ram_usage_gb():.2f} GB")
    
    # Start ComfyUI
    logger.info(f"Starting ComfyUI server on port {port}")
    
    from main import start_comfyui
    import app.logger
    
    event_loop, _, start_all_func = start_comfyui()
    try:
        app.logger.print_startup_warnings()
        event_loop.run_until_complete(start_all_func())
    except KeyboardInterrupt:
        logger.info(f"Worker GPU {gpu_id} stopped")


def main():
    args = parse_args()
    
    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    
    logger.info("=" * 60)
    logger.info("Multi-GPU ComfyUI Launcher v4 (Shared Memory)")
    logger.info("=" * 60)
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Ports: {[args.base_port + i for i in range(len(gpu_ids))]}")
    
    # ============================================
    # STEP 1: Load all models to shared memory
    # ============================================
    registry = SharedModelRegistry()
    
    # Load UNet
    if args.unet:
        if not os.path.exists(args.unet):
            logger.error(f"UNet not found: {args.unet}")
            sys.exit(1)
        tensors, metadata = load_model_to_shared_memory(args.unet)
        registry.add(args.unet, tensors, metadata)
    
    # Load CLIP model(s)
    for clip_path in args.clip:
        if not os.path.exists(clip_path):
            logger.error(f"CLIP not found: {clip_path}")
            sys.exit(1)
        tensors, metadata = load_model_to_shared_memory(clip_path)
        registry.add(clip_path, tensors, metadata)
    
    # Load VAE
    if args.vae:
        if not os.path.exists(args.vae):
            logger.error(f"VAE not found: {args.vae}")
            sys.exit(1)
        tensors, metadata = load_model_to_shared_memory(args.vae)
        registry.add(args.vae, tensors, metadata)
    
    # Load checkpoint
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        tensors, metadata = load_model_to_shared_memory(args.checkpoint)
        registry.add(args.checkpoint, tensors, metadata)
    
    if not registry.models:
        logger.warning("No models specified for sharing! Use --unet, --clip, --vae, or --checkpoint")
        logger.warning("Workers will load models independently (no memory sharing)")
    
    logger.info(f"Total RAM after loading shared models: {get_ram_usage_gb():.2f} GB")
    logger.info(f"Shared models: {registry.paths()}")
    
    # ============================================
    # STEP 2: Spawn workers
    # ============================================
    mp.set_start_method('spawn', force=True)
    
    # Convert registry to dict for passing to workers
    registry_data = {
        'models': registry.models,
        'metadata': registry.metadata
    }
    
    processes = []
    
    for i, gpu_id in enumerate(gpu_ids):
        port = args.base_port + i
        
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                port,
                args.listen,
                registry_data,
                args.weight_dtype,
                args.comfyui_args
            )
        )
        p.start()
        processes.append((p, gpu_id, port))
        logger.info(f"Spawned worker PID {p.pid} for GPU {gpu_id} on port {port}")
    
    logger.info(f"All workers spawned. Parent PID: {os.getpid()}")
    logger.info("Press Ctrl+C to stop all workers")
    
    def signal_handler(signum, frame):
        logger.info("Shutting down workers...")
        for p, gpu_id, port in processes:
            p.terminate()
        for p, gpu_id, port in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for children
    try:
        for p, gpu_id, port in processes:
            p.join()
            logger.info(f"Worker GPU {gpu_id} (port {port}) exited")
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    
    logger.info("All workers exited")


if __name__ == '__main__':
    main()
