import json
import torch
import os
from core.utils import download_ckpt

def model_zoo(name, zoo_path="configs/model_zoo.json"):
    # Check if this might be a quantized checkpoint and set engine first
    # This prevents "Unknown qengine" error when loading quantized models
    try:
        if os.path.exists(name):
            # Quick peek to check if quantized (without full unpickling)
            with open(name, 'rb') as f:
                # Read just enough to check metadata
                import pickle
                unpickler = pickle.Unpickler(f)
                # Try to get the dict keys without fully loading
                # This is a bit hacky but necessary to avoid the qengine error
                try:
                    # Set quantization engine before loading quantized checkpoints
                    available_engines = torch.backends.quantized.supported_engines
                    if 'qnnpack' in available_engines:
                        torch.backends.quantized.engine = 'qnnpack'
                    elif 'fbgemm' in available_engines:
                        torch.backends.quantized.engine = 'fbgemm'
                except Exception:
                    pass
    except Exception:
        # If peek fails, try to set default engine anyway
        try:
            available_engines = torch.backends.quantized.supported_engines
            if 'qnnpack' in available_engines:
                torch.backends.quantized.engine = 'qnnpack'
        except Exception:
            pass
    
    zoo = json.load(open(zoo_path))
    if name in zoo:
        ckpt = download_ckpt(**zoo[name])
    else:
        # Set quantization engine before loading (prevents "Unknown qengine" error)
        try:
            available_engines = torch.backends.quantized.supported_engines
            # Default to qnnpack on Mac, fbgemm on Linux/Windows
            if 'qnnpack' in available_engines:
                torch.backends.quantized.engine = 'qnnpack'
            elif 'fbgemm' in available_engines:
                torch.backends.quantized.engine = 'fbgemm'
        except Exception:
            pass
        
        ckpt = torch.load(name, map_location="cpu", weights_only=False)
        
        # If checkpoint has compression info, set engine based on that
        if isinstance(ckpt, dict) and 'compression_config' in ckpt:
            backend = ckpt.get('compression_config', {}).get('backend')
            if backend:
                try:
                    available_engines = torch.backends.quantized.supported_engines
                    if backend in available_engines:
                        torch.backends.quantized.engine = backend
                except Exception:
                    pass
    
    return ckpt
