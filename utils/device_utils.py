import torch
import logging

def get_device(device_name=None):
    """
    Determines the best available device.
    Prioritizes CUDA -> MPS -> CPU.
    """
    if device_name:
        return torch.device(device_name)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def is_mps_available():
    return torch.backends.mps.is_available()

def is_cuda_available():
    return torch.cuda.is_available()

def empty_cache():
    if is_cuda_available():
        torch.cuda.empty_cache()
    elif is_mps_available():
        torch.mps.empty_cache()

def get_device_name():
    device = get_device()
    if device.type == 'cuda':
        return torch.cuda.get_device_name(0)
    elif device.type == 'mps':
        return "Apple Silicon (MPS)"
    else:
        return "CPU"
