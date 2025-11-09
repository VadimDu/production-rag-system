"""
Device configuration for the RAG system.

This module provides utilities for setting up the computation device
based on available hardware.
"""

import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """Setup computation device based on available hardware
    
    Returns:
        torch.device: The configured device (mps, cuda, or cpu)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def get_device_info() -> Tuple[torch.device, str]:
    """Get device information
    
    Returns:
        Tuple[torch.device, str]: Device object and device name
    """
    device = setup_device()
    
    if device.type == "mps":
        device_name = "Apple Silicon GPU"
    elif device.type == "cuda":
        device_name = "NVIDIA CUDA"
    else:
        device_name = "CPU"
    
    return device, device_name