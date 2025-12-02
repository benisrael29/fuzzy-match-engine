"""GPU utilities for acceleration support."""
import numpy as np
from typing import Optional, Tuple, Any, Union
import warnings

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def detect_gpu() -> Tuple[bool, Optional[str]]:
    """
    Detect if GPU is available and return status.
    
    Returns:
        Tuple of (is_available, device_name or None)
    """
    if not CUPY_AVAILABLE:
        return False, None
    
    try:
        device = cp.cuda.Device(0)
        device.use()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        return True, device_name
    except Exception:
        return False, None


def get_gpu_memory_info() -> Optional[Tuple[int, int]]:
    """
    Get GPU memory information.
    
    Returns:
        Tuple of (free_memory_mb, total_memory_mb) or None if unavailable
    """
    if not CUPY_AVAILABLE:
        return None
    
    try:
        mempool = cp.get_default_memory_pool()
        meminfo = cp.cuda.runtime.memGetInfo()
        free_mb = meminfo[0] // (1024 * 1024)
        total_mb = meminfo[1] // (1024 * 1024)
        return free_mb, total_mb
    except Exception:
        return None


def should_use_gpu(
    use_gpu: Union[bool, str],
    dataset_size: int,
    min_size_threshold: int = 100000
) -> bool:
    """
    Determine if GPU should be used based on configuration and dataset size.
    
    Args:
        use_gpu: True, False, or "auto"
        dataset_size: Size of the dataset
        min_size_threshold: Minimum dataset size to use GPU (default: 100K)
    
    Returns:
        True if GPU should be used
    """
    if use_gpu is False:
        return False
    
    if not CUPY_AVAILABLE:
        if use_gpu is True:
            warnings.warn("CuPy not available, falling back to CPU")
        return False
    
    if use_gpu == "auto":
        is_available, _ = detect_gpu()
        if not is_available:
            return False
        return dataset_size >= min_size_threshold
    
    if use_gpu is True:
        is_available, _ = detect_gpu()
        if not is_available:
            warnings.warn("GPU not available, falling back to CPU")
            return False
        return True
    
    return False


def get_array_module(use_gpu: bool):
    """
    Get the appropriate array module (CuPy or NumPy).
    
    Args:
        use_gpu: Whether to use GPU
    
    Returns:
        CuPy module if use_gpu and available, else NumPy
    """
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np


def to_gpu(array: np.ndarray, use_gpu: bool) -> Union[np.ndarray, Any]:
    """
    Convert numpy array to GPU array if GPU is enabled.
    
    Args:
        array: NumPy array
        use_gpu: Whether to use GPU
    
    Returns:
        GPU array (CuPy) or original NumPy array
    """
    if use_gpu and CUPY_AVAILABLE:
        try:
            return cp.asarray(array)
        except Exception:
            warnings.warn("Failed to transfer array to GPU, using CPU")
            return array
    return array


def to_cpu(array: Any) -> np.ndarray:
    """
    Convert GPU array back to CPU (NumPy) array.
    
    Args:
        array: GPU array (CuPy) or NumPy array
    
    Returns:
        NumPy array
    """
    if CUPY_AVAILABLE and hasattr(array, 'get'):
        return array.get()
    return np.asarray(array)


def check_gpu_memory_available(required_mb: int, memory_limit: Optional[int] = None) -> bool:
    """
    Check if sufficient GPU memory is available.
    
    Args:
        required_mb: Required memory in MB
        memory_limit: Optional memory limit in MB
    
    Returns:
        True if sufficient memory is available
    """
    if not CUPY_AVAILABLE:
        return False
    
    try:
        meminfo = get_gpu_memory_info()
        if meminfo is None:
            return False
        
        free_mb, total_mb = meminfo
        available_mb = memory_limit if memory_limit and memory_limit < free_mb else free_mb
        
        return available_mb >= required_mb
    except Exception:
        return False


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if CUPY_AVAILABLE:
        try:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
        except Exception:
            pass

