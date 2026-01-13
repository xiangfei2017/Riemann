# BSD 3-Clause License
#
# Copyright (c) 2025, Fei Xiang
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Riemann Library CUDA Module: GPU Acceleration Support

This module provides CUDA/GPU acceleration support for the Riemann library, including:
- Device management (CPU/GPU)
- Context management for device switching
- Memory management utilities
- CUDA availability checking

The module leverages CuPy for CUDA operations when available, and falls back to CPU-only mode
when CUDA is not present or CuPy cannot be imported.
"""

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    print("Warning: Cannot import cupy, riemann will only work on CPU")
    CUPY_AVAILABLE = False
    cp = None

from contextlib import contextmanager
import threading

# 线程本地存储，用于跟踪当前是否在CUDA设备上下文中
_thread_local = threading.local()

# 全局默认设备
_default_device = None

class Device:
    """
    Represents a device (CPU or CUDA GPU).
    
    Args:
        type_or_device (str or int): A device type ('cpu' or 'cuda') or a device index.
    """
    
    def __init__(self, device:str|int='cpu'):
        if isinstance(device, int):
            self.type = 'cuda'
            self.index = device
        elif isinstance(device, str):
            if ':' in device:
                self.type, idx_str = device.split(':')
                self.index = int(idx_str)
            else:
                self.type = device
                self.index = 0 if self.type == 'cuda' else None
        else:
            raise ValueError(f"Invalid device type: {device}")
        
        if self.type == 'cuda' and not CUPY_AVAILABLE:
            raise RuntimeError("CUDA is not available")
        
        if self.type != 'cuda' and self.type != 'cpu':
            raise ValueError(f"Invalid device string: '{self.type}'")
        
        # 检查CUDA设备索引有效性
        if self.type == 'cuda':
            if self.index is None:
                self.index = 0
            # 检查索引是否在有效范围内
            device_num = device_count()
            if self.index >= device_num or self.index < 0:
                raise RuntimeError(f"CUDA device index {self.index} is out of range (0-{device_num-1})")
    
    def __eq__(self, other):
        if not isinstance(other, Device):
            return False
        return self.type == other.type and self.index == other.index
    
    def __str__(self):
        if self.type == 'cuda':
            return f'cuda:{self.index}'
        return 'cpu'
    
    def __repr__(self):
        return f"device(type='{self.type}', index={self.index})"
    
    def __enter__(self):
        """
        Enter the device context.
        """
        # Save current state
        self._old_in_cuda_context = getattr(_thread_local, 'in_cuda_context', False)
        self._old_device = None
        
        if self.type == 'cpu':
            # Using CPU, set context flag to False
            _thread_local.in_cuda_context = False
        else:
            # Using CUDA, save current device and switch to this device
            if CUPY_AVAILABLE:
                self._old_device = current_device()
                set_device(self.index)
            _thread_local.in_cuda_context = True
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the device context.
        """
        # Restore thread-local context state
        _thread_local.in_cuda_context = self._old_in_cuda_context
        
        # Restore device if needed
        if hasattr(self, '_old_device') and self._old_device is not None:
            set_device(self._old_device)


def is_available() -> bool:
    """
    Checks if CUDA is available.
    
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return CUPY_AVAILABLE


def device_count() -> int:
    """
    Returns the number of available CUDA devices.
    
    Returns:
        int: Number of available CUDA devices.
    """
    if not CUPY_AVAILABLE:
        return 0
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def current_device() -> int:
    """
    Returns the index of the current CUDA device.
    
    Returns:
        int: Index of the current CUDA device.
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    return cp.cuda.runtime.getDevice()


def get_device_name(device_idx: int) -> str:
    """
    Returns the name of the CUDA device at the given index.
    
    Args:
        device_idx (int): Index of the CUDA device.
    
    Returns:
        str: Name of the CUDA device.
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    name = cp.cuda.runtime.getDeviceProperties(device_idx)['name']
    # 确保返回的是字符串而不是 bytes
    if isinstance(name, bytes):
        return name.decode('utf-8')
    return name


def set_device(device_idx: int) -> None:
    """
    Sets the current CUDA device.
    
    Args:
        device_idx (int): Index of the CUDA device to set as current.
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    cp.cuda.runtime.setDevice(device_idx)


def empty_cache() -> None:
    """
    Empties the CUDA cache.
    """
    if CUPY_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

def is_in_cuda_context() -> bool:
    """
    Checks if the current thread is inside a CUDA device context.
    
    Returns:
        bool: True if inside a CUDA device context, False otherwise.
    """
    return getattr(_thread_local, 'in_cuda_context', False)

def memory_allocated(device_idx: int | None = None) -> int:
    """
    Returns the amount of memory allocated on the given CUDA device.
    
    Args:
        device_idx (int, optional): Index of the CUDA device. If None, uses the current device.
    
    Returns:
        int: Amount of memory allocated in bytes.
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    
    if device_idx is not None:
        old_device = current_device()
        set_device(device_idx)
        
    try:
        return cp.get_default_memory_pool().used_bytes()
    finally:
        if device_idx is not None:
            set_device(old_device)

def get_default_device() -> Device:
    """
    Gets the default device for tensor creation.
    
    Returns:
        Device: The default device.
    """
    global _default_device
    
    # 如果未设置默认设备，返回CPU
    if _default_device is None:
        return Device('cpu')
    
    return _default_device

def set_default_device(device: str | int | Device) -> None:
    """
    Sets the default device for tensor creation.
    
    Args:
        device (str, int, or Device): The device to set as default. Can be:
            - String: 'cpu', 'cuda', or 'cuda:0', 'cuda:1'
            - Integer: CUDA device index
            - Device object
    
    Example:
        >>> import riemann as rm
        >>> rm.get_default_device()
        device(type='cpu', index=None)
        >>> rm.set_default_device('cuda')
        >>> rm.get_default_device()
        device(type='cuda', index=0)
        >>> rm.set_default_device('cuda:1')
        >>> rm.get_default_device()
        device(type='cuda', index=1)
    """
    global _default_device
    
    # 解析设备参数并创建Device对象
    if isinstance(device, (str,int)):
        _default_device = Device(device)
    elif isinstance(device, Device):
        _default_device = device
    else:
        raise ValueError(f"Invalid device type: {type(device).__name__}")
