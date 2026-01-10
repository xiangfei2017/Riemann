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
        index (int, optional): Device index. If None, will be inferred from the first argument.
    """
    
    def __init__(self, type_or_device='cpu', index=None):
        if isinstance(type_or_device, int):
            self.type = 'cuda'
            self.index = type_or_device
        elif isinstance(type_or_device, str):
            if ':' in type_or_device:
                self.type, idx_str = type_or_device.split(':')
                self.index = int(idx_str)
            else:
                self.type = type_or_device
                self.index = 0 if self.type == 'cuda' else None
        else:
            raise ValueError(f"Invalid device type: {type_or_device}")
        
        if self.type == 'cuda' and not CUPY_AVAILABLE:
            raise RuntimeError("CUDA is not available")
        
        if self.type != 'cuda' and self.type != 'cpu':
            raise ValueError(f"Invalid device type: {self.type}")
        
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
    return cp.cuda.runtime.getDeviceProperties(device_idx)['name']


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


def set_default_device(device) -> None:
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
    if isinstance(device, str):
        _default_device = Device(device)
    elif isinstance(device, int):
        _default_device = Device('cuda', device)
    elif isinstance(device, Device):
        _default_device = device
    else:
        raise ValueError(f"Invalid device type: {type(device).__name__}")


@contextmanager
def device(device_id):
    """
    Context manager that changes the current device to the given one.
    
    This context manager is thread-safe, as it leverages the thread-local nature
    of CUDA contexts. Each thread has its own current device setting.
    
    Args:
        device_id (int, str, or Device): Device to use in the context. Can be:
            - Integer: CUDA device index (e.g., 0, 1)
            - String: 'cpu', 'cuda', or 'cuda:0', 'cuda:1'
            - Device object: Device('cpu') or Device('cuda:0')
    
    Example:
        >>> with device('cuda:1'):
        ...     x = tensor([1, 2, 3])  # Automatically uses CUDA device 1
        >>> with device(0):
        ...     y = tensor([4, 5, 6])  # Automatically uses CUDA device 0
        >>> with device('cpu'):
        ...     z = tensor([7, 8, 9])  # Automatically uses CPU
    
    Note:
        Supports nesting of CPU and GPU contexts.
    """
    
    # 解析设备参数
    is_cpu = False
    device_idx = 0
    
    if isinstance(device_id, Device):
        if device_id.type == 'cpu':
            is_cpu = True
        else:
            device_idx = device_id.index
    elif isinstance(device_id, int):
        # 整数被视为CUDA设备索引
        device_idx = device_id
    elif isinstance(device_id, str):
        if ':' in device_id:
            device_type, idx_str = device_id.split(':')
            if device_type == 'cpu':
                is_cpu = True
            elif device_type == 'cuda':
                device_idx = int(idx_str)
            else:
                raise ValueError("Device context manager only supports 'cpu' and 'cuda' devices")
        else:
            if device_id == 'cpu':
                is_cpu = True
            elif device_id == 'cuda':
                device_idx = 0
            else:
                raise ValueError("Device context manager only supports 'cpu' and 'cuda' devices")
    else:
        raise ValueError(f"Invalid device type: {type(device_id).__name__}")
    
    # 检查设备索引有效性（仅当使用CUDA时）
    if not is_cpu:
        device_num = device_count()
        if device_idx >= device_num or device_idx < 0:
            raise RuntimeError(f"CUDA device index {device_idx} is out of range (0-{device_num-1})")
    
    # 保存当前设备并切换到新设备
    old_in_cuda_context = getattr(_thread_local, 'in_cuda_context', False)
    old_device = None
    
    try:
        if is_cpu:
            # 使用CPU时，标记为不在CUDA上下文中
            _thread_local.in_cuda_context = False
        else:
            # 使用CUDA时，保存当前设备并切换到新设备
            old_device = current_device()
            set_device(device_idx)
            _thread_local.in_cuda_context = True
        yield
    finally:
        # 恢复线程的CUDA上下文状态
        _thread_local.in_cuda_context = old_in_cuda_context
        # 如果之前使用的是CUDA，恢复到之前的设备
        if old_device is not None:
            set_device(old_device)

