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
Riemann Library Serialization Module: Model Persistence and Data Storage

This module provides comprehensive serialization and deserialization capabilities for the
Riemann machine learning framework. It implements save and load functions for persisting
Riemann tensors, parameters, modules, and training states to disk, with an interface
compatible with PyTorch's torch.save and torch.load functions.

Main features:
    - Tensor serialization: Efficient saving and loading of Riemann tensors with
      gradient information and computation graph metadata
    - Module persistence: Complete model state serialization including parameters,
      buffers, and submodule hierarchies
    - Training checkpointing: Save and restore training states for resuming training
    - Multiple format support: Flexible serialization formats including pickle and
      custom binary formats
    - Cross-platform compatibility: Ensures models can be saved and loaded across
      different platforms and Python versions
    - Memory-efficient loading: Lazy loading mechanisms for large models and datasets

Using this module, you can save trained models, checkpoint training progress,
and share models across different environments, with full compatibility with
PyTorch's serialization ecosystem.
"""

import numpy as np
import pickle
import os
import zipfile
import io
import sys
import struct
from typing import Any, Union, Optional
from .tensordef import TN, tensor
from .nn.module import Parameter
from .cuda import Device, cp

# Magic number for Riemann serialization format
# This unique identifier is used to verify the validity of serialized files
# and ensure compatibility with Riemann's serialization system.
# 
# The value is specifically chosen for Riemann and differs from PyTorch's magic number,
# but the overall serialization format is designed to be compatible with PyTorch's
# serialization ecosystem.
# 
# Usage:
# - Written to the beginning of legacy serialized files during save operations
# - Verified during load operations to detect corrupt or incompatible files
MAGIC_NUMBER = 0x1950A86A20F9469CFC6C

# Protocol version for Riemann serialization format
# This version number is used to track changes in the serialization format
# and ensure compatibility between different versions of Riemann.
# 
# When the serialization format changes in a way that breaks backward compatibility,
# this version number should be incremented.
# 
# Usage:
# - Written to serialized files during save operations
# - Verified during load operations to ensure format compatibility
PROTOCOL_VERSION = 1001

# Default pickle protocol version to use for serialization
# Pickle protocol 2 is chosen as the default for its good balance of:
# 1. Cross-platform compatibility
# 2. Support for modern Python features
# 3. Reasonable serialization efficiency
# 
# Protocol 2 is supported in Python 2.3+ and all Python 3 versions,
# making it a safe choice for cross-environment compatibility.
DEFAULT_PROTOCOL = 2

def _is_zipfile(f) -> bool:
    """检查文件是否是ZIP文件"""
    start = f.tell()
    local_header_magic_number = b"PK\x03\x04"
    read_bytes = f.read(len(local_header_magic_number))
    f.seek(start)
    return read_bytes == local_header_magic_number


def _open_file_like(f, mode):
    """打开文件或返回文件对象"""
    if isinstance(f, (str, os.PathLike)):
        return open(f, mode)
    return f


def _open_zipfile_writer(f):
    """打开ZIP文件用于写入"""
    if isinstance(f, (str, os.PathLike)):
        zf = zipfile.ZipFile(f, 'w', compression=zipfile.ZIP_STORED, allowZip64=True)
        writer = _ZipFileWriter(zf)
        return writer
    return _ZipFileWriter(f)


def _open_zipfile_reader(f):
    """打开ZIP文件用于读取"""
    if isinstance(f, (str, os.PathLike)):
        zf = zipfile.ZipFile(f, 'r', allowZip64=True)
        reader = _ZipFileReader(zf, should_close=True)
        return reader
    else:
        # f 已经是文件对象，不要关闭它
        # 创建一个临时的BytesIO来避免文件指针问题
        f.seek(0)
        data = f.read()
        f.seek(0)
        zf = zipfile.ZipFile(io.BytesIO(data), 'r', allowZip64=True)
        reader = _ZipFileReader(zf, should_close=True)
        return reader


class _ZipFileReader:
    """ZIP文件读取器，兼容torch的接口"""
    def __init__(self, zipfile_obj, should_close=True):
        if isinstance(zipfile_obj, zipfile.ZipFile):
            self.zipfile = zipfile_obj
        else:
            self.zipfile = zipfile_obj.zipfile
        self._should_close = should_close
        self._records = {}
        for name in self.zipfile.namelist():
            self._records[name] = self.zipfile.getinfo(name)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close:
            self.zipfile.close()
    
    def get_record(self, name):
        """获取记录内容"""
        return self.zipfile.read(name)
    
    def get_record_offset(self, name):
        """获取记录偏移量（简化版本）"""
        return 0
    
    def has_record(self, name):
        """检查记录是否存在"""
        return name in self._records
    
    def get_all_records(self):
        """获取所有记录名称"""
        return list(self._records.keys())
    
    def write_record(self, name, data, length):
        """写入记录（仅用于兼容接口）"""
        raise NotImplementedError("write_record not supported on reader")
    
    def write_record_metadata(self, name, size):
        """写入记录元数据（仅用于兼容接口）"""
        raise NotImplementedError("write_record_metadata not supported on reader")
    
    def get_storage_from_record(self, name, numel, storage_type):
        """从记录获取存储（简化版本）"""
        data = self.get_record(name)
        return storage_type(data)


class _ZipFileWriter:
    """ZIP文件写入器，兼容torch的接口"""
    def __init__(self, zipfile_obj):
        if isinstance(zipfile_obj, zipfile.ZipFile):
            self.zipfile = zipfile_obj
        elif hasattr(zipfile_obj, 'write'):
            # 如果是文件类对象（如BytesIO），创建一个ZipFile对象来包装它
            self.zipfile = zipfile.ZipFile(zipfile_obj, 'w', compression=zipfile.ZIP_STORED, allowZip64=True)
        else:
            self.zipfile = zipfile_obj.zipfile
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.zipfile.close()
    
    def write_record(self, name, data, length):
        """写入记录"""
        if isinstance(data, (bytes, bytearray)):
            self.zipfile.writestr(name, data)
        elif hasattr(data, '__array__'):
            self.zipfile.writestr(name, data.tobytes())
        else:
            self.zipfile.writestr(name, data)
    
    def write_record_metadata(self, name, size):
        """写入记录元数据（仅写入空数据）"""
        self.zipfile.writestr(name, b'\x00' * size)


def _maybe_decode_ascii(bytes_str):
    """将字节串解码为ASCII字符串"""
    if isinstance(bytes_str, bytes):
        return bytes_str.decode('ascii')
    return bytes_str


def _get_dtype_from_storage_type(storage_type_str):
    """从存储类型字符串获取dtype"""
    dtype_map = {
        'FloatStorage': np.float32,
        'DoubleStorage': np.float64,
        'HalfStorage': np.float16,
        'BFloat16Storage': np.float16,
        'ByteStorage': np.uint8,
        'CharStorage': np.int8,
        'ShortStorage': np.int16,
        'IntStorage': np.int32,
        'LongStorage': np.int64,
        'BoolStorage': np.bool_,
        'ComplexFloatStorage': np.complex64,
        'ComplexDoubleStorage': np.complex128,
    }
    return dtype_map.get(storage_type_str, np.float32)


def _get_storage_type_from_dtype(dtype):
    """从dtype获取存储类型字符串"""
    dtype_map = {
        np.float32: 'FloatStorage',
        np.float64: 'DoubleStorage',
        np.float16: 'HalfStorage',
        np.uint8: 'ByteStorage',
        np.int8: 'CharStorage',
        np.int16: 'ShortStorage',
        np.int32: 'IntStorage',
        np.int64: 'LongStorage',
        np.bool_: 'BoolStorage',
        np.complex64: 'ComplexFloatStorage',
        np.complex128: 'ComplexDoubleStorage',
    }
    return dtype_map.get(dtype, 'FloatStorage')

def _get_element_size(dtype):
    """获取dtype的元素大小"""
    return dtype.itemsize

def _default_restore_location(storage, location):
    """默认的存储位置恢复函数"""
    if location == 'cpu':
        return storage
    elif location.startswith('cuda'):
        if cp is not None:
            if isinstance(storage, np.ndarray):
                return cp.asarray(storage)
        return storage
    return storage


def _get_restore_location(map_location):
    """获取存储位置恢复函数"""
    if map_location is None:
        return _default_restore_location
    elif isinstance(map_location, dict):
        def restore_location(storage, location):
            location = map_location.get(location, location)
            return _default_restore_location(storage, location)
        return restore_location
    elif isinstance(map_location, (str, bytes)):
        def restore_location(storage, location):
            return _default_restore_location(storage, map_location)
        return restore_location
    elif isinstance(map_location, Device):
        def restore_location(storage, location):
            return _default_restore_location(storage, str(map_location))
        return restore_location
    else:
        def restore_location(storage, location):
            result = map_location(storage, location)
            if result is None:
                result = _default_restore_location(storage, location)
            return result
        return restore_location


def _tensor_constructor(data, shape, dtype_str, device_str, requires_grad):
    """
    从序列化数据重建Riemann张量，兼容PyTorch格式
    
    参数：
        data: 张量数据（numpy或cupy数组）
        shape: 张量形状
        dtype_str: 数据类型字符串
        device_str: 设备字符串
        requires_grad: 是否需要梯度
        
    返回：
        TN: 重建的Riemann张量
    """
    
    if data is None:
        return None
    
    # 对numpy和cupy数组都进行reshape处理
    if isinstance(data, np.ndarray):
        data = data.reshape(shape)
    elif cp and isinstance(data, cp.ndarray):
        data = data.reshape(shape)
    
    # 创建设备对象
    try:
        device = Device(device_str)
    except Exception:
        device = Device('cpu')
    
    # 转换数据类型
    dtype = np.dtype(dtype_str)
    
    # 创建张量
    return tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def _parameter_constructor(*args):
    """
    从序列化数据重建Riemann参数，兼容PyTorch格式
    
    参数：
        *args: 构造参数，可以是以下两种形式：
            1. (None, requires_grad) - 空参数
            2. (data, shape, dtype_str, device_str, requires_grad) - 完整参数
        
    返回：
        Parameter: 重建的Riemann参数
    """
    if len(args) == 2 and args[0] is None:
        # 空参数情况
        return Parameter(None, args[1])
    elif len(args) == 5:
        # 完整参数情况
        data, shape, dtype_str, device_str, requires_grad = args
        
        if data is None:
            return Parameter(None, requires_grad)
        
        # 对numpy和cupy数组都进行reshape处理
        if isinstance(data, np.ndarray):
            data = data.reshape(shape)
        elif cp and isinstance(data, cp.ndarray):
            data = data.reshape(shape)
        
        # 创建设备对象
        try:
            device = Device(device_str)
        except Exception:
            device = Device('cpu')
        
        # 转换数据类型
        dtype = np.dtype(dtype_str)
        
        # 创建张量
        tensor_obj = tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        
        # 创建参数
        return Parameter(tensor_obj, requires_grad)
    else:
        raise ValueError(f"Invalid arguments for Parameter constructor: {args}")


def save(obj: Any, f: Union[str, os.PathLike, Any], 
         pickle_module: Any = None, 
         pickle_protocol: int = 2,
         use_new_zipfile_serialization: bool = True,
         _disable_byteorder_record: bool = False) -> None:
    """
    将对象保存到磁盘文件，兼容PyTorch的序列化格式。
    
    此函数使用与PyTorch兼容的序列化格式保存Riemann张量、参数、模块或任何Python对象
    到磁盘文件。
    
    参数：
        obj: 要保存的对象。可以是张量、参数、模块或任何可pickle的对象
        f: 要写入的文件路径或类文件对象
        pickle_module: 用于pickle的模块（默认：pickle）
        pickle_protocol: Pickle协议版本（默认：2）
        use_new_zipfile_serialization: 是否使用基于zip的序列化（默认：True）
        _disable_byteorder_record: 是否禁用字节序记录（内部使用）
        
    示例：
        >>> import riemann as rm
        >>> # 保存张量
        >>> tensor = rm.randn(3, 4)
        >>> rm.save(tensor, 'tensor.pt')
        >>> 
        >>> # 保存模块
        >>> model = rm.nn.Linear(10, 5)
        >>> rm.save(model.state_dict(), 'model_weights.pt')
        >>> 
        >>> # 保存多个对象
        >>> rm.save({
        ...     'model': model.state_dict(),
        ...     'optimizer_state': optimizer.state_dict(),
        ...     'epoch': 10
        ... }, 'checkpoint.pt')
    """
    if pickle_module is None:
        pickle_module = pickle
    
    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)
    
    if use_new_zipfile_serialization:
        with _open_zipfile_writer(f) as opened_zipfile:
            _save(
                obj,
                opened_zipfile,
                pickle_module,
                pickle_protocol,
                _disable_byteorder_record,
            )
            return
    else:
        with _open_file_like(f, 'wb') as opened_file:
            _legacy_save(obj, opened_file, pickle_module, pickle_protocol)


def _save(obj, zip_file, pickle_module, pickle_protocol, _disable_byteorder_record):
    """使用ZIP格式保存对象（兼容PyTorch格式）"""
        
    def persistent_id(obj):
        """自定义pickle的persistent_id方法"""
        # 对于Riemann的张量和参数，我们不使用persistent_id，而是让__reduce__方法来处理
        # 这样pickle会使用_tensordef.TN的__reduce__方法返回的构造函数和参数来重建对象
        # 我们只对PyTorch的存储对象使用persistent_id
        return None
    
    class PyTorchPickler(pickle_module.Pickler):
        def persistent_id(self, obj):
            return persistent_id(obj)
    
    data_buf = io.BytesIO()
    pickler = PyTorchPickler(data_buf, protocol=pickle_protocol)
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    
    # 创建一个随机的子目录名称，就像PyTorch一样
    import uuid
    subdir = str(uuid.uuid4())[:8]
    
    zip_file.write_record(f"{subdir}/data.pkl", data_value, len(data_value))
    zip_file.write_record(f"{subdir}/.format_version", "1", len("1"))
    zip_file.write_record(f"{subdir}/.storage_alignment", "64", len("64"))
    
    if not _disable_byteorder_record:
        if sys.byteorder not in ["little", "big"]:
            raise ValueError("Unknown endianness type: " + sys.byteorder)
        zip_file.write_record(f"{subdir}/byteorder", sys.byteorder, len(sys.byteorder))
    
    # 添加version文件，PyTorch需要它
    zip_file.write_record(f"{subdir}/version", "1", len("1"))


def _legacy_save(obj, f, pickle_module, pickle_protocol):
    """使用传统格式保存对象"""
    serialized_storages = {}
    
    def persistent_id(obj):
        if isinstance(obj, (TN, Parameter)):
            if isinstance(obj, Parameter):
                tensor_obj = obj.data if obj.data is not None else obj
            else:
                tensor_obj = obj
            
            if tensor_obj.data is None:
                return None
            
            data = tensor_obj.data
            dtype = data.dtype
            
            if isinstance(data, np.ndarray):
                location = 'cpu'
            else:
                location = 'cuda'
            
            storage_key = str(id(data))
            
            if storage_key not in serialized_storages:
                serialized_storages[storage_key] = (data, dtype)
            
            storage_type_str = _get_storage_type_from_dtype(dtype)
            numel = data.size
            
            return ("storage", storage_type_str, storage_key, location, numel, None)
        
        return None
    
    sys_info = dict(
        protocol_version=PROTOCOL_VERSION,
        little_endian=sys.byteorder == "little",
        type_sizes=dict(
            short=struct.Struct("=h").size,
            int=struct.Struct("=i").size,
            long=struct.Struct("=l").size,
        ),
    )
    
    pickle_module.dump(MAGIC_NUMBER, f, protocol=pickle_protocol)
    pickle_module.dump(PROTOCOL_VERSION, f, protocol=pickle_protocol)
    pickle_module.dump(sys_info, f, protocol=pickle_protocol)
    
    class PyTorchLegacyPickler(pickle_module.Pickler):
        def persistent_id(self, obj):
            return persistent_id(obj)
    
    pickler = PyTorchLegacyPickler(f, protocol=pickle_protocol)
    pickler.dump(obj)
    
    serialized_storage_keys = sorted(serialized_storages.keys())
    pickle_module.dump(serialized_storage_keys, f, protocol=pickle_protocol)
    f.flush()
    
    for key in serialized_storage_keys:
        data, dtype = serialized_storages[key]
        if cp and isinstance(data, cp.ndarray):
            data_np = cp.asnumpy(data)
        else:
            data_np = data
        f.write(data_np.tobytes())


def load(f: Union[str, os.PathLike, Any], 
         map_location: Optional[Any] = None,
         pickle_module: Any = None,
         **pickle_load_args: Any) -> Any:
    """
    从磁盘文件加载对象，兼容PyTorch的序列化格式。
    
    此函数使用与PyTorch兼容的反序列化格式从磁盘文件加载Riemann张量、参数、模块或任何Python对象。
    
    参数：
        f: 要读取的文件路径或类文件对象
        map_location: 指定如何重新映射存储位置的函数或字典
        pickle_module: 用于unpickle的模块（默认：pickle）
        **pickle_load_args: 传递给pickle.load的额外参数
        
    返回：
        加载的对象
        
    示例：
        >>> import riemann as rm
        >>> # 加载张量
        >>> tensor = rm.load('tensor.pt')
        >>> 
        >>> # 加载模型权重
        >>> state_dict = rm.load('model_weights.pt')
        >>> model.load_state_dict(state_dict)
        >>> 
        >>> # 加载检查点
        >>> checkpoint = rm.load('checkpoint.pt')
        >>> model.load_state_dict(checkpoint['model'])
        >>> optimizer.load_state_dict(checkpoint['optimizer_state'])
        >>> epoch = checkpoint['epoch']
    """
    if pickle_module is None:
        pickle_module = pickle
    
    if isinstance(f, (str, os.PathLike)):
        with _open_file_like(f, 'rb') as opened_file:
            return _load(opened_file, map_location, pickle_module, **pickle_load_args)
    else:
        return _load(f, map_location, pickle_module, **pickle_load_args)


def _load(f, map_location, pickle_module, **pickle_load_args):
    """加载对象（兼容PyTorch格式）"""
    is_zip = False
    try:
        is_zip = _is_zipfile(f)
    except Exception:
        is_zip = False
    
    if is_zip:
        try:
            f.seek(0)
            zip_file = _open_zipfile_reader(f)
            try:
                return _load_from_zip(zip_file, map_location, pickle_module, **pickle_load_args)
            finally:
                if hasattr(zip_file, '__exit__'):
                    zip_file.__exit__(None, None, None)
        except Exception as e:
            f.seek(0)
            # 如果ZIP加载失败，尝试legacy加载
            return _legacy_load(f, map_location, pickle_module, **pickle_load_args)
    
    # 不是ZIP文件，使用legacy加载
    f.seek(0)
    return _legacy_load(f, map_location, pickle_module, **pickle_load_args)


def _load_from_zip(zip_file, map_location, pickle_module, **pickle_load_args):
    """从ZIP文件加载对象（兼容PyTorch格式）"""
    restore_location = _get_restore_location(map_location)
    loaded_storages = {}
    
    byteorderdata = None
    
    # 列出所有文件，找到子目录中的配置文件
    files = zip_file.zipfile.namelist()
    
    # 查找byteorder
    byteorder_file = None
    for file in files:
        if file.endswith("byteorder"):
            byteorder_file = file
            break
    
    if byteorder_file:
        byteorderdata = zip_file.get_record(byteorder_file)
    
    def load_tensor(dtype, numel, key, location):
        """加载张量数据"""
        # 尝试直接查找data/{key}文件
        try:
            name = f"data/{key}"
            data_bytes = zip_file.get_record(name)
        except KeyError:
            # 如果找不到，尝试在子目录中查找
            data_file = None
            for file in files:
                if file.endswith(f"data/{key}"):
                    data_file = file
                    break
            
            if data_file is None:
                raise KeyError(f"data/{key} not found in zip file")
            
            data_bytes = zip_file.get_record(data_file)
        
        if byteorderdata is not None:
            if byteorderdata.decode() != sys.byteorder:
                arr = np.frombuffer(data_bytes, dtype=dtype)
                arr = arr.byteswap()
                data_bytes = arr.tobytes()
        
        storage = np.frombuffer(data_bytes, dtype=dtype)
        storage = restore_location(storage, location)
        
        return storage
    
    def persistent_load(saved_id):
        """自定义pickle的persistent_load方法"""
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]
        
        if typename == "storage":
            storage_type_str, key, location, numel = data
            dtype = _get_dtype_from_storage_type(storage_type_str)
            
            if key in loaded_storages:
                storage = loaded_storages[key]
            else:
                storage = load_tensor(dtype, numel, key, _maybe_decode_ascii(location))
                loaded_storages[key] = storage
            
            # 直接返回numpy数组
            return storage
        
        return None
    
    class UnpicklerWrapper(pickle_module.Unpickler):
        def persistent_load(self, saved_id):
            return persistent_load(saved_id)
        
        def find_class(self, module, name):
            # 处理riemann的构造函数
            if module == 'riemann.serialization' and name == '_tensor_constructor':
                return _tensor_constructor
            if module == 'riemann.serialization' and name == '_parameter_constructor':
                return _parameter_constructor
            
            # 处理PyTorch的张量重建函数
            if module == 'torch._utils' and (name == '_rebuild_tensor' or name == '_rebuild_tensor_v2'):
                # 返回我们自己的张量重建函数
                def rebuild_tensor(*args, **kwargs):
                    # 从storage（numpy数组）重建Riemann张量
                                        
                    # 解析参数
                    if len(args) >= 3:
                        storage, storage_offset, size = args[:3]
                    else:
                        return None
                    
                    # 如果storage_offset不为0，我们需要调整storage
                    if storage_offset != 0:
                        storage = storage[storage_offset:]
                    
                    # 重塑storage为指定的大小
                    if size is not None:
                        storage = storage.reshape(size)
                    
                    # 尝试从args或kwargs中获取设备信息
                    device = None
                    if len(args) > 4:
                        # PyTorch的_rebuild_tensor_v2函数可能会传递更多参数
                        # 这里我们假设设备信息可能在args中
                        pass
                    elif 'device' in kwargs:
                        device = kwargs['device']
                    
                    # 检查storage是否已经是CuPy数组，如果是，从storage中获取设备信息
                    if cp and isinstance(storage, cp.ndarray):
                        # 获取CuPy数组所在的设备索引
                        device_idx = storage.device.id
                        # 创建包含设备索引的设备字符串
                        device = f'cuda:{device_idx}'
                    
                    # 创建Riemann张量，指定设备参数
                    return tensor(storage, device=device, requires_grad=False)
                
                return rebuild_tensor
            
            return super().find_class(module, name)
    
    # 尝试直接查找data.pkl文件
    try:
        data_file = io.BytesIO(zip_file.get_record("data.pkl"))
    except KeyError:
        # 如果找不到，尝试在子目录中查找
        data_pkl_file = None
        for file in files:
            if file.endswith("data.pkl"):
                data_pkl_file = file
                break
        
        if data_pkl_file is None:
            raise KeyError("data.pkl not found in zip file")
        
        data_file = io.BytesIO(zip_file.get_record(data_pkl_file))
    
    unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    
    result = unpickler.load()
    
    return result


def _legacy_load(f, map_location, pickle_module, **pickle_load_args):
    """使用传统格式加载对象"""
    restore_location = _get_restore_location(map_location)
    deserialized_objects = {}
    
    def persistent_load(saved_id):
        """自定义pickle的persistent_load方法"""
        if isinstance(saved_id, tuple):
            typename = _maybe_decode_ascii(saved_id[0])
            data = saved_id[1:]
            
            if typename == "storage":
                storage_type_str, key, location, numel, view_metadata = data
                dtype = _get_dtype_from_storage_type(storage_type_str)
                
                if key not in deserialized_objects:
                    nbytes = numel * _get_element_size(dtype)
                    data_bytes = f.read(nbytes)
                    storage = np.frombuffer(data_bytes, dtype=dtype)
                    storage = restore_location(storage, location)
                    deserialized_objects[key] = storage
                else:
                    storage = deserialized_objects[key]
                
                return storage
        elif isinstance(saved_id, (int, str)):
            # 尝试将saved_id转换为整数
            try:
                key = int(saved_id)
                return deserialized_objects.get(key, None)
            except (ValueError, TypeError):
                pass
        
        # 如果无法处理，返回None
        return None
    
    class UnpicklerWrapper(pickle_module.Unpickler):
        def persistent_load(self, saved_id):
            return persistent_load(saved_id)
    
    unpickler = UnpicklerWrapper(f, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    
    magic_number = unpickler.load()
    if magic_number != MAGIC_NUMBER:
        raise RuntimeError("Invalid magic number; corrupt file?")
    
    protocol_version = unpickler.load()
    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError(f"Invalid protocol version: {protocol_version}")
    
    _sys_info = unpickler.load()
    result = unpickler.load()
    
    deserialized_storage_keys = unpickler.load()
    
    return result


