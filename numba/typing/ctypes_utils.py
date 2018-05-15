"""
Support for typing ctypes function pointers.
"""

from __future__ import absolute_import

import ctypes
import sys

from numba import types
from . import templates

_CData = ctypes._SimpleCData.__bases__[0]

_make_unsigned = lambda x: getattr(types, 'uint{}'.format(ctypes.sizeof(x) * 8))

_FROM_CTYPES = {
    ctypes.c_bool: types.boolean,

    ctypes.c_int: types.int_,
    ctypes.c_int8:  types.int8,
    ctypes.c_int16: types.int16,
    ctypes.c_int32: types.int32,
    ctypes.c_int64: types.int64,

    ctypes.c_uint: types.uint,
    ctypes.c_uint8: types.uint8,
    ctypes.c_uint16: types.uint16,
    ctypes.c_uint32: types.uint32,
    ctypes.c_uint64: types.uint64,

    ctypes.c_float: types.float32,
    ctypes.c_double: types.float64,
    ctypes.c_longdouble: types.longdouble,

    ctypes.c_ubyte: types.byte,
    ctypes.c_byte: types.byte,

    ctypes.c_short: types.short,
    ctypes.c_ushort: types.ushort,

    ctypes.c_long: types.long_,
    ctypes.c_longlong: types.longlong,

    ctypes.c_ulong: types.ulong,
    ctypes.c_ulonglong: types.ulonglong,

    ctypes.c_ssize_t: _make_unsigned(ctypes.c_ssize_t),
    ctypes.c_size_t: _make_unsigned(ctypes.c_size_t),
    # ctypes.HRESULT: types.void,  # TODO os dependant

    ctypes.c_char: types.char,
    ctypes.c_char_p: types.CPointer(types.char),
    ctypes.c_wchar: _make_unsigned(ctypes.c_wchar),  # TODO not ideal representation https://github.com/numpy/numpy/issues/10100
    ctypes.c_wchar_p: types.CPointer(_make_unsigned(ctypes.c_wchar)),

    ctypes.c_void_p: types.voidptr,
    ctypes.py_object: types.ffi_forced_object,
}

_TO_CTYPES = {v: k for (k, v) in _FROM_CTYPES.items()}


def from_ctypes(ctypeobj):
    """
    Convert the given ctypes type to a Numba type.
    """
    if ctypeobj is None:
        # Special case for the restype of void-returning functions
        return types.none

    assert isinstance(ctypeobj, type), ctypeobj

    def _convert_internal(ctypeobj):
        # Recursive helper
        if issubclass(ctypeobj, ctypes._Pointer):
            valuety = _convert_internal(ctypeobj._type_)
            if valuety is not None:
                return types.CPointer(valuety)
        else:
            return _FROM_CTYPES.get(ctypeobj)

    ty = _convert_internal(ctypeobj)
    if ty is None:
        raise TypeError("Unsupported ctypes type: %s" % ctypeobj)
    return ty


def to_ctypes(ty):
    """
    Convert the given Numba type to a ctypes type.
    """
    assert isinstance(ty, types.Type), ty

    if ty is types.none:
        # Special case for the restype of void-returning functions
        return None

    def _convert_internal(ty):
        if isinstance(ty, types.CPointer):
            return ctypes.POINTER(_convert_internal(ty.dtype))
        else:
            return _TO_CTYPES.get(ty)

    ctypeobj = _convert_internal(ty)
    if ctypeobj is None:
        raise TypeError("Cannot convert Numba type '%s' to ctypes type"
                        % (ty,))
    return ctypeobj


def is_ctypes_funcptr(obj):
    try:
        # Is it something of which we can get the address
        ctypes.cast(obj, ctypes.c_void_p)
    except ctypes.ArgumentError:
        return False
    else:
        # Does it define argtypes and restype
        return hasattr(obj, 'argtypes') and hasattr(obj, 'restype')


def get_pointer(ctypes_func):
    """
    Get a pointer to the underlying function for a ctypes function as an
    integer.
    """
    return ctypes.cast(ctypes_func, ctypes.c_void_p).value


def make_function_type(cfnptr):
    """
    Return a Numba type for the given ctypes function pointer.
    """
    if cfnptr.argtypes is None:
        raise TypeError("ctypes function %r doesn't define its argument types; "
                        "consider setting the `argtypes` attribute"
                        % (cfnptr.__name__,))
    cargs = [from_ctypes(a)
             for a in cfnptr.argtypes]
    cret = from_ctypes(cfnptr.restype)
    if sys.platform == 'win32' and not cfnptr._flags_ & ctypes._FUNCFLAG_CDECL:
        # 'stdcall' calling convention under Windows
        cconv = 'x86_stdcallcc'
    else:
        # Default C calling convention
        cconv = None

    sig = templates.signature(cret, *cargs)
    return types.ExternalFunctionPointer(sig, cconv=cconv,
                                         get_pointer=get_pointer)
