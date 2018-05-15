"""
Implementation of support for ctypes library
"""

from numba.targets.imputils import Registry
from numba import types, cgutils
from numba.datamodel.models import StructModel
from numba.datamodel import register_default
from numba.typing.ctypes_utils import from_ctypes, _CData
from numba.typing.typeof import typeof_impl
from numba.pythonapi import box, unbox, NativeValue

registry = Registry()
_struct_cache = {}


class CDataType(types.Type):
    def __init__(self):
        super(CDataType, self).__init__(name='_CData')


class StructureType(CDataType):
    def __init__(self, members, packed=True):
        self.members = members
        self.packed = packed
        super(StructureType, self).__init__(name='Structure')

cdata_type = CDataType()

@typeof_impl.register(_CData)
def typeof_cdata(val, c):
    t = from_ctypes(val.__class__)
    if t is None and hasattr(val, '_fields_'):
        # Generate new type instance representing struct with members
        t = _struct_cache.get(val.__class__, None)
        if t is None:
            # TODO support anonymous and nested structures
            members = [None] * len(val._fields_)
            for i, field in enumerate(val._fields_):
                if len(field) > 2:
                    raise NotImplementedError("Bitwidth specification not supported")
                members[i] = (field[0], from_ctypes(field[1]))

            t = StructureType(members, getattr(val, '_pack_', 0) == 1)
            _struct_cache[val.__class__] = t
    return t

@register_default(StructureType)
class StructureModel(StructModel):
    def __init__(self, dmm, fe_type):
        super(StructModel, self).__init__(self, dmm, fe_type, fe_type.members, packed=fe_type.packed)

@box(CDataType)
def box_cdata(typ, obj, c):
    pass

@unbox(CDataType)
def unbox_cdata(typ, obj, c):
    cdata = cgutils.create_struct_proxy(typ, 'data')(c.context, c.builder) # ref= )
    return NativeValue(cdata, cgutils.is_not_null(c.builder, c.pyapi.err_occurred()))