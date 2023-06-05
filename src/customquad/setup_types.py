# Copied from dolfinx/python/test/unit/fem/test_custom_assembler.py

# Copyright (C) 2019-2020 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for custom Python assemblers"""

import ctypes
import ctypes.util
import importlib
import os
import pathlib

import cffi
import numba
import numba.core.typing.cffi_utils as cffi_support
import numpy as np
import numpy.typing

import dolfinx
import dolfinx.pkgconfig
from dolfinx.fem.petsc import load_petsc_lib

import petsc4py.lib
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py import get_config as PETSc_get_config

# Get details of PETSc install
petsc_dir = PETSc_get_config()['PETSC_DIR']
petsc_arch = petsc4py.lib.getPathArchPETSc()[1]

# Get PETSc int and scalar types
if np.dtype(PETSc.ScalarType).kind == 'c':
    complex = True
else:
    complex = False

scalar_size = np.dtype(PETSc.ScalarType).itemsize
index_size = np.dtype(PETSc.IntType).itemsize

if index_size == 8:
    c_int_t = "int64_t"
    ctypes_index: numpy.typing.DTypeLike = ctypes.c_int64
elif index_size == 4:
    c_int_t = "int32_t"
    ctypes_index = ctypes.c_int32
else:
    raise RuntimeError(f"Cannot translate PETSc index size into a C type, index_size: {index_size}.")

if complex and scalar_size == 16:
    c_scalar_t = "double _Complex"
    numba_scalar_t = numba.types.complex128
elif complex and scalar_size == 8:
    c_scalar_t = "float _Complex"
    numba_scalar_t = numba.types.complex64
elif not complex and scalar_size == 8:
    c_scalar_t = "double"
    numba_scalar_t = numba.types.float64
elif not complex and scalar_size == 4:
    c_scalar_t = "float"
    numba_scalar_t = numba.types.float32
else:
    raise RuntimeError(
        f"Cannot translate PETSc scalar type to a C type, complex: {complex} size: {scalar_size}.")


petsc_lib_ctypes = load_petsc_lib(ctypes.cdll.LoadLibrary)
# Get the PETSc MatSetValuesLocal function via ctypes
# ctypes does not support static types well, ignore type check errors
MatSetValues_ctypes = petsc_lib_ctypes.MatSetValuesLocal
MatSetValues_ctypes.argtypes = [ctypes.c_void_p, ctypes_index, ctypes.POINTER(  # type: ignore
    ctypes_index), ctypes_index, ctypes.POINTER(ctypes_index), ctypes.c_void_p, ctypes.c_int]  # type: ignore
del petsc_lib_ctypes


# CFFI - register complex types
ffi = cffi.FFI()
cffi_support.register_type(ffi.typeof('double _Complex'), numba.types.complex128)
cffi_support.register_type(ffi.typeof('float _Complex'), numba.types.complex64)

# Get MatSetValuesLocal from PETSc available via cffi in ABI mode
ffi.cdef("""int MatSetValuesLocal(void* mat, {0} nrow, const {0}* irow,
                                  {0} ncol, const {0}* icol, const {1}* y, int addv);
""".format(c_int_t, c_scalar_t))


petsc_lib_cffi = load_petsc_lib(ffi.dlopen)
MatSetValues_abi = petsc_lib_cffi.MatSetValuesLocal


# @pytest.fixture
def get_matsetvalues_api():
    """Make MatSetValuesLocal from PETSc available via cffi in API mode"""
    if dolfinx.pkgconfig.exists("dolfinx"):
        dolfinx_pc = dolfinx.pkgconfig.parse("dolfinx")
    else:
        raise RuntimeError("Could not find DOLFINx pkgconfig file")

    worker = os.getenv('PYTEST_XDIST_WORKER', None)
    module_name = "_petsc_cffi_{}".format(worker)
    if MPI.COMM_WORLD.Get_rank() == 0:
        ffibuilder = cffi.FFI()
        ffibuilder.cdef("""
            typedef int... PetscInt;
            typedef ... PetscScalar;
            typedef int... InsertMode;
            int MatSetValuesLocal(void* mat, PetscInt nrow, const PetscInt* irow,
                                PetscInt ncol, const PetscInt* icol,
                                const PetscScalar* y, InsertMode addv);
        """)
        ffibuilder.set_source(module_name, """
            #include "petscmat.h"
        """,
                              libraries=['petsc'],
                              include_dirs=[os.path.join(petsc_dir, petsc_arch, 'include'),
                                            os.path.join(petsc_dir, 'include')] + dolfinx_pc["include_dirs"],
                              library_dirs=[os.path.join(petsc_dir, petsc_arch, 'lib')],
                              extra_compile_args=[])

        # Build module in same directory as test file
        # path = pathlib.Path(__file__).parent.absolute()
        path = pathlib.Path(os.getcwd())

        ffibuilder.compile(tmpdir=path, verbose=True)

    MPI.COMM_WORLD.Barrier()

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError("Failed to find CFFI generated module")
    module = importlib.util.module_from_spec(spec)

    cffi_support.register_module(module)
    cffi_support.register_type(module.ffi.typeof("PetscScalar"), numba_scalar_t)
    return module.lib.MatSetValuesLocal


# See https://github.com/numba/numba/issues/4036 for why we need 'sink'
@numba.njit
def sink(*args):
    pass
