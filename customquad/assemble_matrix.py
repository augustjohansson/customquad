import dolfinx
import dolfinx.log

import numba
import numpy

from .setup_types import ffi, PETSc, sink, get_matsetvalues_api
from . import utils

def assemble_matrix():
    pass

def custom_assemble_matrix(form, qr_data):
    # qr_data is a list of tuples containing (cells, qr_pts, qr_w,
    # qr_n) (both for volume and surface integrals). Here, each of
    # qr_pts, qr_w and qr_n should be list(numpy.array) with len(list)
    # == number of cells

    # FIXME: need to match qr_data list with correct integral_ids. A
    # custom integral have integral_ids == -1.

    # Form
    cpp_form = dolfinx.Form(form)._cpp_object
    integral_ids = cpp_form.integrals.integral_ids(dolfinx.fem.IntegralType.custom)
    assert len(integral_ids) == len(qr_data)

    # Function space
    V = form.arguments()[0].ufl_function_space()
    assert V == form.arguments()[1].ufl_function_space()

    # Vertices
    vertices, coords, gdim = utils.get_vertices(V.mesh)

    # Form data
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = dolfinx.cpp.fem.pack_constants(cpp_form)

    # Assembly
    ufc_form = dolfinx.jit.ffcx_jit(form)
    matsetvalueslocal = get_matsetvalues_api()
    mode = PETSc.InsertMode.ADD_VALUES
    A = dolfinx.cpp.fem.create_matrix(cpp_form)
    A.zeroEntries()

    for k, integral_id in enumerate(integral_ids):
        print(k, integral_id)

        # Get kernel
        kernel = ufc_form.create_custom_integral(integral_id).tabulate_tensor

        # Get qr data for this kernel
        qd = qr_data[k]
        cells = qd[0]
        qr_pts = qd[1]
        qr_w = qd[2]
        qr_n = []
        if len(qd) == 4:
            qr_n = qd[3]
        qr_pts, qr_w, qr_n, cells = utils.check_qr(qr_pts, qr_w, qr_n, cells)

        # Dofs
        dofs, num_loc_dofs = utils.get_dofs(V)

        local_assemble_matrix(A.handle,
                              kernel,
                              (vertices, coords, gdim),
                              (dofs, num_loc_dofs),
                              (form_coeffs, form_consts),
                              (qr_pts, qr_w, qr_n, cells),
                              matsetvalueslocal,
                              mode)
    return A


@numba.njit#(fastmath=True)
def local_assemble_matrix(A_handle, kernel, mesh, dofmap, form_data, qr, matsetvalueslocal, mode):

    # Unpack
    v, x, gdim = mesh
    dofs, num_loc_dofs = dofmap
    coeffs, constants = form_data
    qr_pts, qr_w, qr_n, cells = qr

    # Initialize
    num_loc_vertices = v.shape[1]
    cell_coords = numpy.zeros((num_loc_vertices, gdim))
    A_local = numpy.empty((num_loc_dofs, num_loc_dofs), dtype=PETSc.ScalarType)

    for cell in cells:
        for j in range(num_loc_vertices):
            cell_coords[j] = x[v[cell, j], 0:gdim]

        A_local[:] = 0.0
        kernel(ffi.from_buffer(A_local),
               ffi.from_buffer(coeffs[cell, :]),
               ffi.from_buffer(constants),
               ffi.from_buffer(cell_coords),
               len(qr_w[cell]),
               ffi.from_buffer(qr_pts[cell]),
               ffi.from_buffer(qr_w[cell]),
               ffi.from_buffer(qr_n[cell]))
        pos = dofs[cell, :]

        matsetvalueslocal(A_handle,
                          num_loc_dofs, ffi.from_buffer(pos),
                          num_loc_dofs, ffi.from_buffer(pos),
                          ffi.from_buffer(A_local),
                          mode)
    sink(A_local, dofs)
