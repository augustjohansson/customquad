import dolfinx
import dolfinx.log

import numba
import numpy

from .setup_types import ffi, PETSc
from . import utils

def assemble_vector():
    pass


def custom_assemble_vector(form, qr_data):
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

    # Dofs
    dofs, num_loc_dofs = utils.get_dofs(V)

    # Vertices
    vertices, coords, gdim = utils.get_vertices(V.mesh)

    # Form data
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = dolfinx.cpp.fem.pack_constants(cpp_form)

    # Assembly
    ufc_form = dolfinx.jit.ffcx_jit(form)
    b = dolfinx.cpp.la.create_vector(V.dofmap.index_map)

    with b.localForm() as b_local:
        b_local.set(0.0)
        qr_n = []

        for k, integral_id in enumerate(integral_ids):
            # Get kernel
            kernel = ufc_form.create_custom_integral(integral_id).tabulate_tensor

            # Get qr data for this kernel
            qd = qr_data[k]
            cells = qd[0]
            qr_pts = qd[1]
            qr_w = qd[2]
            if len(qd) == 4:
                qr_n = qd[3]
            qr_pts, qr_w, qr_n, cells = utils.check_qr(qr_pts, qr_w, qr_n, cells)

            local_assemble_vector(numpy.asarray(b_local),
                                  kernel,
                                  (vertices, coords, gdim),
                                  (dofs, num_loc_dofs),
                                  (form_coeffs, form_consts),
                                  (qr_pts, qr_w, qr_n, cells))
            b_local.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return b



@numba.njit#(fastmath=True)
def local_assemble_vector(b, kernel, mesh, dofmap, form_data, qr):

    # Unpack
    v, x, gdim = mesh
    dofs, num_loc_dofs = dofmap
    coeffs, constants = form_data
    qr_pts, qr_w, qr_n, cells = qr

    # Initialize
    num_loc_vertices = v.shape[1]
    cell_coords = numpy.zeros((num_loc_vertices, gdim))
    b_local = numpy.zeros(num_loc_dofs, dtype=PETSc.ScalarType)

    for cell in cells:
        for j in range(num_loc_vertices):
            cell_coords[j] = x[v[cell, j], 0:gdim]
        # print("in the assembler:")
        # print(cell_coords)
        # print(qr_pts[cell])
        # print(qr_w[cell])
        # print(qr_n[cell])
        # utils.print_for_header(b_local,coeffs[cell],constants,cell_coords,len(qr_w[cell]),qr_pts[cell],qr_w[cell],qr_n[cell])
        # import ipdb; ipdb.set_trace()

        b_local.fill(0.0)
        kernel(ffi.from_buffer(b_local),
               ffi.from_buffer(coeffs[cell]),
               ffi.from_buffer(constants),
               ffi.from_buffer(cell_coords),
               len(qr_w[cell]),
               ffi.from_buffer(qr_pts[cell]),
               ffi.from_buffer(qr_w[cell]),
               ffi.from_buffer(qr_n[cell]))

        # print("after assemble", b_local)
        # import ipdb; ipdb.set_trace()

        for j in range(num_loc_dofs):
            b[dofs[cell, j]] += b_local[j]
