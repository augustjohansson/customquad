import dolfinx
import numba
import numpy as np
from .setup_types import ffi, PETSc
from . import utils


def assemble_vector(form, qr_data):
    # qr_data is a list of tuples containing (cells, qr_pts, qr_w,
    # qr_n) (both for volume and surface integrals). Here, each of
    # qr_pts, qr_w and qr_n should be list(numpy.array) with len(list)
    # == number of cells

    # FIXME: need to match qr_data list with correct integral_ids. A
    # custom integral have integral_ids == -1.

    # See also https://gist.github.com/IgorBaratta/d0b84fd5d77f2628204097a1c0b180fb

    V = form.function_spaces[0]
    dofs, num_loc_dofs = utils.get_dofs(V)
    vertices, coords, gdim = utils.get_vertices(V.mesh)

    integral_ids = form.integral_ids(dolfinx.cpp.fem.IntegralType.cell)
    all_coeffs = dolfinx.cpp.fem.pack_coefficients(form)
    consts = dolfinx.cpp.fem.pack_constants(form)

    b = dolfinx.cpp.la.petsc.create_vector(V.dofmap.index_map, V.dofmap.index_map_bs)

    for i, id in enumerate(integral_ids):
        kernel = getattr(
            form.ufcx_form.integrals(dolfinx.cpp.fem.IntegralType.cell)[i],
            "tabulate_tensor_runtime_float64",
        )

        coeffs = all_coeffs[(dolfinx.cpp.fem.IntegralType.cell, id)]

        assemble_cells(
            b,
            kernel,
            vertices,
            coords,
            dofs,
            num_loc_dofs,
            coeffs,
            consts,
            qr_data[i],
        )

    return b


@numba.njit  # (fastmath=True)
def assemble_cells(b, kernel, vertices, coords, dofs, num_loc_dofs, coeffs, consts, qr):
    # Unpack qr
    if len(qr) == 3:
        cells, qr_pts, qr_w = qr
        qr_n = qr_pts  # dummy
    else:
        cells, qr_pts, qr_w, qr_n = qr
        assert len(cells) == len(qr_n)

    assert len(cells) == len(qr_pts)
    assert len(cells) == len(qr_w)

    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    b_local = np.zeros(num_loc_dofs, dtype=PETSc.ScalarType)
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array([0], dtype=np.uint8)

    for k, cell in enumerate(cells):
        cell_coords[:, :] = coords[vertices[cell, :]]
        num_quadrature_points = len(qr_w[k])
        b_local.fill(0.0)

        kernel(
            ffi.from_buffer(b_local),
            ffi.from_buffer(coeffs[cell]),
            ffi.from_buffer(consts),
            ffi.from_buffer(cell_coords),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
            num_quadrature_points,
            ffi.from_buffer(qr_pts[k]),
            ffi.from_buffer(qr_w[k]),
            ffi.from_buffer(qr_n[k]),
        )

        # FIXME: Change to petsc set_values_local from setup_types?
        for j in range(num_loc_dofs):
            b[dofs[cell, j]] += b_local[j]
