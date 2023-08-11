import dolfinx
import numba
import numpy as np
from .setup_types import ffi, PETSc
from . import utils


def assemble_scalar(form, qr_data, perm):
    vertices, coords, _ = utils.get_vertices(form.mesh)
    integral_ids = form.integral_ids(dolfinx.cpp.fem.IntegralType.cell)
    all_coeffs = dolfinx.cpp.fem.pack_coefficients(form)
    consts = dolfinx.cpp.fem.pack_constants(form)

    m = np.zeros(1, dtype=PETSc.ScalarType)

    for i, id in enumerate(integral_ids):
        kernel = getattr(
            form.ufcx_form.integrals(dolfinx.cpp.fem.IntegralType.cell)[i],
            "tabulate_tensor_runtime_float64",
        )

        coeffs = all_coeffs[(dolfinx.cpp.fem.IntegralType.cell, id)]

        assemble_cells(m, kernel, vertices, coords, coeffs, consts, qr_data[i], perm)

    return m[0]


@numba.njit  # (fastmath=True)
def assemble_cells(m, kernel, vertices, coords, coeffs, consts, qr, perm):
    # Unpack qr
    if len(qr) == 3:
        cells, qr_pts, qr_w = qr
        qr_n = qr_pts  # dummy
    else:
        cells, qr_pts, qr_w, qr_n = qr
        assert len(cells) == len(qr_n)

    assert len(cells) == len(qr_pts)
    assert len(cells) == len(qr_w)

    # # This is needed if running w/o numba. With numba, if
    # len(coeffs) == 0, the coeffs[cell] is still ok
    # if len(coeffs) = 0:
    #   coeffs = [] * len(cells)

    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    m_local = np.zeros(1, dtype=PETSc.ScalarType)
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array(perm, dtype=np.uint8)

    for k, cell in enumerate(cells):
        cell_coords[:, :] = coords[vertices[cell, :]]
        num_quadrature_points = len(qr_w[k])
        m_local.fill(0.0)

        kernel(
            ffi.from_buffer(m_local),
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

        m[0] += m_local[0]
