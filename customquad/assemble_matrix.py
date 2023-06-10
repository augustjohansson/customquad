import dolfinx
import numba
import numpy as np
from .setup_types import ffi, PETSc, sink, get_matsetvalues_api
from . import utils

# See assemble_matrix_cffi in test_custom_assembler


def assemble_matrix(form, qr_data):
    V = form.function_spaces[0]
    dofs, num_loc_dofs = utils.get_dofs(V)
    vertices, coords, gdim = utils.get_vertices(V.mesh)

    integral_ids = form.integral_ids(dolfinx.cpp.fem.IntegralType.cell)
    all_coeffs = dolfinx.cpp.fem.pack_coefficients(form)
    consts = dolfinx.cpp.fem.pack_constants(form)

    A = dolfinx.cpp.fem.petsc.create_matrix(form)
    A.zeroEntries()
    Ah = A.handle

    set_vals = get_matsetvalues_api()
    mode = PETSc.InsertMode.ADD_VALUES

    for i, id in enumerate(integral_ids):
        kernel = getattr(
            form.ufcx_form.integrals(dolfinx.cpp.fem.IntegralType.cell)[i],
            "tabulate_tensor_runtime_float64",
        )

        coeffs = all_coeffs[(dolfinx.cpp.fem.IntegralType.cell, id)]
        assemble_cells(
            Ah,
            kernel,
            vertices,
            coords,
            dofs,
            num_loc_dofs,
            coeffs,
            consts,
            qr_data[i],
            set_vals,
            mode,
        )

    return A


@numba.njit
def assemble_cells(
    Ah,
    kernel,
    vertices,
    coords,
    dofmap,
    num_loc_dofs,
    coeffs,
    consts,
    qr,
    set_vals,
    mode,
):
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
    A_local = np.zeros((num_loc_dofs, num_loc_dofs), dtype=PETSc.ScalarType)
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array([0], dtype=np.uint8)

    for k, cell in enumerate(cells):
        pos = dofmap[cell, :]
        cell_coords[:, :] = coords[vertices[cell, :]]
        num_quadrature_points = len(qr_w[k])
        A_local.fill(0.0)

        kernel(
            ffi.from_buffer(A_local),
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

        set_vals(
            Ah,
            num_loc_dofs,
            ffi.from_buffer(pos),
            num_loc_dofs,
            ffi.from_buffer(pos),
            ffi.from_buffer(A_local),
            mode,
        )

    sink(A_local, dofmap)
