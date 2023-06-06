import pytest
import dolfinx
import ufl
from mpi4py import MPI
import numpy as np
import customquad as cq


def test_volume():
    # Mesh
    N = 10
    h = 1.0 / N
    cell_type = dolfinx.mesh.CellType.quadrilateral
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type)

    # Classify cells
    dim = mesh.topology.dim
    num_cells = mesh.topology.index_map(dim).size_local
    all_cells = np.arange(num_cells, dtype=np.int32)
    ge = dolfinx.cpp.mesh.entities_to_geometry(mesh, dim, all_cells, False)
    centroids = np.mean(mesh.geometry.x[ge], axis=1)
    xc = centroids[:, 0]
    yc = centroids[:, 1]

    left = xc < h
    right = xc > 1.0 - h
    bottom = yc < h
    top = yc > 1.0 - h
    outside_cells = np.where(np.logical_or.reduce((left, right, bottom, top)))[0]
    assert len(outside_cells) == 4 * (N - 1)

    left = xc < 2 * h
    right = xc > 1.0 - 2 * h
    bottom = yc < 2 * h
    top = yc > 1.0 - 2 * h
    cut_cells = np.where(np.logical_or.reduce((left, right, bottom, top)))[0]
    cut_cells = np.setdiff1d(cut_cells, outside_cells)
    assert len(cut_cells) == 4 * (N - 3)

    uncut_cells = np.setdiff1d(all_cells, outside_cells)
    uncut_cells = np.setdiff1d(uncut_cells, cut_cells)
    assert len(uncut_cells) == (N - 4) ** 2

    # Setup mesh tags
    uncut_cell_tag = 1
    cut_cell_tag = 2
    outside_cell_tag = 3
    ghost_penalty_tag = 4

    celltags = cq.utils.get_celltags(
        mesh,
        cut_cells,
        uncut_cells,
        outside_cells,
        uncut_cell_tag=uncut_cell_tag,
        cut_cell_tag=cut_cell_tag,
        outside_cell_tag=outside_cell_tag,
    )

    # Compute volume in two parts: first integrate over the uncut
    # cells in the interior using standard dolfinx functions. We need
    # to pass the correct cell tag, the subdomain data specifying the
    # cell tags, as well as the domain (since there are no test or
    # trial functions which can provide the mesh via the function
    # space).
    vol_bulk = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(
            1.0 * ufl.dx(uncut_cell_tag, subdomain_data=celltags, domain=mesh)
        )
    )
    assert abs(vol_bulk - len(uncut_cells) * h**2) < 1e-10

    # Compute the area over the cut cells by integration using runtime
    # assembly. First create the simplest quadrature: midpoint
    # quadrature in all cut cells.
    qr_pts = np.tile([0.5] * dim, [len(cut_cells), 1])
    qr_w = np.tile(1.0, [len(cut_cells), 1])
    qr_n = qr_pts  # dummy
    qr_cut = [(cut_cells, qr_pts, qr_w, qr_n)]

    # We don't need subdomain data here: the integration domain is
    # given by the cells in qr_cut.
    dx_cut = ufl.Measure("dx", metadata={"quadrature_rule": "runtime"})
    vol_cut = cq.assemble_scalar(dolfinx.fem.form(1.0 * dx_cut(domain=mesh)), qr_cut)
    assert abs(vol_cut - len(cut_cells) * h**2) < 1e-10

    # To check everything, compute the volume of the cells outside as
    # well
    qr_pts = np.tile([0.5] * dim, [len(outside_cells), 1])
    qr_w = np.tile(1.0, [len(outside_cells), 1])
    qr_n = qr_pts  # dummy
    qr_outside = [(outside_cells, qr_pts, qr_w, qr_n)]
    dx_outside = ufl.Measure("dx", metadata={"quadrature_rule": "runtime"})
    vol_outside = cq.assemble_scalar(
        dolfinx.fem.form(1.0 * dx_outside(domain=mesh)), qr_outside
    )
    assert abs(vol_bulk + vol_cut + vol_outside - 1.0) < 1e-10
