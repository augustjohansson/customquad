import pytest
import dolfinx
import ufl
from mpi4py import MPI
import numpy as np
import customquad as cq


def get_mesh():
    # Mesh
    N = 10
    cell_type = dolfinx.mesh.CellType.quadrilateral
    # mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type)
    xmin = np.array([-1.23, -11.11])
    xmax = np.array([3.33, 99.99])
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, np.array([xmin, xmax]), np.array([N, N]), cell_type=cell_type
    )
    h = (xmax - xmin) / N

    # Classify cells
    dim = mesh.topology.dim
    num_cells = mesh.topology.index_map(dim).size_local
    all_cells = np.arange(num_cells, dtype=np.int32)
    ge = dolfinx.cpp.mesh.entities_to_geometry(mesh, dim, all_cells, False)
    centroids = np.mean(mesh.geometry.x[ge], axis=1)
    xc = centroids[:, 0]
    yc = centroids[:, 1]

    left = xc < xmin[0] + h[0]
    right = xc > xmax[0] - h[0]
    bottom = yc < xmin[1] + h[1]
    top = yc > xmax[1] - h[1]
    outside_cells = np.where(np.logical_or.reduce((left, right, bottom, top)))[0]
    assert len(outside_cells) == 4 * (N - 1)

    left = xc < xmin[0] + 2 * h[0]
    right = xc > xmax[0] - 2 * h[0]
    bottom = yc < xmin[1] + 2 * h[1]
    top = yc > xmax[1] - 2 * h[1]
    cut_cells = np.where(np.logical_or.reduce((left, right, bottom, top)))[0]
    cut_cells = np.setdiff1d(cut_cells, outside_cells)
    assert len(cut_cells) == 4 * (N - 3)

    uncut_cells = np.setdiff1d(all_cells, outside_cells)
    uncut_cells = np.setdiff1d(uncut_cells, cut_cells)
    assert len(uncut_cells) == (N - 4) ** 2

    # Setup mesh tags
    cut_cell_tag = 1
    uncut_cell_tag = 2
    outside_cell_tag = 3

    celltags = cq.utils.get_celltags(
        mesh,
        cut_cells,
        uncut_cells,
        outside_cells,
        cut_cell_tag=cut_cell_tag,
        uncut_cell_tag=uncut_cell_tag,
        outside_cell_tag=outside_cell_tag,
    )

    return (
        mesh,
        h,
        celltags,
        cut_cell_tag,
        uncut_cell_tag,
        outside_cell_tag,
    )


def test_volume():
    # Compute the full volume of the domain given by the mesh

    (
        mesh,
        h,
        celltags,
        cut_cell_tag,
        uncut_cell_tag,
        outside_cell_tag,
    ) = get_mesh()

    dim = mesh.topology.dim
    cell_vol = h[0] * h[1]
    num_cells = cq.utils.get_num_cells(mesh)

    # Extract cell types from celltags
    cut_cells = np.where(celltags.values == cut_cell_tag)[0]
    uncut_cells = np.where(celltags.values == uncut_cell_tag)[0]
    outside_cells = np.where(celltags.values == outside_cell_tag)[0]

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
    assert abs(vol_bulk - len(uncut_cells) * cell_vol) < 1e-10

    # Compute the volume over the cut cells by integration using
    # runtime assembly. First create a simple quadrature (the qr_pts
    # doesn't matter here). Fenics will internally do the mapping to
    # the physical cell.
    qr_pts = np.tile([0.5] * dim, [len(cut_cells), 1])
    qr_w = np.tile(1.0, [len(cut_cells), 1])
    qr_cut = [(cut_cells, qr_pts, qr_w)]

    # We don't need subdomain data here: the integration domain is
    # given by the cells in qr_cut.
    dx_cut = ufl.Measure("dx", metadata={"quadrature_rule": "runtime"})
    vol_cut = cq.assemble_scalar(dolfinx.fem.form(1.0 * dx_cut(domain=mesh)), qr_cut)
    assert abs(vol_cut - len(cut_cells) * cell_vol) / abs(vol_cut) < 1e-10

    # To check everything, compute the volume of the cells outside as
    # well
    qr_pts = np.tile([0.5] * dim, [len(outside_cells), 1])
    qr_w = np.tile(1.0, [len(outside_cells), 1])
    qr_outside = [(outside_cells, qr_pts, qr_w)]
    dx_outside = ufl.Measure("dx", metadata={"quadrature_rule": "runtime"})
    vol_outside = cq.assemble_scalar(
        dolfinx.fem.form(1.0 * dx_outside(domain=mesh)), qr_outside
    )

    # Verify
    total_vol = num_cells * cell_vol
    exact_vol = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(domain=mesh)))
    assert abs(total_vol - exact_vol) / exact_vol < 1e-10
    assert abs(vol_bulk + vol_cut + vol_outside - exact_vol) / exact_vol < 1e-10


def tensor_product_volumes(mesh, entities, dim):
    ge = dolfinx.cpp.mesh.entities_to_geometry(mesh, dim, entities, False)
    x = mesh.geometry.x[ge]
    xmin = np.min(x, axis=1)
    xmax = np.max(x, axis=1)
    dx = xmax - xmin

    if dim == 1:
        volumes = np.max(dx, axis=1)
    else:
        volumes = np.prod(dx[:, :dim], axis=1)

    return volumes


def test_area():
    # Compute the exterior area of the domain given by the mesh. This
    # illustrates the need to scale the quadrature weight by the
    # volume of the cell and the area of the facet.

    (
        mesh,
        h,
        celltags,
        cut_cell_tag,
        uncut_cell_tag,
        outside_cell_tag,
    ) = get_mesh()

    dim = mesh.topology.dim

    # Find exterior facets
    mesh.topology.create_connectivity(dim - 1, dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    num_ext_facets = len(exterior_facets)

    # Find the cells of the exterior facets
    facet_to_cells = mesh.topology.connectivity(dim - 1, dim)
    exterior_cells = [facet_to_cells.links(f)[0] for f in exterior_facets]
    assert num_ext_facets == len(exterior_cells)

    # The points does not matter
    qr_pts = np.tile([0.5] * dim, [num_ext_facets, 1])

    # The facet integrals are computed as cell integrals with facet
    # quadrature. Since fenics internally multiplies with cell volume,
    # divide the qr weight by the volume and use the right facet
    # weight (which is trivially 1 here). We must also remember that
    # the quadrature should be defined cell-wise. Some external cells
    # have two facets (in the corners cells of a square). At such
    # cells, we could have a quadrature rule with two weights (one for
    # each facet), but here we simply have two separate quadrature
    # rules.
    cell_volume = tensor_product_volumes(mesh, exterior_cells, dim)
    facet_area = tensor_product_volumes(mesh, exterior_facets, dim - 1)
    qr_w = np.empty([num_ext_facets, 1])
    for k in range(num_ext_facets):
        qr_w[k] = 1.0 / cell_volume[k] * facet_area[k]

    qr_data = [(exterior_cells, qr_pts, qr_w)]
    dx = ufl.Measure("dx", metadata={"quadrature_rule": "runtime"})
    area = cq.assemble_scalar(dolfinx.fem.form(1.0 * dx(domain=mesh)), qr_data)

    # Verify
    exact_area = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(1.0 * ufl.ds(domain=mesh))
    )
    assert abs(area - exact_area) / exact_area < 1e-10


def test_normals():
    pass


def test_many_integral_ids():
    pass
