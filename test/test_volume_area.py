import pytest
import dolfinx
import ufl
from mpi4py import MPI
import numpy as np
import customquad as cq
import common


def test_volume():
    # Compute the full volume of the domain given by the mesh

    (
        mesh,
        h,
        celltags,
        cut_cell_tag,
        uncut_cell_tag,
        outside_cell_tag,
    ) = common.get_mesh()

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

    # The measure for the integration with quadrature rules at runtime
    dx = ufl.Measure("dx", metadata={"quadrature_rule": "runtime"})

    # Compute the volume over the cut cells by integration using
    # runtime assembly. First create a simple quadrature (the qr_pts
    # doesn't matter here). Fenics will internally do the mapping to
    # the physical cell.
    qr_pts = np.tile([0.5] * dim, [len(cut_cells), 1])
    qr_w = np.tile(1.0, [len(cut_cells), 1])
    qr_cut = [(cut_cells, qr_pts, qr_w)]

    # We don't need subdomain data here: the integration domain is
    # given by the cells in qr_cut. We need the domain=mesh since
    # there's no function space in the integrand.
    form = dolfinx.fem.form(1.0 * dx(domain=mesh))
    vol_cut = cq.assemble_scalar(form, qr_cut)
    assert abs(vol_cut - len(cut_cells) * cell_vol) / abs(vol_cut) < 1e-10

    # To check everything, compute the volume of the cells outside as
    # well
    qr_pts = np.tile([0.5] * dim, [len(outside_cells), 1])
    qr_w = np.tile(1.0, [len(outside_cells), 1])
    qr_outside = [(outside_cells, qr_pts, qr_w)]
    form = dolfinx.fem.form(1.0 * dx(domain=mesh))
    vol_outside = cq.assemble_scalar(form, qr_outside)

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


def test_full_area():
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
    ) = common.get_mesh()

    dim = mesh.topology.dim

    # Find exterior facets
    mesh.topology.create_connectivity(dim - 1, dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    num_ext_facets = len(exterior_facets)

    # Find the cells of the exterior facets
    facet_to_cells = mesh.topology.connectivity(dim - 1, dim)
    exterior_cells = [facet_to_cells.links(f)[0] for f in exterior_facets]
    assert num_ext_facets == len(exterior_cells)

    # The points does not matter since we integrate over a constant
    qr_pts = np.tile([0.0] * dim, [num_ext_facets, 1])

    # The facet integrals are computed as cell integrals with facet
    # quadrature. Since fenics internally multiplies with cell volume,
    # we must divide the qr weight by the volume and use the right
    # facet weight (which is trivially 1 here). We must also remember
    # that the quadrature should be defined cell-wise. Some external
    # cells have two facets (in the corners cells of a square). At
    # such cells, we could have a quadrature rule with two weights
    # (one for each facet), but here we simply have two separate
    # quadrature rules.
    cell_volume = tensor_product_volumes(mesh, exterior_cells, dim)
    facet_area = tensor_product_volumes(mesh, exterior_facets, dim - 1)
    qr_w = np.empty([num_ext_facets, 1])
    for k in range(num_ext_facets):
        qr_w[k] = facet_area[k] / cell_volume[k]

    qr_data = [(exterior_cells, qr_pts, qr_w)]

    # Integral
    dx = ufl.Measure("dx", metadata={"quadrature_rule": "runtime"})
    form = dolfinx.fem.form(1.0 * dx(domain=mesh))
    area = cq.assemble_scalar(form, qr_data)

    # Verify
    form = dolfinx.fem.form(1.0 * ufl.ds(domain=mesh))
    exact_area = dolfinx.fem.assemble_scalar(form)
    assert abs(area - exact_area) / exact_area < 1e-10


def test_numbering():
    N = 1
    cell_type = dolfinx.mesh.CellType.quadrilateral
    xmin = np.array([0.0, 0.0])
    xmax = np.array([1.0, 1.0])
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, np.array([xmin, xmax]), np.array([N, N]), cell_type=cell_type
    )
    tdim = mesh.topology.dim

    def bottom(x):
        return np.isclose(x[1], xmin[1])

    def top(x):
        return np.isclose(x[1], xmax[1])

    def left(x):
        return np.isclose(x[0], xmin[0])

    def right(x):
        return np.isclose(x[0], xmax[0])

    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, bottom)
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, top)
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, left)
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, right)

    mesh.topology.create_connectivity(tdim - 1, 0)
    f2n = mesh.topology.connectivity(tdim - 1, 0)

    bottom_nodes = f2n.links(bottom_facets)
    top_nodes = f2n.links(top_facets)
    left_nodes = f2n.links(left_facets)
    right_nodes = f2n.links(right_facets)

    bottom_midpoint = np.array([0.5, 0.0])
    top_midpoint = np.array([0.5, 1.0])
    left_midpoint = np.array([0.0, 0.5])
    right_midpoint = np.array([1.0, 0.5])

    assert np.all(
        np.mean(mesh.geometry.x[bottom_nodes, 0:tdim], axis=0) == bottom_midpoint
    )
    assert np.all(np.mean(mesh.geometry.x[top_nodes, 0:tdim], axis=0) == top_midpoint)
    assert np.all(np.mean(mesh.geometry.x[left_nodes, 0:tdim], axis=0) == left_midpoint)
    assert np.all(
        np.mean(mesh.geometry.x[right_nodes, 0:tdim], axis=0) == right_midpoint
    )
