import pytest
import basix
import dolfinx
import ufl
from mpi4py import MPI
import numpy as np
import FIAT
import common
import customquad as cq
from create_high_order_mesh import (
    create_high_order_quad_mesh,
    create_high_order_hex_mesh,
)

flatten = lambda l: [item for sublist in l for item in sublist]


@pytest.mark.parametrize(
    ("assembler, norm"),
    [
        (common.assemble_scalar_test, common.scalar_norm),
        (common.assemble_vector_test, common.vector_norm),
        (common.assemble_matrix_test, common.matrix_norm),
    ],
)
@pytest.mark.parametrize(
    ("polynomial_order", "quadrature_degree"),
    [(1, 2), (2, 4), (3, 6)]  # , (4, 8)]
    # ("polynomial_order", "quadrature_degree"),
    # [(3, 6)],  # , (4, 8)]
)
@pytest.mark.parametrize("fcn", [common.fcn1, common.fcn2, common.fcn3, common.fcn4])
# @pytest.mark.xfail
def test_high_order_quads(assembler, norm, polynomial_order, quadrature_degree, fcn):
    Nx = 1
    Ny = 1
    mesh = create_high_order_quad_mesh(Nx, Ny, polynomial_order)
    fiat_element = FIAT.reference_element.UFCQuadrilateral()

    perm = np.arange((polynomial_order + 1) ** 2).tolist()

    b, b_ref = assembler(
        mesh, fiat_element, polynomial_order, quadrature_degree, fcn, perm
    )

    assert norm(b - b_ref) / norm(b_ref) < 1e-10


@pytest.mark.parametrize(
    ("assembler, norm"),
    [
        (common.assemble_scalar_test, common.scalar_norm),
        (common.assemble_vector_test, common.vector_norm),
        (common.assemble_matrix_test, common.matrix_norm),
    ],
)
@pytest.mark.parametrize(
    ("polynomial_order", "quadrature_degree"), [(1, 2), (2, 4)]  # , (3, 2), (4, 3)]
)
@pytest.mark.parametrize("fcn", [common.fcn1, common.fcn2, common.fcn3, common.fcn4])
@pytest.mark.xfail
def test_high_order_hexes(assembler, norm, polynomial_order, quadrature_degree, fcn):
    Nx = 2
    Ny = 3
    Nz = 4
    mesh = create_high_order_hex_mesh(Nx, Ny, Nz, polynomial_order)
    fiat_element = FIAT.reference_element.UFCHexahedron()

    b, b_ref = assembler(mesh, fiat_element, polynomial_order, quadrature_degree, fcn)

    assert norm(b - b_ref) / norm(b_ref) < 1e-10


# @pytest.mark.xfail
def test_edge_integral():
    # Test bdry integral with basis function
    N = 1
    polynomial_order = 2
    mesh = create_high_order_quad_mesh(N, N, polynomial_order)
    tdim = mesh.topology.dim
    xmin = np.min(mesh.geometry.x, axis=0)[0:tdim]
    xmax = np.max(mesh.geometry.x, axis=0)[0:tdim]
    xdiff = xmax - xmin

    num_cells = cq.utils.get_num_cells(mesh)
    cells = np.arange(num_cells)

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
    v = ufl.TestFunction(V)
    integrand = 1 * v

    # From test_assembly_ds_domains in test_assemble_domains
    def bottom(x):
        return np.isclose(x[1], xmin[1])

    def top(x):
        return np.isclose(x[1], xmax[1])

    def left(x):
        return np.isclose(x[0], xmin[0])

    def right(x):
        return np.isclose(x[0], xmax[0])

    bottom_tag = 10
    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, bottom)
    bottom_vals = np.full(bottom_facets.shape, bottom_tag, np.intc)

    top_tag = 11
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, top)
    top_vals = np.full(top_facets.shape, top_tag, np.intc)

    left_tag = 12
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, left)
    left_vals = np.full(left_facets.shape, left_tag, np.intc)

    right_tag = 13
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, right)
    right_vals = np.full(right_facets.shape, right_tag, np.intc)

    indices = np.hstack((bottom_facets, top_facets, left_facets, right_facets))
    values = np.hstack((bottom_vals, top_vals, left_vals, right_vals))

    indices, pos = np.unique(indices, return_index=True)
    marker = dolfinx.mesh.meshtags(mesh, tdim - 1, indices, values[pos])
    ds = ufl.Measure("ds", subdomain_data=marker)
    dx = ufl.dx(metadata={"quadrature_rule": "runtime"})

    mesh.topology.create_connectivity(tdim - 1, 0)
    f2n = mesh.topology.connectivity(tdim - 1, 0)

    bottom_nodes = f2n.links(bottom_facets)
    top_nodes = f2n.links(top_facets)
    left_nodes = f2n.links(left_facets)
    right_nodes = f2n.links(right_facets)

    w0 = (1 + 1 / np.sqrt(3)) / 2
    w1 = (1 - 1 / np.sqrt(3)) / 2

    def gauss(x):
        return np.array([w0 * x[0] + w1 * x[1], w1 * x[0] + w0 * x[1]]).flatten()

    x = mesh.geometry.x[:, 0:tdim]
    qr_pts = np.empty((4, 4))
    qr_pts[0] = gauss(x[bottom_nodes])
    qr_pts[1] = gauss(x[top_nodes])
    qr_pts[2] = gauss(x[left_nodes])
    qr_pts[3] = gauss(x[right_nodes])

    facet_size = np.array([xdiff[0], xdiff[0], xdiff[1], xdiff[1]])
    cell_volume = np.prod(xdiff)
    qr_w = 0.5 * facet_size / cell_volume

    bottom_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, bottom_facets)
    top_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, top_facets)
    left_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, left_facets)
    right_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, right_facets)

    tags = [bottom_tag, top_tag, left_tag, right_tag]

    fiat_element = FIAT.reference_element.UFCInterval()
    quadrature_degree = 2
    q = FIAT.create_quadrature(fiat_element, quadrature_degree)
    qr_pts_local = np.array([[0.0, 0, 0, 0]])
    qr_pts_local[0][0] = q.get_points()[0][0]
    qr_pts_local[0][2] = q.get_points()[1][0]
    qr_w_local = np.tile(q.get_weights().flatten(), [num_cells, 1])
    perm = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    for k in range(1):
        # qr_pts_local = np.expand_dims(qr_pts[k], axis=0)
        # qr_w_local = np.expand_dims(np.repeat(qr_w[k], 2), axis=0)

        b = cq.assemble_vector(
            dolfinx.fem.form(integrand * dx), [(cells, qr_pts_local, qr_w_local)], perm
        )

        ds_local = ds(tags[k])
        b_exact = dolfinx.fem.petsc.assemble_vector(
            dolfinx.fem.form(integrand * ds_local)
        )

        # breakpoint()

        assert np.linalg.norm(b.array - b_exact.array) < 1e-10
