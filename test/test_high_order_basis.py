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
    # ("polynomial_order", "quadrature_degree"), [(1, 2), (2, 4), (3, 4)]  # , (4, 4)]
    ("polynomial_order", "quadrature_degree"),
    [(2, 4)],  # , (4, 4)]
)
@pytest.mark.parametrize("fcn", [common.fcn1, common.fcn2, common.fcn3, common.fcn4])
def test_high_order_quads(assembler, norm, polynomial_order, quadrature_degree, fcn):
    Nx = 2
    Ny = 3
    mesh = create_high_order_quad_mesh(Nx, Ny, polynomial_order)
    fiat_element = FIAT.reference_element.UFCQuadrilateral()

    b, b_ref = assembler(mesh, fiat_element, polynomial_order, quadrature_degree, fcn)

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
# @pytest.mark.xfail
@pytest.mark.skip
def test_high_order_hexes(assembler, norm, polynomial_order, quadrature_degree, fcn):
    Nx = 2
    Ny = 3
    Nz = 4
    mesh = create_high_order_hex_mesh(Nx, Ny, Nz, polynomial_order)
    fiat_element = FIAT.reference_element.UFCHexahedron()

    b, b_ref = assembler(mesh, fiat_element, polynomial_order, quadrature_degree, fcn)

    assert norm(b - b_ref) / norm(b_ref) < 1e-10


def test_edges():
    # Integrate 2x+y over the edges of a rectangle. Find the edges
    # using the topology of the mesh.

    N = 1
    polynomial_order = 1

    xmin = np.array([0.0, 0.0])
    xmax = np.array([1.0, 1.0])
    cell_type = dolfinx.mesh.CellType.quadrilateral
    mesh1 = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, np.array([xmin, xmax]), np.array([N, N]), cell_type=cell_type
    )

    mesh2 = create_high_order_quad_mesh(N, N, polynomial_order)

    mesh = mesh2

    tdim = mesh.topology.dim
    xmin = np.min(mesh.geometry.x, axis=0)[0:tdim]
    xmax = np.max(mesh.geometry.x, axis=0)[0:tdim]
    xdiff = xmax - xmin

    num_cells = cq.utils.get_num_cells(mesh)
    cells = np.arange(num_cells)

    x = ufl.SpatialCoordinate(mesh)
    fcn = lambda x: 2 * x[0] + x[1]
    dx = ufl.dx(metadata={"quadrature_rule": "runtime"})
    form = dolfinx.fem.form(fcn(x) * dx(domain=mesh))

    # Facet nodes
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    f2n = mesh.topology.connectivity(tdim - 1, 0)

    cell_volume = np.prod(xmax - xmin)

    midpoint = [None] * 4
    facet_area = [None] * 4
    qr_pts = [None] * 4
    qr_w = [None] * 4
    m = [None] * 4
    m_exact = [None] * 4

    for k in range(4):
        n = f2n.links(k)

        midpoint[k] = np.mean(mesh.geometry.x[n], axis=0)[0:tdim]
        qr_pts[k] = np.expand_dims((midpoint[k] - xmin) / (xmax - xmin), axis=0)

        facet_area[k] = np.linalg.norm(np.diff(mesh.geometry.x[n], axis=0))
        qr_w[k] = np.array([[facet_area[k] / cell_volume]])

        qr_data = [(cells, qr_pts[k], qr_w[k])]
        m[k] = cq.assemble_scalar(form, qr_data)
        m_exact[k] = fcn(midpoint[k]) * facet_area[k]

        print(k, n, midpoint[k], qr_pts[k], qr_w[k], m[k], m_exact[k])
        breakpoint()
        assert abs(m[k] - m_exact[k]) / m_exact[k] < 1e-10


def test_edge_integral():
    # Test bdry integral with basis function
    N = 1
    polynomial_order = 1
    quadrature_degree = 2 * polynomial_order

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

    fiat_element = FIAT.reference_element.UFCInterval()
    q1d = FIAT.create_quadrature(fiat_element, quadrature_degree)

    facet_size = np.array([xdiff[0], xdiff[0], xdiff[1], xdiff[1]])
    cell_volume = np.prod(xdiff)
    qr_w = 0.5 * facet_size / cell_volume

    bottom_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, bottom_facets)
    top_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, top_facets)
    left_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, left_facets)
    right_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, right_facets)

    tags = [bottom_tag, top_tag, left_tag, right_tag]

    # qr_pts are x0, y0, x1, y1, ...
    qr_pts_local = np.zeros(tdim * q1d.get_points().size)
    qr_pts_local[0::2] = q1d.get_points().flatten()
    qr_pts_local = np.expand_dims(qr_pts_local, axis=0)
    qr_w_local = np.expand_dims(q1d.get_weights().flatten(), axis=0)

    for k in range(4):
        b = cq.assemble_vector(
            dolfinx.fem.form(integrand * dx),
            [(cells, qr_pts_local, qr_w_local)],
        )

        ds_local = ds(tags[k])
        b_exact = dolfinx.fem.petsc.assemble_vector(
            dolfinx.fem.form(integrand * ds_local)
        )

        assert np.linalg.norm(b.array - b_exact.array) < 1e-10
