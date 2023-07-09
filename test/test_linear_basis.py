import pytest
import dolfinx
from mpi4py import MPI
import numpy as np
import customquad as cq
import ufl
import FIAT
import common
from petsc4py import PETSc
import itertools


@pytest.mark.parametrize(
    ("assembler, norm"),
    [
        (common.assemble_scalar_test, common.scalar_norm),
        (common.assemble_vector_test, common.vector_norm),
        (common.assemble_matrix_test, common.matrix_norm),
    ],
)
@pytest.mark.parametrize("N", [[1, 1], [3, 1], [1, 3], [3, 4]])
@pytest.mark.parametrize("xmin", [[0, 0], [-0.25, -10.25]])
@pytest.mark.parametrize("xmax", [[1, 1], [1.25, 17.5]])
@pytest.mark.parametrize("fcn", [common.fcn1, common.fcn2, common.fcn3, common.fcn4])
def test_quads_assembly(assembler, norm, N, xmin, xmax, fcn):
    polynomial_order = 1
    quadrature_degree = 2
    fiat_element = FIAT.reference_element.UFCQuadrilateral()
    cell_type = dolfinx.mesh.CellType.quadrilateral
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([xmin, xmax]),
        np.array(N),
        cell_type,
    )

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
@pytest.mark.parametrize("N", [[1, 1, 1], [3, 1, 1], [1, 3, 1], [1, 1, 3], [3, 4, 5]])
@pytest.mark.parametrize("xmin", [[0, 0, 0], [-0.25, -10.25, 0.25]])
@pytest.mark.parametrize("xmax", [[1, 1, 1], [1.25, 17.5, 4.4]])
@pytest.mark.parametrize("fcn", [common.fcn1, common.fcn2, common.fcn3, common.fcn4])
def test_hexes_assembly(assembler, norm, N, xmin, xmax, fcn):
    polynomial_order = 1
    quadrature_degree = 2
    fiat_element = FIAT.reference_element.UFCHexahedron()
    cell_type = dolfinx.mesh.CellType.hexahedron
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        np.array([xmin, xmax]),
        np.array(N),
        cell_type,
    )

    b, b_ref = assembler(mesh, fiat_element, polynomial_order, quadrature_degree, fcn)
    assert norm(b - b_ref) / norm(b_ref) < 1e-10


def test_edges():
    # Integrate 2x+y over the edges of a rectangle. Find the edges
    # using the topology of the mesh.

    N = 1
    cell_type = dolfinx.mesh.CellType.quadrilateral
    xmin = np.array([-1.0, -1.0])
    xmax = np.array([4.0, 1.0])
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, np.array([xmin, xmax]), np.array([N, N]), cell_type=cell_type
    )
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

        assert abs(m[k] - m_exact[k]) / m_exact[k] < 1e-10


def test_edge_integral():
    # Test bdry integral with basis function
    N = 1
    cell_type = dolfinx.mesh.CellType.quadrilateral
    xmin = np.array([-0.25, -10.25])
    xmax = np.array([1.25, 17.5])
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, np.array([xmin, xmax]), np.array([N, N]), cell_type=cell_type
    )
    tdim = mesh.topology.dim
    num_cells = cq.utils.get_num_cells(mesh)
    cells = np.arange(num_cells)

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
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

    # Complicated way of calculating qr_pts, but this illustrates the numbering
    qr_pts_ref = np.array([[0.5, 0.0], [0.5, 1.0], [0, 0.5], [1.0, 0.5]])
    xdiff = xmax - xmin
    x = mesh.geometry.x[:, 0:tdim]
    qr_pts = np.empty((4, 2))
    qr_pts[0] = (np.mean(x[bottom_nodes], axis=0) - xmin) / xdiff
    qr_pts[1] = (np.mean(x[top_nodes], axis=0) - xmin) / xdiff
    qr_pts[2] = (np.mean(x[left_nodes], axis=0) - xmin) / xdiff
    qr_pts[3] = (np.mean(x[right_nodes], axis=0) - xmin) / xdiff
    for k in range(4):
        assert np.all(abs(qr_pts[k] - qr_pts_ref[k]) < 1e-10)

    facet_size = np.array([xdiff[0], xdiff[0], xdiff[1], xdiff[1]])
    cell_volume = np.prod(xdiff)
    qr_w = facet_size / cell_volume

    tags = [bottom_tag, top_tag, left_tag, right_tag]
    for k in range(4):
        qr_pts_local = np.expand_dims(qr_pts[k], axis=0)
        qr_w_local = np.expand_dims(qr_w[k], axis=(0, 1))

        b = cq.assemble_vector(
            dolfinx.fem.form(integrand * dx),
            [(cells, qr_pts_local, qr_w_local)],
        )

        ds_local = ds(tags[k])
        b_exact = dolfinx.fem.petsc.assemble_vector(
            dolfinx.fem.form(integrand * ds_local)
        )

        for m in range(len(b.array)):
            assert abs(b.array[m] - b_exact.array[m]) < 1e-10


# def test_edges3():
#     # One edge at a time
#     N = 1
#     cell_type = dolfinx.mesh.CellType.quadrilateral
#     # xmin = np.array([-1.0, -1.0])
#     # xmax = np.array([4.0, 1.0])
#     xmin = np.array([0.0, 0.0])
#     xmax = np.array([4.0, 1.0])
#     mesh = dolfinx.mesh.create_rectangle(
#         MPI.COMM_WORLD, np.array([xmin, xmax]), np.array([N, N]), cell_type=cell_type
#     )
#     num_cells = cq.utils.get_num_cells(mesh)
#     cells = np.arange(num_cells)
#     tdim = mesh.topology.dim

#     V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
#     v = ufl.TestFunction(V)
#     integrand = 1 * v

#     # From test_assembly_ds_domains in test_assemble_domains
#     def bottom(x):
#         return np.isclose(x[1], xmin[1])

#     bottom_tag = 1
#     bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, bottom)
#     bottom_vals = np.full(bottom_facets.shape, bottom_tag, np.intc)

#     indices = np.hstack((bottom_facets))
#     values = np.hstack((bottom_vals))

#     indices, pos = np.unique(indices, return_index=True)
#     marker = dolfinx.mesh.meshtags(mesh, tdim - 1, indices, values[pos])
#     ds = ufl.Measure("ds", subdomain_data=marker, domain=mesh)

#     # Bottom facet
#     qr_pts = np.array([[0.5, 0.0]])
#     dx = xmax - xmin
#     facet_area = np.max(dx)
#     cell_volume = np.prod(dx)
#     qr_w = np.array([[facet_area / cell_volume]])

#     b1 = cq.assemble_vector(
#         dolfinx.fem.form(
#             integrand * ufl.dx(metadata={"quadrature_rule": "runtime"}, domain=mesh)
#         ),
#         [(cells, qr_pts, qr_w)],
#     )

#     b1_exact = dolfinx.fem.petsc.assemble_vector(
#         dolfinx.fem.form(integrand * ds(bottom_tag))
#     )

#     breakpoint()


# def test_corners():
#     N = 1
#     cell_type = dolfinx.mesh.CellType.quadrilateral
#     xmin = np.array([-1.0, -1.0])
#     xmax = np.array([4.0, 1.0])
#     mesh = dolfinx.mesh.create_rectangle(
#         MPI.COMM_WORLD, np.array([xmin, xmax]), np.array([N, N]), cell_type=cell_type
#     )
#     num_cells = cq.utils.get_num_cells(mesh)
#     cells = np.arange(num_cells)

#     tdim = mesh.topology.dim
#     mesh.topology.create_connectivity(tdim, 0)
#     c2n = mesh.topology.connectivity(tdim, 0)
#     # c2n = mesh.geometry.dofmap.array
#     breakpoint()

#     y = ufl.SpatialCoordinate(mesh)
#     # x = mesh.geometry.x
#     fcn = lambda x: 2 * x[0] + x[1]
#     dx = ufl.dx(metadata={"quadrature_rule": "runtime"})
#     form = dolfinx.fem.form(fcn(y) * dx(domain=mesh))

#     cell_volume = np.prod(xmax - xmin)
#     qr_w = np.array([[1.0]])

#     qr_pts = [None] * 4
#     m = [None] * 4
#     m_exact = [None] * 4
#     for k in range(4):
#         n = c2n.links(0)[k]
#         # n = c2n[k]
#         x = mesh.geometry.x[n, : mesh.geometry.dim]
#         qr_pts[k] = np.expand_dims((x - xmin) / (xmax - xmin), axis=0)
#         qr_data = [(cells, qr_pts[k], qr_w)]
#         m[k] = cq.assemble_scalar(form, qr_data)
#         m_exact[k] = fcn(x) * cell_volume

#     breakpoint()
#     # N = 1
#     # cell_type = dolfinx.mesh.CellType.quadrilateral
#     # mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type)
#     # num_cells = cq.utils.get_num_cells(mesh)
#     # cells = np.arange(num_cells)

#     # # x = ufl.SpatialCoordinate(mesh)
#     # # integrand = 2 * x[0] + x[1]
#     # # form = dolfinx.fem.form(
#     # #     integrand * ufl.dx(metadata={"quadrature_rule": "runtime"}, domain=mesh)
#     # # )

#     # V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
#     # f = dolfinx.fem.Function(V)
#     # fcn = lambda x: np.sqrt(2 * x[0] + x[1])
#     # f.interpolate(fcn)
#     # integrand = ufl.inner(f, f)

#     # form = dolfinx.fem.form(integrand * ufl.dx(metadata={"quadrature_rule": "runtime"}))

#     # qr_w = np.array([[1.0]])
#     # qr_data = [(cells, qr_pts, qr_w)]
#     # m = cq.assemble_scalar(form, qr_data)

#     # assert abs(m - m_exact) < 1e-10
