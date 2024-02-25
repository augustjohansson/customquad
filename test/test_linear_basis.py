import pytest
import dolfinx
from mpi4py import MPI
import numpy as np
import customquad as cq
import ufl
import FIAT
import common


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
@pytest.mark.parametrize("use_dolfinx_mesh", [False, True])
def test_quads_assembly(assembler, norm, N, xmin, xmax, fcn, use_dolfinx_mesh):

    polynomial_order = 1
    quadrature_degree = 2
    fiat_element = FIAT.reference_element.UFCQuadrilateral()

    if use_dolfinx_mesh:
        cell_type = dolfinx.mesh.CellType.quadrilateral
        mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([xmin, xmax]),
            np.array(N),
            cell_type,
        )
    else:
        mesh = cq.create_mesh(np.array([xmin, xmax]), np.array(N), polynomial_order)

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
@pytest.mark.parametrize("use_dolfinx_mesh", [False, True])
def test_hexes_assembly(assembler, norm, N, xmin, xmax, fcn, use_dolfinx_mesh):

    polynomial_order = 1
    quadrature_degree = 2
    fiat_element = FIAT.reference_element.UFCHexahedron()

    if use_dolfinx_mesh:
        cell_type = dolfinx.mesh.CellType.hexahedron
        mesh = dolfinx.mesh.create_box(
            MPI.COMM_WORLD,
            np.array([xmin, xmax]),
            np.array(N),
            cell_type,
        )
    else:
        mesh = cq.create_mesh(np.array([xmin, xmax]), np.array(N), polynomial_order)

    b, b_ref = assembler(mesh, fiat_element, polynomial_order, quadrature_degree, fcn)
    assert norm(b - b_ref) / norm(b_ref) < 1e-10


@pytest.mark.parametrize("use_dolfinx_mesh", [False, True])
def test_edges(use_dolfinx_mesh):
    # Integrate 2x+y over the edges of a rectangle. Find the edges
    # using the topology of the mesh.

    N = 1
    xmin = np.array([-1.0, -1.0])
    xmax = np.array([4.0, 1.0])

    if use_dolfinx_mesh:
        cell_type = dolfinx.mesh.CellType.quadrilateral
        mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([xmin, xmax]),
            np.array([N, N]),
            cell_type=cell_type,
        )
    else:
        polynomial_order = 1
        mesh = cq.create_mesh(
            np.array([xmin, xmax]), np.array([N, N]), polynomial_order
        )

    num_cells = cq.utils.get_num_cells(mesh)
    cells = np.arange(num_cells)

    x = ufl.SpatialCoordinate(mesh)

    def fcn(x):
        return 2 * x[0] + x[1]

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


@pytest.mark.parametrize("use_dolfinx_mesh", [False, True])
def test_edge_integral(use_dolfinx_mesh):

    # Test bdry integral with basis function
    N = 1
    xmin = np.array([-0.25, -10.25])
    xmax = np.array([1.25, 17.5])

    if use_dolfinx_mesh:
        cell_type = dolfinx.mesh.CellType.quadrilateral
        mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([xmin, xmax]),
            np.array([N, N]),
            cell_type=cell_type,
        )
    else:
        polynomial_order = 1
        mesh = cq.create_mesh(
            np.array([xmin, xmax]), np.array([N, N]), polynomial_order
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

        for m, y in enumerate(b.array):
            assert abs(y - b_exact.array[m]) < 1e-10


@pytest.mark.parametrize("use_dolfinx_mesh", [True, False])
def test_face_integral(use_dolfinx_mesh):

    # Test bdry integral with basis function
    N = 1
    bbmin = np.array([-0.25, -10.25, -4.4])
    bbmax = np.array([1.25, 17.5, 0.1])

    if use_dolfinx_mesh:
        cell_type = dolfinx.mesh.CellType.hexahedron
        mesh = dolfinx.mesh.create_box(
            MPI.COMM_WORLD,
            np.array([bbmin, bbmax]),
            np.array([N, N, N]),
            cell_type,
        )
    else:
        polynomial_order = 1
        mesh = cq.create_mesh(
            np.array([bbmin, bbmax]), np.array([N, N, N]), polynomial_order
        )

    tdim = mesh.topology.dim
    num_cells = cq.utils.get_num_cells(mesh)
    cells = np.arange(num_cells)

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    integrand = 1 * v

    # fmt: off
    def xmin(x): return np.isclose(x[0], bbmin[0])
    def xmax(x): return np.isclose(x[0], bbmax[0])
    def ymin(x): return np.isclose(x[1], bbmin[1])
    def ymax(x): return np.isclose(x[1], bbmax[1])
    def zmin(x): return np.isclose(x[2], bbmin[2])
    def zmax(x): return np.isclose(x[2], bbmax[2])
    # fmt: on

    xmin_tag = 10
    xmin_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, xmin)
    xmin_vals = np.full(xmin_facets.shape, xmin_tag, np.intc)

    xmax_tag = 11
    xmax_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, xmax)
    xmax_vals = np.full(xmax_facets.shape, xmax_tag, np.intc)

    ymin_tag = 12
    ymin_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, ymin)
    ymin_vals = np.full(ymin_facets.shape, ymin_tag, np.intc)

    ymax_tag = 13
    ymax_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, ymax)
    ymax_vals = np.full(ymax_facets.shape, ymax_tag, np.intc)

    zmin_tag = 14
    zmin_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, zmin)
    zmin_vals = np.full(zmin_facets.shape, zmin_tag, np.intc)

    zmax_tag = 15
    zmax_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, zmax)
    zmax_vals = np.full(zmax_facets.shape, zmax_tag, np.intc)

    indices = np.hstack(
        (xmin_facets, xmax_facets, ymin_facets, ymax_facets, zmin_facets, zmax_facets)
    )
    values = np.hstack(
        (xmin_vals, xmax_vals, ymin_vals, ymax_vals, zmin_vals, zmax_vals)
    )

    indices, pos = np.unique(indices, return_index=True)
    marker = dolfinx.mesh.meshtags(mesh, tdim - 1, indices, values[pos])
    ds = ufl.Measure("ds", subdomain_data=marker)
    dx = ufl.dx(metadata={"quadrature_rule": "runtime"})

    mesh.topology.create_connectivity(tdim - 1, 0)
    f2n = mesh.topology.connectivity(tdim - 1, 0)
    xmin_nodes = f2n.links(xmin_facets)
    xmax_nodes = f2n.links(xmax_facets)
    ymin_nodes = f2n.links(ymin_facets)
    ymax_nodes = f2n.links(ymax_facets)
    zmin_nodes = f2n.links(zmin_facets)
    zmax_nodes = f2n.links(zmax_facets)

    # Complicated way of calculating qr_pts, but this illustrates the numbering
    qr_pts_ref = np.array(
        [
            [0.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 1.0],
        ]
    )
    xdiff = bbmax - bbmin
    x = mesh.geometry.x[:, 0:tdim]

    num_faces = 6
    qr_pts = np.empty((num_faces, tdim))
    qr_pts[0] = (np.mean(x[xmin_nodes], axis=0) - bbmin) / xdiff
    qr_pts[1] = (np.mean(x[xmax_nodes], axis=0) - bbmin) / xdiff
    qr_pts[2] = (np.mean(x[ymin_nodes], axis=0) - bbmin) / xdiff
    qr_pts[3] = (np.mean(x[ymax_nodes], axis=0) - bbmin) / xdiff
    qr_pts[4] = (np.mean(x[zmin_nodes], axis=0) - bbmin) / xdiff
    qr_pts[5] = (np.mean(x[zmax_nodes], axis=0) - bbmin) / xdiff

    for k in range(num_faces):
        assert np.all(abs(qr_pts[k] - qr_pts_ref[k]) < 1e-10)

    facet_size = np.empty(num_faces)
    facet_size[0] = facet_size[1] = xdiff[1] * xdiff[2]
    facet_size[2] = facet_size[3] = xdiff[0] * xdiff[2]
    facet_size[4] = facet_size[5] = xdiff[0] * xdiff[1]
    cell_volume = np.prod(xdiff)
    qr_w = facet_size / cell_volume

    tags = [xmin_tag, xmax_tag, ymin_tag, ymax_tag, zmin_tag, zmax_tag]

    for k in range(num_faces):
        qr_pts_local = np.expand_dims(qr_pts[k], axis=0)
        qr_w_local = np.expand_dims(qr_w[k], axis=(0, 1))
        qr_data = [(cells, qr_pts_local, qr_w_local)]
        b = cq.assemble_vector(dolfinx.fem.form(integrand * dx), qr_data)

        ds_local = ds(tags[k])
        b_exact = dolfinx.fem.petsc.assemble_vector(
            dolfinx.fem.form(integrand * ds_local)
        )

        for m, v in enumerate(b.array):
            assert abs(v - b_exact.array[m]) < 1e-10
