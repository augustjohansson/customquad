import pytest
import dolfinx
from mpi4py import MPI
import numpy as np
import customquad as cq
import ufl
import common


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("use_dolfinx_mesh", [True, False])
def test_eval_at_nodes(use_dolfinx_mesh, gdim):

    N = [1] * gdim
    xmin = np.array([-0.25, -10.25, -3.33])[0:gdim]
    xmax = np.array([1.25, 17.5, 5.55])[0:gdim]

    if use_dolfinx_mesh:
        if gdim == 2:
            cell_type = dolfinx.mesh.CellType.quadrilateral
            mesh = dolfinx.mesh.create_rectangle(
                MPI.COMM_WORLD,
                np.array([xmin, xmax]),
                np.array(N),
                cell_type=cell_type,
            )
        else:
            cell_type = dolfinx.mesh.CellType.hexahedron
            mesh = dolfinx.mesh.create_box(
                MPI.COMM_WORLD,
                np.array([xmin, xmax]),
                np.array(N),
                cell_type,
            )
    else:
        polynomial_order = 1
        mesh = cq.create_mesh(np.array([xmin, xmax]), np.array(N), polynomial_order)

    num_cells = cq.utils.get_num_cells(mesh)
    cells = np.arange(num_cells)
    xdiff = xmax - xmin
    cell_volume = np.prod(xdiff)

    x = ufl.SpatialCoordinate(mesh)

    # fmt: off
    if gdim == 2:
        def fcn(x):
            return x[0] + 2 * x[1]
    else:
        def fcn(x):
            return x[0] + 2 * x[1] + 4 * x[2]
    # fmt: on

    dx = ufl.dx(metadata={"quadrature_rule": "runtime"})
    form = dolfinx.fem.form(fcn(x) * dx(domain=mesh))
    qr_w = np.expand_dims([1.0 / cell_volume], axis=0)

    for x in mesh.geometry.x:
        qr_pt = (x[0:gdim] - xmin) / xdiff
        qr_pts = np.expand_dims(qr_pt, axis=0)
        qr_data = [(cells, qr_pts, qr_w)]
        m = cq.assemble_scalar(form, qr_data)
        m_exact = fcn(x)
        assert abs(m - m_exact) / abs(m_exact) < 1e-10


def test_entities_to_geometry_2d():

    N = 1
    xmin = np.array([-0.25, -10.25])
    xmax = np.array([1.25, 17.5])
    polynomial_order = 1
    mesh = cq.create_mesh(np.array([xmin, xmax]), np.array([N, N]), polynomial_order)
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
    facets = [bottom_facets[0], top_facets[0], left_facets[0], right_facets[0]]

    for facet in facets:
        f2n = dolfinx.cpp.mesh.entities_to_geometry(mesh, tdim - 1, [facet], False)
        f2n2 = common.entities_to_geometry(mesh, tdim - 1, [facet])
        assert np.all(f2n == f2n2)


def test_entities_to_geometry_3d():

    N = [1] * 3
    bbxmin = np.array([-0.25, -10.25, -4.4])
    bbxmax = np.array([1.25, 17.5, 0.1])
    polynomial_order = 1
    mesh = cq.create_mesh(np.array([bbxmin, bbxmax]), np.array(N), polynomial_order)
    tdim = mesh.topology.dim

    def xmin(x):
        return np.isclose(x[0], bbxmin[0])

    def xmax(x):
        return np.isclose(x[0], bbxmax[0])

    def ymin(x):
        return np.isclose(x[1], bbxmin[1])

    def ymax(x):
        return np.isclose(x[1], bbxmax[1])

    def zmin(x):
        return np.isclose(x[2], bbxmin[2])

    def zmax(x):
        return np.isclose(x[2], bbxmax[2])

    fcns = [xmin, xmax, ymin, ymax, zmin, zmax]

    for k, fcn in enumerate(fcns):
        facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, fcn)
        f2n = dolfinx.cpp.mesh.entities_to_geometry(mesh, tdim - 1, facets, False)
        f2n2 = common.entities_to_geometry(mesh, tdim - 1, facets)
        assert np.all(f2n == f2n2)


def test_high_order_edge_numbering():

    N = 1
    xmin = np.array([-0.25, -10.25])
    xmax = np.array([1.25, 17.5])
    polynomial_order = 2
    mesh = cq.create_mesh(np.array([xmin, xmax]), np.array([N, N]), polynomial_order)
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
    facets = [bottom_facets[0], top_facets[0], left_facets[0], right_facets[0]]

    f2n = common.entities_to_geometry(mesh, 1, facets)

    bottom_nodes = f2n[0]
    top_nodes = f2n[1]
    left_nodes = f2n[2]
    right_nodes = f2n[3]
    bdry_nodes = [bottom_nodes, top_nodes, left_nodes, right_nodes]

    # See https://defelement.com/elements/lagrange.html
    ref_nodes = [[0, 1, 4], [2, 3, 7], [0, 2, 5], [1, 3, 6]]

    for k, ref in enumerate(ref_nodes):
        assert np.all(bdry_nodes[k] == ref)


def test_high_order_face_numbering():

    N = [1] * 3
    bbxmin = np.array([-0.25, -10.25, -4.4])
    bbxmax = np.array([1.25, 17.5, 0.1])
    polynomial_order = 2
    mesh = cq.create_mesh(np.array([bbxmin, bbxmax]), np.array(N), polynomial_order)
    tdim = mesh.topology.dim

    def xmin(x):
        return np.isclose(x[0], bbxmin[0])

    def xmax(x):
        return np.isclose(x[0], bbxmax[0])

    def ymin(x):
        return np.isclose(x[1], bbxmin[1])

    def ymax(x):
        return np.isclose(x[1], bbxmax[1])

    def zmin(x):
        return np.isclose(x[2], bbxmin[2])

    def zmax(x):
        return np.isclose(x[2], bbxmax[2])

    fcns = [xmin, xmax, ymin, ymax, zmin, zmax]

    # See https://defelement.com/elements/lagrange.html
    ref_nodes = [
        [0, 2, 4, 6, 9, 10, 14, 17, 22],
        [1, 3, 5, 7, 11, 12, 15, 18, 23],
        [0, 1, 4, 5, 8, 10, 12, 16, 21],
        [2, 3, 6, 7, 13, 14, 15, 19, 24],
        [0, 1, 2, 3, 8, 9, 11, 13, 20],
        [4, 5, 6, 7, 16, 17, 18, 19, 25],
    ]

    for k, fcn in enumerate(fcns):
        facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, fcn)
        f2n = common.entities_to_geometry(mesh, tdim - 1, facets)
        assert np.all(f2n[0] == ref_nodes[k])


@pytest.mark.parametrize("gdim", [2, 3])
def test_high_order_mesh_numbering_against_basix(gdim):

    N = [1] * gdim
    xmin = np.array([0.0, 0.0, 0.0])[0:gdim]
    xmax = np.array([1.0, 1.0, 1.0])[0:gdim]
    polynomial_order = 2
    mesh = cq.create_mesh(np.array([xmin, xmax]), np.array(N), polynomial_order)

    # From https://defelement.com/elements/lagrange.html
    if gdim == 2:
        x_basix = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [1.0, 0.5],
                [0.5, 1.0],
                [0.5, 0.5],
            ]
        )
    else:
        # https://defelement.com/elements/examples/hexahedron-lagrange-equispaced-2.html
        x_basix = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                # 8 - 10
                [0.5, 0, 0],
                [0, 0.5, 0],
                [0, 0, 0.5],
                # 11 - 13
                [1, 0.5, 0],
                [1, 0, 0.5],
                [0.5, 1, 0],
                # 14 - 16
                [0, 1, 0.5],
                [1, 1, 0.5],
                [0.5, 0, 1],
                # 17 - 19
                [0, 0.5, 1],
                [1, 0.5, 1],
                [0.5, 1, 1],
                # 20 - 22
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
                # 23 - 25
                [1, 0.5, 0.5],
                [0.5, 1, 0.5],
                [0.5, 0.5, 1],
                # 26
                [0.5, 0.5, 0.5],
            ]
        )
        assert len(np.unique(x_basix, axis=0)) == 27

    # For new elements, use this and adjust the perm
    # for i, y in enumerate(x_basix):
    #     for j, x in enumerate(mesh.geometry.x[:, 0:gdim]):
    #         if np.linalg.norm(x - y) < 1e-10:
    #             print(i, j)

    assert np.linalg.norm(mesh.geometry.x[:, 0:gdim] - np.array(x_basix)) < 1e-15


@pytest.mark.parametrize("gdim", [2, 3])
def test_eval_at_nodes_high_order(gdim):

    N = [1] * gdim
    xmin = np.array([-0.25, -10.25, -3.33])[0:gdim]
    xmax = np.array([1.25, 17.5, 5.55])[0:gdim]
    polynomial_order = 2
    mesh = cq.create_mesh(np.array([xmin, xmax]), np.array(N), polynomial_order)

    num_cells = cq.utils.get_num_cells(mesh)
    cells = np.arange(num_cells)
    xdiff = xmax - xmin
    cell_volume = np.prod(xdiff)

    x = ufl.SpatialCoordinate(mesh)

    # fmt: off
    if gdim == 2:
        def fcn(x):
            return x[0] + 4 * x[1]
    else:
        def fcn(x):
            return (1 + x[0] + 4 * x[1]) * (1 + 7 * x[2])
    # fmt: on

    dx = ufl.dx(metadata={"quadrature_rule": "runtime"})
    form = dolfinx.fem.form(fcn(x) * dx(domain=mesh))
    qr_w = np.expand_dims([1.0 / cell_volume], axis=0)

    m = [None] * len(mesh.geometry.x)
    m_exact = [None] * len(mesh.geometry.x)

    for k, x in enumerate(mesh.geometry.x):
        qr_pt = (x[0:gdim] - xmin) / xdiff
        qr_pts = np.expand_dims(qr_pt, axis=0)
        qr_data = [(cells, qr_pts, qr_w)]
        m[k] = cq.assemble_scalar(form, qr_data)

        m_exact[k] = fcn(x)

        assert abs(m[k] - m_exact[k]) / abs(m_exact[k]) < 1e-13


@pytest.mark.parametrize("use_dolfinx_mesh", [True, False])
def test_edges(use_dolfinx_mesh):

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

    tdim = mesh.topology.dim
    num_cells = cq.utils.get_num_cells(mesh)
    cells = np.arange(num_cells)

    x = ufl.SpatialCoordinate(mesh)

    def fcn(x):
        return x[0] + 2 * x[1]

    dx = ufl.dx(metadata={"quadrature_rule": "runtime"})
    form = dolfinx.fem.form(fcn(x) * dx(domain=mesh))

    num_facets = 4
    mesh.topology.create_connectivity(tdim - 1, 0)
    f2n = mesh.topology.connectivity(tdim - 1, 0)
    assert f2n.num_nodes == num_facets

    xdiff = xmax - xmin
    cell_volume = np.prod(xdiff)

    midpoint = [None] * num_facets
    facet_area = [None] * num_facets
    qr_pts = [None] * num_facets
    qr_w = [None] * num_facets
    m = [None] * num_facets
    m_exact = [None] * num_facets

    for k in range(num_facets):
        n = f2n.links(k)

        midpoint[k] = np.mean(mesh.geometry.x[n], axis=0)[0:tdim]
        qr_pts[k] = np.expand_dims((midpoint[k] - xmin) / (xmax - xmin), axis=0)

        facet_area[k] = np.linalg.norm(np.diff(mesh.geometry.x[n], axis=0))
        qr_w[k] = np.array([[facet_area[k] / cell_volume]])

        qr_data = [(cells, qr_pts[k], qr_w[k])]
        m[k] = cq.assemble_scalar(form, qr_data)
        m_exact[k] = fcn(midpoint[k]) * facet_area[k]

        assert abs(m[k] - m_exact[k]) / abs(m_exact[k]) < 1e-10


@pytest.mark.parametrize("use_dolfinx_mesh", [True, False])
def test_faces(use_dolfinx_mesh):

    N = 1
    xmin = np.array([-0.25, -10.25, -4.4])
    xmax = np.array([1.25, 17.5, 0.1])

    if use_dolfinx_mesh:
        cell_type = dolfinx.mesh.CellType.hexahedron
        mesh = dolfinx.mesh.create_box(
            MPI.COMM_WORLD,
            np.array([xmin, xmax]),
            np.array([N, N, N]),
            cell_type,
        )
    else:
        polynomial_order = 1
        mesh = cq.create_mesh(
            np.array([xmin, xmax]), np.array([N, N, N]), polynomial_order
        )

    tdim = mesh.topology.dim
    num_cells = cq.utils.get_num_cells(mesh)
    cells = np.arange(num_cells)

    x = ufl.SpatialCoordinate(mesh)

    def fcn(x):
        return x[0] + 2 * x[1] + 4 * x[2]

    dx = ufl.dx(metadata={"quadrature_rule": "runtime"})
    form = dolfinx.fem.form(fcn(x) * dx(domain=mesh))

    num_facets = 6
    mesh.topology.create_connectivity(tdim - 1, 0)
    f2n = mesh.topology.connectivity(tdim - 1, 0)
    assert f2n.num_nodes == num_facets

    xdiff = xmax - xmin
    cell_volume = np.prod(xdiff)

    midpoint = [None] * num_facets
    facet_area = [None] * num_facets
    qr_pts = [None] * num_facets
    qr_w = [None] * num_facets
    m = [None] * num_facets
    m_exact = [None] * num_facets

    for k in range(num_facets):
        n = f2n.links(k)

        midpoint[k] = np.mean(mesh.geometry.x[n], axis=0)[0:tdim]
        qr_pts[k] = np.expand_dims((midpoint[k] - xmin) / (xmax - xmin), axis=0)

        facet_area[k] = np.linalg.norm(np.diff(mesh.geometry.x[n], axis=0))
        qr_w[k] = np.array([[facet_area[k] / cell_volume]])

        qr_data = [(cells, qr_pts[k], qr_w[k])]
        m[k] = cq.assemble_scalar(form, qr_data)
        m_exact[k] = fcn(midpoint[k]) * facet_area[k]

        assert abs(m[k] - m_exact[k]) / abs(m_exact[k]) < 1e-10
