import pytest
import dolfinx
import numpy as np
import customquad as cq
import ufl
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
def test_quads_assembly(assembler, norm, N, xmin, xmax, fcn):

    polynomial_order = 2
    quadrature_degree = 4
    mesh = cq.create_mesh(np.array([xmin, xmax]), np.array(N), polynomial_order)
    b, b_ref = assembler(mesh, polynomial_order, quadrature_degree, fcn)
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

    polynomial_order = 2
    quadrature_degree = 4
    mesh = cq.create_mesh(np.array([xmin, xmax]), np.array(N), polynomial_order)
    b, b_ref = assembler(mesh, polynomial_order, quadrature_degree, fcn)
    assert norm(b - b_ref) / norm(b_ref) < 1e-10


@pytest.mark.parametrize("gdim", [2, 3])
def test_dirac_property(gdim):

    N = [1] * gdim
    xmin = np.array([-0.25, -10.25, -4.4])[0:gdim]
    xmax = np.array([1.25, 17.5, 0.1])[0:gdim]

    polynomial_order = 2
    mesh = cq.create_mesh(np.array([xmin, xmax]), np.array(N), polynomial_order)

    num_cells = cq.utils.get_num_cells(mesh)
    cells = np.arange(num_cells)

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
    v = ufl.TestFunction(V)
    integrand = 1 * v
    dx = ufl.dx(metadata={"quadrature_rule": "runtime"})
    form = dolfinx.fem.form(integrand * dx)

    xdiff = xmax - xmin
    cell_volume = np.prod(xdiff)
    qr_w = np.array([[1.0 / cell_volume]])

    for k, y in enumerate(V.tabulate_dof_coordinates()):
        qr = (y[0:gdim] - xmin) / xdiff
        qr_pts = np.expand_dims(qr, axis=0)
        b = cq.assemble_vector(form, [(cells, qr_pts, qr_w)])
        b.array[k] -= 1

        assert np.linalg.norm(b.array, np.inf) < 1e-14


def test_edge_integral():

    # Test bdry integral with basis function
    N = 1
    xmin = np.array([-0.25, -10.25])
    xmax = np.array([1.25, 17.5])

    polynomial_order = 2
    mesh = cq.create_mesh(np.array([xmin, xmax]), np.array([N, N]), polynomial_order)

    tdim = mesh.topology.dim
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

    tags = [bottom_tag, top_tag, left_tag, right_tag]

    qr0 = 0.5 * (1.0 - 1.0 / np.sqrt(3))
    qr1 = 1 - qr0
    qr_pts = np.empty((4, 4))
    qr_pts[0] = np.array([qr0, 0.0, qr1, 0.0])
    qr_pts[1] = np.array([qr0, 1.0, qr1, 1.0])
    qr_pts[2] = np.array([0.0, qr0, 0.0, qr1])
    qr_pts[3] = np.array([1.0, qr0, 1.0, qr1])

    xdiff = xmax - xmin
    facet_size = np.array([xdiff[0], xdiff[0], xdiff[1], xdiff[1]])
    cell_volume = np.prod(xdiff)
    qr_w = np.transpose(np.tile(0.5 * facet_size / cell_volume, (2, 1))).copy()

    for k, tag in enumerate(tags):
        qr_pts_local = np.expand_dims(qr_pts[k], axis=0)
        qr_w_local = np.expand_dims(qr_w[k], axis=0)

        b = cq.assemble_vector(
            dolfinx.fem.form(integrand * dx),
            [(cells, qr_pts_local, qr_w_local)],
        )

        b_exact = dolfinx.fem.petsc.assemble_vector(
            dolfinx.fem.form(integrand * ds(tag))
        )

        for m, y in enumerate(b.array):
            assert abs(y - b_exact.array[m]) < 1e-10
