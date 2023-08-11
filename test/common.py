import dolfinx
import ufl
from ufl import inner
import numpy as np
import FIAT
from numpy import sin, pi, exp
import customquad as cq
from mpi4py import MPI


def fcn1(x):
    return x[0] ** 0


def fcn2(x):
    return x[0]


def fcn3(x):
    return sin(pi * x[0]) * sin(pi * x[1])


def fcn4(x):
    return exp(x[0] * x[1])


def scalar_norm(x):
    return abs(x)


def vector_norm(x):
    return np.linalg.norm(x)


def matrix_norm(x):
    return x.norm()


def assemble_scalar_test(
    mesh, fiat_element, polynomial_order, quadrature_degree, fcn, perm
):
    # Setup integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
    f = dolfinx.fem.Function(V)
    f.interpolate(fcn)
    integrand = inner(f, f)

    # Runtime quadrature
    L = dolfinx.fem.form(integrand * ufl.dx(metadata={"quadrature_rule": "runtime"}))
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(num_cells)
    q = FIAT.create_quadrature(fiat_element, quadrature_degree)
    qr_pts = np.tile(q.get_points().flatten(), [num_cells, 1])
    qr_w = np.tile(q.get_weights().flatten(), [num_cells, 1])
    b = cq.assemble_scalar(L, [(cells, qr_pts, qr_w)], perm)

    # Reference
    L_ref = dolfinx.fem.form(integrand * ufl.dx)
    b_ref = dolfinx.fem.assemble_scalar(L_ref)

    return b, b_ref


def assemble_vector_test(
    mesh, fiat_element, polynomial_order, quadrature_degree, fcn, perm
):
    # Setup integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
    v = ufl.TestFunction(V)
    f = dolfinx.fem.Function(V)
    f.interpolate(fcn)
    integrand = inner(f, v)

    # Runtime quadrature
    L = dolfinx.fem.form(integrand * ufl.dx(metadata={"quadrature_rule": "runtime"}))
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(num_cells)
    q = FIAT.create_quadrature(fiat_element, quadrature_degree)
    qr_pts = np.tile(q.get_points().flatten(), [num_cells, 1])
    qr_w = np.tile(q.get_weights().flatten(), [num_cells, 1])
    b = cq.assemble_vector(L, [(cells, qr_pts, qr_w)], perm)

    # Reference
    L_ref = dolfinx.fem.form(integrand * ufl.dx)
    b_ref = dolfinx.fem.petsc.assemble_vector(L_ref)

    return b, b_ref


def assemble_matrix_test(
    mesh, fiat_element, polynomial_order, quadrature_degree, fcn, perm
):
    # Setup integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    integrand = inner(u, v)

    # Runtime quadrature
    L = dolfinx.fem.form(integrand * ufl.dx(metadata={"quadrature_rule": "runtime"}))
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(num_cells)
    q = FIAT.create_quadrature(fiat_element, quadrature_degree)
    qr_pts = np.tile(q.get_points().flatten(), [num_cells, 1])
    qr_w = np.tile(q.get_weights().flatten(), [num_cells, 1])
    A = cq.assemble_matrix(L, [(cells, qr_pts, qr_w)], perm)
    A.assemble()

    # Reference
    L_ref = dolfinx.fem.form(integrand * ufl.dx)
    A_ref = dolfinx.fem.petsc.assemble_matrix(L_ref)
    A_ref.assemble()

    return A, A_ref


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
