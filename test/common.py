import dolfinx
import ufl
from ufl import inner
import numpy as np
import FIAT
import libcutfemx
from numpy import sin, pi, exp


def fcn1(x):
    return x[0] ** 0


def fcn2(x):
    return x[0]


def fcn3(x):
    return sin(pi * x[0]) * sin(pi * x[1])


def fcn4(x):
    return exp(x[0] * x[1])


def assemble_scalar_test(mesh, fiat_element, polynomial_order, quadrature_degree, fcn):
    # Setup integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
    v = ufl.TestFunction(V)
    f = dolfinx.fem.Function(V)
    f.interpolate(fcn)
    integrand = inner(f, f)

    # Runtime quadrature
    L = integrand * ufl.dx(metadata={"quadrature_rule": "runtime"})
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(num_cells)
    q = FIAT.create_quadrature(fiat_element, quadrature_degree)
    qr_pts = np.tile(q.get_points().flatten(), [num_cells, 1])
    qr_w = np.tile(q.get_weights().flatten(), [num_cells, 1])
    qr_n = qr_pts  # dummy
    b = libcutfemx.custom_assemble_scalar(L, [(cells, qr_pts, qr_w, qr_n)])

    # Reference
    L_ref = integrand * ufl.dx
    b_ref = dolfinx.fem.assemble_scalar(dolfinx.fem.form(L_ref))

    return b, b_ref


def assemble_vector_test(mesh, fiat_element, polynomial_order, quadrature_degree, fcn):
    # Setup integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
    v = ufl.TestFunction(V)
    f = dolfinx.fem.Function(V)
    f.interpolate(fcn)
    integrand = inner(f, v)

    # Runtime quadrature
    L = integrand * ufl.dx(metadata={"quadrature_rule": "runtime"})
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(num_cells)
    q = FIAT.create_quadrature(fiat_element, quadrature_degree)
    qr_pts = np.tile(q.get_points().flatten(), [num_cells, 1])
    qr_w = np.tile(q.get_weights().flatten(), [num_cells, 1])
    qr_n = qr_pts  # dummy
    b = libcutfemx.custom_assemble_vector(L, [(cells, qr_pts, qr_w, qr_n)])

    # Reference
    L_ref = integrand * ufl.dx
    b_ref = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L_ref))

    return b, b_ref


def assemble_matrix_test():
    # Setup integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    integrand = inner(u, v)

    # Runtime quadrature
    L = integrand * ufl.dx(metadata={"quadrature_rule": "runtime"})
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(num_cells)
    q = FIAT.create_quadrature(fiat_element, quadrature_degree)
    qr_pts = np.tile(q.get_points().flatten(), [num_cells, 1])
    qr_w = np.tile(q.get_weights().flatten(), [num_cells, 1])
    qr_n = qr_pts  # dummy
    b = libcutfemx.custom_assemble_matrix(L, [(cells, qr_pts, qr_w, qr_n)])

    # Reference
    L_ref = integrand * ufl.dx
    b_ref = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(L_ref))

    return b, b_ref
