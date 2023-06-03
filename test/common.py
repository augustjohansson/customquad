import dolfinx
import ufl
from ufl import inner
import numpy as np
import FIAT
import libcutfemx
from numpy import sin, pi, exp


def rhs1(x):
    return x[0] ** 0


def rhs2(x):
    return x[0]


def rhs3(x):
    return sin(pi * x[0]) * sin(pi * x[1])


def rhs4(x):
    return exp(x[0] * x[1])


def assemble_scalar_test():
    pass


def assemble_vector_test(mesh, fiat_element, polynomial_order, quadrature_degree, rhs):
    # Setup integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
    v = ufl.TestFunction(V)
    f = dolfinx.fem.Function(V)
    f.interpolate(rhs)
    L_eqn = inner(f, v)

    # Runtime quadrature
    L = L_eqn * ufl.dx(metadata={"quadrature_rule": "runtime"})
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(num_cells)
    q = FIAT.create_quadrature(fiat_element, quadrature_degree)
    qr_pts = np.tile(q.get_points().flatten(), [num_cells, 1])
    qr_w = np.tile(q.get_weights().flatten(), [num_cells, 1])
    qr_n = qr_pts  # dummy
    b = libcutfemx.custom_assemble_vector(L, [(cells, qr_pts, qr_w, qr_n)])

    # Reference
    L_ref = L_eqn * ufl.dx
    b_ref = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L_ref))

    return b, b_ref


def assemble_matrix_test():
    pass
