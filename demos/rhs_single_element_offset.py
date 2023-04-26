import dolfinx
import libcutfemx
import ufl
from ufl import grad, inner
from mpi4py import MPI
import numpy
from numpy import sin, pi, exp
import FIAT
from petsc4py import PETSc

# # Clear cache
# from shutil import rmtree

# rmtree("/root/.cache/fenics", True)


def rhs1(x):
    return x[0] ** 0


def rhs2(x):
    return x[0]


def rhs3(x):
    return sin(pi * x[0]) * sin(pi * x[1])


def rhs4(x):
    return exp(x[0] * x[1])


def ksp_solve(A, b):
    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    vec = b.copy()
    ksp.solve(b, vec)
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return vec


def vec_to_function(vec, V, tag="fcn"):
    uh = dolfinx.Function(V)
    uh.vector.setArray(vec.array)
    uh.name = tag
    return uh


def write(filename, mesh, u):
    from dolfinx.io import XDMFFile

    with XDMFFile(
        MPI.COMM_WORLD, filename, "w", encoding=XDMFFile.Encoding.HDF5
    ) as xdmffile:
        xdmffile.write_mesh(mesh)
        xdmffile.write_function(u)


cell_type = dolfinx.cpp.mesh.CellType.quadrilateral
Nx = 1
Ny = 1
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, numpy.array([[-1.5, -0.111], [0.1, 0.77]]), numpy.array([Nx, Ny]), cell_type
)
cells = numpy.arange(Nx * Ny)

V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
v = ufl.TestFunction(V)

for k, rhs in enumerate([rhs1, rhs2, rhs3, rhs4]):
    f = dolfinx.fem.Function(V)
    f.interpolate(rhs)

    L_eqn = inner(f, v)

    # Runtime quadrature
    L = L_eqn * ufl.dx(metadata={"quadrature_rule": "runtime"})
    degree = 2
    q = FIAT.create_quadrature(FIAT.reference_element.UFCQuadrilateral(), degree)
    qr_pts = q.get_points().flatten()
    qr_w = q.get_weights().flatten()
    qr_n = qr_pts  # dummy
    b = libcutfemx.custom_assemble_vector(L, [(cells, [qr_pts], [qr_w], [qr_n])])
    print("custom b", b.array)
    print("custom b norm", numpy.linalg.norm(b.array))

    # Reference
    Lref = L_eqn * ufl.dx
    bref = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Lref))
    print("bref", bref.array)
    print("bref norm", numpy.linalg.norm(bref.array))

    assert (
        numpy.linalg.norm(b.array - bref.array) / numpy.linalg.norm(bref.array) < 1e-10
    )
