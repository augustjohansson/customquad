import dolfinx
import libcutfemx
import ufl
from ufl import grad, inner
from mpi4py import MPI
import numpy
from numpy import sin, pi
import FIAT
from petsc4py import PETSc

# Clear cache
from shutil import rmtree

rmtree("/root/.cache/fenics", True)


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
mesh = dolfinx.RectangleMesh(
    MPI.COMM_WORLD,
    [numpy.array([0, 0, 0]), numpy.array([1, 1, 0])],
    [Nx, Ny],
    cell_type=cell_type,
)
cells = numpy.arange(Nx * Ny)

V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
v = ufl.TestFunction(V)

# f = dolfinx.Function(V)
# def rhs(x):
#     return x[0] #sin(pi*x[0])*sin(pi*x[1])
# f.interpolate(rhs)

# L_eqn = inner(f, v)
L_eqn = 1*v

L = L_eqn * ufl.dx(metadata={"quadrature_rule": "runtime"})

degree = 1
q = FIAT.create_quadrature(FIAT.reference_element.UFCQuadrilateral(), degree)
qr_pts = q.get_points().flatten()
qr_w = q.get_weights().flatten()

b = libcutfemx.custom_assemble_vector(L, [(cells, [qr_pts], [qr_w])])
print("custom b", b.array)
print("custom b norm", numpy.linalg.norm(b.array))

# Reference
L = L_eqn * ufl.dx
b = dolfinx.fem.assemble_vector(L)
print("ref b", b.array)
print("ref b norm", numpy.linalg.norm(b.array))
