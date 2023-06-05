import dolfinx
import libcutfemx
import ufl
from ufl import grad, inner
from mpi4py import MPI
import numpy as np
from numpy import sin, pi
import FIAT
from petsc4py import PETSc

# # Clear cache
# from shutil import rmtree

# rmtree("/root/.cache/fenics", True)


def mumps_solve(A, b):
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
    uh = dolfinx.fem.Function(V)
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
Nx = 10
Ny = 10
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([1, 1])],
    [Nx, Ny],
    cell_type=cell_type,
)
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cells = np.arange(num_cells)

V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Function(V)


def rhs(x):
    return sin(pi * x[0]) * sin(pi * x[1])


f.interpolate(rhs)

a_eqn = inner(grad(u), grad(v))
L_eqn = inner(f, v)

# BC
tdim = mesh.topology.dim
mesh.topology.create_entities(tdim - 1)
mesh.topology.create_connectivity(tdim - 1, tdim)
exterior_facets = dolfinx.cpp.mesh.exterior_facet_indices(mesh.topology)
dofs = dolfinx.fem.locate_dofs_topological(
    V=V, entity_dim=tdim - 1, entities=exterior_facets
)
bc = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V=V)

# # Runtime quadrature
# metadata = {"quadrature_rule": "runtime"}
# a = a_eqn * ufl.dx(metadata=metadata)
# L = L_eqn * ufl.dx(metadata=metadata)

# degree = 2
# q = FIAT.create_quadrature(FIAT.reference_element.UFCQuadrilateral(), degree)
# qr_pts = np.tile(q.get_points().flatten(), [num_cells, 1])
# qr_w = np.tile(q.get_weights().flatten(), [num_cells, 1])
# qr_n = qr_pts  # dummy
# qr_data = [(cells, qr_pts, qr_w, qr_n)]

# A = libcutfemx.custom_assemble_matrix(a, qr_data)
# A.assemble()
# b = libcutfemx.custom_assemble_vector(L, qr_data)
# libcutfemx.utils.dump("/tmp/Acustom.txt", A, True)
# print("custom A norm", A.norm())
# print("custom b norm", np.linalg.norm(b.array))

# vec = mumps_solve(A, b)
# u = vec_to_function(vec, V, "u")
# print("custom u", u.vector.array)
# write("poisson.xdmf", mesh, u)


# Reference
a = dolfinx.fem.form(a_eqn * ufl.dx)
L = dolfinx.fem.form(L_eqn * ufl.dx)

A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
A.assemble()
b = dolfinx.fem.petsc.assemble_vector(L)
dolfinx.fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b, [bc])
libcutfemx.utils.dump("/tmp/Aref.txt", A, True)
print("ref A norm", A.norm())
print("ref b norm", np.linalg.norm(b.array))

vec = mumps_solve(A, b)
u = vec_to_function(vec, V, "uref")
print("ref u", u.vector.array)
write("poisson_ref.xdmf", mesh, u)
