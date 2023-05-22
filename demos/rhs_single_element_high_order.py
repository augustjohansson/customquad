import basix
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


# See test_higher_order_mesh.py:test_quadrilateral_mesh
def coord_to_vertex(x, y):
    return (order + 1) * y + x


def get_points(order):
    points = []
    points += [[i / order, 0] for i in range(order + 1)]
    for j in range(1, order):
        points += [[i / order + 0.1, j / order] for i in range(order + 1)]
    points += [[j / order, 1] for j in range(order + 1)]
    return points


def get_cells(order):
    cell = [
        coord_to_vertex(i, j)
        for i, j in [(0, 0), (order, 0), (0, order), (order, order)]
    ]
    if order > 1:
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, order))

        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(i, j))
    return [cell]


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
order = 2
points = get_points(order)
cells = get_cells(order)
domain = ufl.Mesh(
    basix.ufl_wrapper.create_vector_element(
        "Q",
        "quadrilateral",
        order,
        gdim=2,
        lagrange_variant=basix.LagrangeVariant.equispaced,
    )
)
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain)

tdim = 2
num_cells = mesh.topology.index_map(tdim).size_local
cells = numpy.arange(num_cells)

V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", order))
v = ufl.TestFunction(V)

for k, rhs in enumerate([rhs1, rhs2, rhs3, rhs4]):
    f = dolfinx.fem.Function(V)
    f.interpolate(rhs)

    L_eqn = inner(f, v)

    # Runtime quadrature
    L = L_eqn * ufl.dx(metadata={"quadrature_rule": "runtime"})
    degree = 6
    q = FIAT.create_quadrature(FIAT.reference_element.UFCQuadrilateral(), degree)
    print(f"degree={degree} gave", len(q.get_weights()), "quadrature points")
    qr_pts = numpy.tile(q.get_points().flatten(), [num_cells, 1])
    qr_w = numpy.tile(q.get_weights().flatten(), [num_cells, 1])
    qr_n = qr_pts  # dummy
    b = libcutfemx.custom_assemble_vector(L, [(cells, qr_pts, qr_w, qr_n)])
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
