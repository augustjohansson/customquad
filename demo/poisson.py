"""This demo requires algoim, a header-only C++ library found at

https://github.com/algoim/algoim/

and the c++ to python library cppyy (installable by pip). The location
of algoim can be set in algoim_utils.py.

"""

import dolfinx
import customquad
import ufl
from ufl import grad, inner, dot, jump, avg
from mpi4py import MPI
import numpy as np
from numpy import sin, pi
from petsc4py import PETSc
import argparse
import algoim_utils


# import os
# os.environ['CC'] = "/usr/lib/ccache/c++" # visible in this process + all children

parser = argparse.ArgumentParser()
parser.add_argument("-factor", type=int, default=1)
parser.add_argument("-algoim", action="store_true")
parser.add_argument("-reuse", action="store_true")
parser.add_argument("-betaN", type=float, default=10.0)
parser.add_argument("-betas", type=float, default=1.0)
args = parser.parse_args()
print("arguments:")
for arg in vars(args):
    print("\t", arg, getattr(args, arg))

resetdata = not args.reuse
filename = "qrdata.pickle"
do_gotools = not args.algoim
domain = "circle"
# domain = "square"
# domain = "sphere"

# Domain
if domain == "square":
    xmin = np.array([-0.033, -0.023])
    xmax = np.array([1.1, 1.1])
    L2_exact = 0.25
    volume_exact = 1.0
    area_exact = 4.0
    gdim = 2

elif domain == "circle":
    xmin = np.array([-1.11, -1.51])
    xmax = np.array([1.55, 1.22])
    L2_exact = 0.93705920078336
    volume_exact = pi
    area_exact = 2 * pi
    gdim = 2

elif domain == "sphere":
    assert args.algoim
    xmin = np.array([-1.11, -1.21, -1.23])
    xmax = np.array([1.23, 1.22, 1.11])
    L2_exact = 0.418879020470132  # 0.517767045525837
    volume_exact = 4 * pi / 3
    area_exact = 4 * pi
    gdim = 3

else:
    RuntimeError("Unknown domain", domain)

# Mesh
NN = np.array([args.factor] * gdim, dtype=np.int32)
if gdim == 2:
    cell_type = dolfinx.mesh.CellType.quadrilateral
    mesh_generator = dolfinx.mesh.create_rectangle
else:
    cell_type = dolfinx.mesh.CellType.hexahedron
    mesh_generator = dolfinx.mesh.create_box
print(f"{NN=}")
print(f"{xmin=}")
print(f"{xmax=}")
mesh = mesh_generator(MPI.COMM_WORLD, np.array([xmin, xmax]), NN, cell_type)
n = ufl.FacetNormal(mesh)
h = ufl.CellDiameter(mesh)
assert mesh.geometry.dim == gdim

# FEM
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Function(V)
g = dolfinx.fem.Function(V)


def exact_solution2(x, y):
    return sin(pi * x) * sin(pi * y)


def exact_solution(x):
    return sin(pi * x[0]) * sin(pi * x[1])


if gdim == 3:
    # def exact_solution2(x, y, z):
    #     return sin(pi*x)*sin(pi*y)*sin(pi*z)
    # def exact_solution(x):
    #     return sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])
    def exact_solution2(x, y, z):
        r = sqrt(x * x + y * y + z * z)
        return 1.0 - r

    def exact_solution(x):
        r = np.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
        return 1.0 - r


g.interpolate(exact_solution)


def rhs(x):
    # return gdim*pi*pi*exact_solution(x)
    return -2 / exact_solution(x)


f.interpolate(rhs)

# PDE
betaN = args.betaN
betas = args.betas
a_bulk = inner(grad(u), grad(v))
L_bulk = inner(f, v)
a_bdry = (
    -inner(dot(n, grad(u)), v) - inner(u, dot(n, grad(v))) + inner(betaN / h * u, v)
)
L_bdry = -inner(g, dot(n, grad(v))) + inner(betaN / h * g, v)
a_stab = betas * avg(h) * inner(jump(n, grad(u)), jump(n, grad(v)))

# Generate (or load) qr
degree = 4
print("Generate qr")
[
    cut_cells,
    uncut_cells,
    outside_cells,
    qr_pts,
    qr_w,
    qr_pts_bdry,
    qr_w_bdry,
    qr_n,
    xyz,
    xyz_bdry,
] = algoim_utils.generate_qr(mesh, NN, degree, filename, resetdata, domain)

# Set up cell tags and face tags
uncut_cell_tag = 1
cut_cell_tag = 2
outside_cell_tag = 3
ghost_penalty_tag = 4
celltags = customquad.utils.get_celltags(
    mesh,
    cut_cells,
    uncut_cells,
    outside_cells,
    uncut_cell_tag=uncut_cell_tag,
    cut_cell_tag=cut_cell_tag,
    outside_cell_tag=outside_cell_tag,
)
facetags = customquad.utils.get_facetags(
    mesh, cut_cells, outside_cells, ghost_penalty_tag=ghost_penalty_tag
)
breakpoint()
with dolfinx.cpp.io.XDMFFile(
    mesh.comm, "output/mesh" + str(args.factor) + ".xdmf", "w"
) as file:
    file.write_mesh(mesh)
    file.write_meshtags(celltags)

# Integration using custom assembler (i.e. integrals over cut cells,
# both cut bulk part and bdry part)
# dx_cut = ufl.dx(metadata={"quadrature_rule": "runtime"}, domain=mesh) # Write like this or as below
dx_cut = ufl.Measure("dx", metadata={"quadrature_rule": "runtime"})
ds_cut = ufl.Measure(
    "dx", subdomain_data=celltags, metadata={"quadrature_rule": "runtime"}
)
ac = a_bulk * dx_cut + a_bdry * ds_cut(cut_cell_tag)
Lc = L_bulk * dx_cut  # + L_bdry*ds_cut(cut_cell_tag)
qr_bulk = (cut_cells, qr_pts, qr_w)
qr_bdry = (cut_cells, qr_pts_bdry, qr_w_bdry, qr_n)
Ac = customquad.assemble_matrix(ac, [qr_bulk, qr_bdry])
Ac.assemble()
bc = customquad.assemble_vector(Lc, [qr_bulk])  # , qr_bdry])

# Integration using standard assembler (uncut cells, ghost penalty faces)
# dx_uncut = ufl.Measure("dx", subdomain_data=celltags, domain=mesh) # Write like this or as below
dx_uncut = ufl.dx(subdomain_data=celltags, domain=mesh)
dS = ufl.Measure("dS", subdomain_data=facetags, domain=mesh)
ax = a_bulk * dx_uncut(uncut_cell_tag) + a_stab * dS(ghost_penalty_tag)
Lx = L_bulk * dx_uncut(uncut_cell_tag)
Ax = dolfinx.fem.assemble.assemble_matrix(ax)
Ax.assemble()
bx = dolfinx.fem.assemble.assemble_vector(Lx)

# Add up
A = Ax + Ac
b = bx + bc

# Check inf
customquad.utils.dump("A.txt", A)
customquad.utils.dump("b.txt", b)
if not np.isfinite(b.array).all():
    RuntimeError()

if not np.isfinite(A.norm()):
    RuntimeError()

inactive_dofs = customquad.utils.get_inactive_dofs(V, cut_cells, uncut_cells)
A = customquad.utils.lock_inactive_dofs(inactive_dofs, A)
if not np.isfinite(A.norm()).all():
    RuntimeError()


def ksp_solve(A, b):
    ksp = PETSc.KSP().create(mesh.comm)
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
        mesh.comm, filename, "w", encoding=XDMFFile.Encoding.HDF5
    ) as xdmffile:
        xdmffile.write_mesh(mesh)
        xdmffile.write_function(u)


# Solve
vec = ksp_solve(A, b)
u = vec_to_function(vec, V, "u")
write("output/poisson" + str(args.factor) + ".xdmf", mesh, u)
if not np.isfinite(vec.array).all():
    RuntimeError("not finite")
if not np.isfinite(u.vector.array).all():
    RuntimeError("not finite")


def assemble(integrand):
    mc = customquad.assemble_scalar(integrand * dx_cut, [qr_bulk])
    # print(f"{mc=}")
    m = dolfinx.fem.assemble_scalar(integrand * dx_uncut(uncut_cell_tag))
    # print(f"{m=}")
    return m + mc


# L2 errors
L2_integrand = inner(u, u)
L2_val = assemble(L2_integrand)
L2_err = abs(L2_val - L2_exact) / L2_exact

# Check functional assembly
volume_func = customquad.assemble_scalar(
    1.0 * dx_cut, [qr_bulk]
) + dolfinx.fem.assemble.assemble_scalar(1.0 * dx_uncut(uncut_cell_tag))
area_func = customquad.assemble_scalar(1.0 * ds_cut(cut_cell_tag), [qr_bdry])
ve = abs(volume_exact - volume_func) / volume_exact
ae = abs(area_exact - area_func) / area_exact
print("functional volume error", ve)
print("functional area error", ae)

# Geometry errors
volume = customquad.utils.volume(xmin, xmax, NN, uncut_cells, qr_w)
area = customquad.utils.area(xmin, xmax, NN, qr_w_bdry)
volume_err = abs(volume_exact - volume) / volume_exact
area_err = abs(area_exact - area) / area_exact
print("qr volume error", volume_err)
print("qr area error", area_err)

# Evaluate solution in qr to see that there aren't any spikes
bb_tree = dolfinx.cpp.geometry.BoundingBoxTree(mesh, gdim)
pts = []
for x in [xyz, xyz_bdry]:
    xnp = np.array(x)
    xcc = xnp[cut_cells]
    # resize to [3,1] format for bbox utils (also for gdim=2)
    for xi in xcc:
        p = xi.copy()
        n = p.size // gdim
        p.resize(n, gdim)
        r = np.zeros((n, 3))
        r[:, 0:gdim] = p.copy()
        for ri in r:
            pts.append(ri)
pts = np.array(pts)
cells = []
for p in pts:
    cells = dolfinx.cpp.geometry.compute_collisions_point(bb_tree, p)
    cell = dolfinx.cpp.geometry.select_colliding_cells(mesh, cells, p, 1)
    # assert len(cell) > 0
    # #assert len(cell) == 1
    # if len(cell) != 1:
    #     #breakpoint()
    #     print("found", len(cell), "cells")
    cells.append(cell)
uvals = u.eval(pts, cells).flatten()
print("u in range", uvals.min(), uvals.max())

if gdim == 2:
    # Save coordinates and solution for plotting
    axis = "axis tight; grid on; xlabel x; ylabel y;"
    filename = "output/uu" + str(args.factor) + ".txt"
    uu = pts
    uu[:, 2] = uvals
    np.savetxt(filename, uu)
    print(f"uu=load('{filename}'); plot3(uu(:,1),uu(:,2),uu(:,3),'.');{axis}")

    # Save xy and error for plotting
    diff = pts
    diff[:, 2] = abs(exact_solution2(pts[:, 0], pts[:, 1]) - uvals)
    filename = "output/diff" + str(args.factor) + ".txt"
    np.savetxt(filename, diff)
    print(f"diff=load('{filename}'); plot3(diff(:,1),diff(:,2),diff(:,3),'.');{axis}")

# Print conv last
print(
    "conv", mesh.hmax(), L2_val, L2_err, volume, volume_err, area, area_err, args.factor
)
