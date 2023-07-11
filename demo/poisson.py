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
from petsc4py import PETSc
import argparse
import algoim_utils

# import os
# os.environ['CC'] = "/usr/lib/ccache/c++" # visible in this process + all children

parser = argparse.ArgumentParser()
parser.add_argument("-factor", type=int, default=4)
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


def write(filename, mesh, data):
    with dolfinx.io.XDMFFile(
        mesh.comm,
        filename,
        "w",
    ) as xdmffile:
        xdmffile.write_mesh(mesh)
        if isinstance(data, dolfinx.mesh.MeshTagsMetaClass):
            xdmffile.write_meshtags(data)
        elif isinstance(data, dolfinx.fem.Function):
            xdmffile.write_function(data)
        else:
            breakpoint()


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
    volume_exact = np.pi
    area_exact = 2 * np.pi
    gdim = 2

elif domain == "sphere":
    assert args.algoim
    xmin = np.array([-1.11, -1.21, -1.23])
    xmax = np.array([1.23, 1.22, 1.11])
    L2_exact = 0.418879020470132  # 0.517767045525837
    volume_exact = 4 * np.pi / 3
    area_exact = 4 * np.pi
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
assert mesh.geometry.dim == gdim
assert gdim == 2

# Generate (or load) qr
degree = 4
algoim_opts = {"verbose": False}
print("Generate qr")
[
    cut_cells,
    uncut_cells,
    outside_cells,
    qr_pts0,
    qr_w0,
    qr_pts_bdry0,
    qr_w_bdry0,
    qr_n0,
    xyz,
    xyz_bdry,
] = algoim_utils.generate_qr(mesh, NN, degree, filename, resetdata, domain, algoim_opts)

print("num cells", customquad.utils.get_num_cells(mesh))
print("num cut_cells", len(cut_cells))
print("num uncut_cells", len(uncut_cells))
print("num outside_cells", len(outside_cells))

# Algoim creates (at the moment) quadrature rules for _all_ cells, not
# only the cut cells. Remove these empty entries
qr_pts = [qr_pts0[k] for k in cut_cells]
qr_w = [qr_w0[k] for k in cut_cells]
qr_pts_bdry = [qr_pts_bdry0[k] for k in cut_cells]
qr_w_bdry = [qr_w_bdry0[k] for k in cut_cells]
qr_n = [qr_n0[k] for k in cut_cells]

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

write("output/celltags" + str(args.factor) + ".xdmf", mesh, celltags)

# FEM
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Function(V)
g = dolfinx.fem.Function(V)
x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)
h = ufl.CellDiameter(mesh)


def exact_solution(x, do_ufl):
    if do_ufl:
        x = ufl.SpatialCoordinate(mesh)
        return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        # return 1.0 + 1e-14 * x[0]  # x[0] + x[1]
    else:
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        # return 1.0 + 1e-14 * x[0]  # x[0] + x[1]


g.interpolate(lambda x: exact_solution(x, False))
f = -ufl.div(ufl.grad(exact_solution(x, True)))
# g.interpolate(lambda x: 0.0 + 1e-14 * x[0])
# f.interpolate(lambda x: 1.0 + 1e-14 * x[0])

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

# Integration using standard assembler (uncut cells, ghost penalty faces)
dx_uncut = ufl.dx(subdomain_data=celltags, domain=mesh)
dS = ufl.dS(subdomain_data=facetags, domain=mesh)

ax = dolfinx.fem.form(
    a_bulk * dx_uncut(uncut_cell_tag) + a_stab * dS(ghost_penalty_tag)
)
Lx = dolfinx.fem.form(L_bulk * dx_uncut(uncut_cell_tag))

Ax = dolfinx.fem.petsc.assemble_matrix(ax)
t = dolfinx.common.Timer()
Ax.assemble()
print("Assemble interior took", t.elapsed()[0])
bx = dolfinx.fem.petsc.assemble_vector(Lx)

# Integration using custom assembler (i.e. integrals over cut cells,
# both cut bulk part and bdry part)
dx_cut = ufl.dx(metadata={"quadrature_rule": "runtime"}, domain=mesh)
ds_cut = ufl.dx(
    subdomain_data=celltags, metadata={"quadrature_rule": "runtime"}, domain=mesh
)

qr_bulk = [(cut_cells, qr_pts, qr_w)]
qr_bdry = [(cut_cells, qr_pts_bdry, qr_w_bdry, qr_n)]

# FIXME make sure we can assemble over many forms
form1 = dolfinx.fem.form(a_bulk * dx_cut)
form2 = dolfinx.fem.form(a_bdry * ds_cut(cut_cell_tag))
# forms = [form1, form2]

t = dolfinx.common.Timer()
Ac1 = customquad.assemble_matrix(form1, qr_bulk)
Ac1.assemble()
print("Runtime assemble bulk took", t.elapsed()[0])

t = dolfinx.common.Timer()
Ac2 = customquad.assemble_matrix(form2, qr_bdry)
Ac2.assemble()
print("Runtime assemble bdry took", t.elapsed()[0])

t = dolfinx.common.Timer()
A = Ax
A += Ac1
A += Ac2
print("Matrix += took", t.elapsed()[0])

L1 = dolfinx.fem.form(L_bulk * dx_cut)
bc1 = customquad.assemble_vector(L1, qr_bulk)
b = bx
b += bc1

# L2 = dolfinx.fem.form(L_bdry * ds_cut(cut_cell_tag))
# print("cut_cell_tag", cut_cell_tag)
# sd = ds_cut.subdomain_data()
# print("indices", sd.indices)
# print("values", sd.values)
# print("cut cells", np.where(sd.values == cut_cell_tag))
# print("should be the same as cut_cells", cut_cells)
# cut_cell_midpoints = dolfinx.mesh.compute_midpoints(mesh, gdim, cut_cells)
# np.set_printoptions(threshold=9999999)
# print(cut_cell_midpoints)

# areaform = dolfinx.fem.form(1.0 * ds_cut(cut_cell_tag))
# cut_area = customquad.assemble_scalar(areaform, qr_bdry)
# print(f"{cut_area}")
# totareaform = dolfinx.fem.form(1.0 * ds_cut)
# tot_cut_area = customquad.assemble_scalar(totareaform, qr_bdry)
# print(f"{tot_cut_area}")

# breakpoint()

L2 = dolfinx.fem.form(L_bdry * ds_cut)
bc2 = customquad.assemble_vector(L2, qr_bdry)
b += bc2

# Add up
# A = Ax + Ac1 + Ac2
# b = bx + bc1 + bc2

# Check inf
# customquad.utils.dump("output/A.txt", A)
# customquad.utils.dump("output/b.txt", b)
# customquad.utils.dump("output/bx.txt", bx)
# customquad.utils.dump("output/bc1.txt", bc1)
# customquad.utils.dump("output/bc2.txt", bc2)

if not np.isfinite(b.array).all():
    RuntimeError()

if not np.isfinite(A.norm()):
    RuntimeError()

t = dolfinx.common.Timer()
inactive_dofs = customquad.utils.get_inactive_dofs(V, cut_cells, uncut_cells)
print("Get inactive_dofs took", t.elapsed()[0])
t = dolfinx.common.Timer()
A = customquad.utils.lock_inactive_dofs(inactive_dofs, A)
print("Lock inactive dofs took", t.elapsed()[0])
# customquad.utils.dump("output/A_locked.txt", A)

if not np.isfinite(A.norm()).all():
    RuntimeError()


def ksp_solve(A, b):
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    vec = b.copy()
    t = dolfinx.common.Timer()
    ksp.solve(b, vec)
    print("Solve took", t.elapsed()[0])
    print("Matrix size", len(b.array))
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return vec


def vec_to_function(vec, V, tag="fcn"):
    uh = dolfinx.fem.Function(V)
    uh.vector.setArray(vec.array)
    uh.name = tag
    return uh


# Solve
vec = ksp_solve(A, b)
u = vec_to_function(vec, V, "u")
write("output/poisson" + str(args.factor) + ".xdmf", mesh, u)
if not np.isfinite(vec.array).all():
    RuntimeError("not finite")
if not np.isfinite(u.vector.array).all():
    RuntimeError("not finite")


def assemble(integrand):
    m_cut = customquad.assemble_scalar(dolfinx.fem.form(integrand * dx_cut), qr_bulk)
    m_uncut = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(integrand * dx_uncut(uncut_cell_tag))
    )
    print(f"{m_cut=}")
    print(f"{m_uncut=}")
    return m_cut + m_uncut


# L2 errors
L2_integrand = inner(u, u)
L2_val = assemble(L2_integrand)
L2_err = abs(L2_val - L2_exact) / L2_exact

# Check functional assembly
area_func = customquad.assemble_scalar(
    dolfinx.fem.form(1.0 * ds_cut(cut_cell_tag)), qr_bdry
)
volume_func = assemble(1.0)
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
bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, gdim)
flatten = lambda l: [item for sublist in l for item in sublist]
pts = np.reshape(flatten(xyz), (-1, 2))
pts_bdry = np.reshape(flatten(xyz_bdry), (-1, 2))
pts_midpt = dolfinx.mesh.compute_midpoints(mesh, gdim, uncut_cells)
pts = np.append(pts, pts_bdry, axis=0)
pts = np.append(pts, pts_midpt[:, 0:gdim], axis=0)
pts = np.hstack((pts, np.zeros((pts.shape[0], 1))))

cell_candidates = dolfinx.cpp.geometry.compute_collisions(bb_tree, pts)
cells = dolfinx.cpp.geometry.compute_colliding_cells(mesh, cell_candidates, pts)
uvals = u.eval(pts, cells.array).flatten()
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
    err = pts
    xy = [pts[:, 0], pts[:, 1]]
    err[:, 2] = abs(exact_solution(xy, False) - uvals)
    filename = "output/err" + str(args.factor) + ".txt"
    np.savetxt(filename, err)
    print(f"err=load('{filename}'); plot3(err(:,1),err(:,2),err(:,3),'.');{axis}")

# Print conv last
h = dolfinx.cpp.mesh.h(mesh, mesh.topology.dim, cut_cells)
print("conv")
print(max(h), L2_val, L2_err, volume, volume_err, area, area_err, args.factor)
