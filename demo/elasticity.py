from contextlib import ExitStack
import argparse
import dolfinx
import customquad
import ufl
from ufl import grad, inner, dot, jump, avg
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import algoim_utils

# Setup arguments
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=16)
parser.add_argument("-algoim", action="store_true")
parser.add_argument("-betaN", type=float, default=10.0)
parser.add_argument("-betas", type=float, default=1.0)
parser.add_argument("-domain", type=str, default="circle")
parser.add_argument("-verbose", action="store_true")
args = parser.parse_args()
print("arguments:")
for arg in vars(args):
    print("\t", arg, getattr(args, arg))


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
            raise RuntimeError("Unknown data type")


# Domain
if args.domain == "circle":
    xmin = np.array([-1.11, -1.51])
    xmax = np.array([1.55, 1.22])
    volume_exact = np.pi
    area_exact = 2 * np.pi

elif args.domain == "sphere":
    xmin = np.array([-1.11, -1.51, -1.23])
    xmax = np.array([1.55, 1.22, 1.11])
    volume_exact = 4 * np.pi / 3
    area_exact = 4 * np.pi

else:
    RuntimeError("Unknown domain", args.domain)

gdim = len(xmin)

# Mesh
NN = np.array([args.N] * gdim, dtype=np.int32)
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

# Generate qr
degree = 1
algoim_opts = {"verbose": args.verbose}
t = dolfinx.common.Timer()
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
] = algoim_utils.generate_qr(mesh, NN, degree, args.domain, algoim_opts)

print("Generating qr took", t.elapsed()[0])
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

# Write cell tags
write("output/celltags" + str(args.N) + ".xdmf", mesh, celltags)

# Data
E = 1
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def epsilon(v):
    return ufl.sym(grad(v))


def sigma(v):
    return 2.0 * mu * epsilon(v) + lmbda * ufl.tr(epsilon(v)) * ufl.Identity(len(v))


# FEM
V = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)
h = ufl.CellDiameter(mesh)
g = dolfinx.fem.Function(V)
f = dolfinx.fem.Function(V)

if gdim == 2:
    # fmt:off
    # LX = LY = 1
    # u_expr = lambda x: np.stack((
    #     np.cos((np.pi*x[1])/LY)*np.sin((np.pi*x[0])/LX),
    #     np.sin((np.pi*x[0])/LX)*np.sin((np.pi*x[1])/LY)
    # ))

    # u_ufl = ufl.as_vector([
    #     ufl.cos((ufl.pi*x[1])/LY)*ufl.sin((ufl.pi*x[0])/LX),
    #     ufl.sin((ufl.pi*x[0])/LX)*ufl.sin((ufl.pi*x[1])/LY)
    # ])

    # f_expr = lambda x: np.stack((
    #     1.0/(LX*LX)*1.0/(LY*LY)*(np.pi*np.pi)*np.cos((np.pi*x[1])/LY)*((LY*LY)*lmbda*np.sin((np.pi*x[0])/LX)+(LX*LX)*mu*np.sin((np.pi*x[0])/LX)+(LY*LY)*mu*np.sin((np.pi*x[0])/LX)*2.0-LX*LY*lmbda*np.cos((np.pi*x[0])/LX)-LX*LY*mu*np.cos((np.pi*x[0])/LX)),
    #     1.0/(LX*LX)*1.0/(LY*LY)*(np.pi*np.pi)*np.sin((np.pi*x[1])/LY)*((LX*LX)*lmbda*np.sin((np.pi*x[0])/LX)+(LX*LX)*mu*np.sin((np.pi*x[0])/LX)*2.0+(LY*LY)*mu*np.sin((np.pi*x[0])/LX)+LX*LY*lmbda*np.cos((np.pi*x[0])/LX)+LX*LY*mu*np.cos((np.pi*x[0])/LX))
    # ))
    # energy_exact = 28.646314904123273

    # u_expr = lambda x: np.stack((
    #     np.cos(x[1]*np.pi)*np.sin(x[0]*np.pi),
    #     np.cos(x[0]*np.pi)*np.sin(x[1]*np.pi)
    # ))
    # u_ufl = ufl.as_vector([
    #     ufl.cos(x[1]*ufl.pi)*ufl.sin(x[0]*ufl.pi),
    #     ufl.cos(x[0]*ufl.pi)*ufl.sin(x[1]*ufl.pi)
    # ])
    # f_expr = lambda x: np.stack((
    #     2*np.pi**2*np.cos(x[1]*np.pi)*np.sin(x[0]*np.pi)*(lmbda + 2*mu),
    #     2*np.pi**2*np.cos(x[0]*np.pi)*np.sin(x[1]*np.pi)*(lmbda + 2*mu)
    # ))

    u_expr = lambda x: np.stack((
        x[0]*x[0],
        x[1]*x[1]
    ))
    u_ufl = ufl.as_vector([
        x[0]*x[0],
        x[1]*x[1]
    ])
    f_expr = lambda x: np.stack((
        (- 2*lmbda - 4*mu)*np.ones(x.shape[1]),
        (- 2*lmbda - 4*mu)*np.ones(x.shape[1])
    ))
    energy_exact = -4.229067033679951

    # u_expr = lambda x: np.stack((
    #     x[0]*x[0]+x[1]*x[1] - 1,
    #     x[0]*x[0]+x[1]*x[1] - 1,
    # ))
    # u_ufl = ufl.as_vector([
    #     x[0]*x[0]+x[1]*x[1] - 1,
    #     x[0]*x[0]+x[1]*x[1] - 1,
    # ])
    # f_expr = lambda x: np.stack((
    #     np.ones(x.shape[1]),
    #     np.ones(x.shape[1])
    # ))
    # energy_exact = 99

    # fmt: on


g.interpolate(u_expr)
f.interpolate(f_expr)

# PDE
betaN = args.betaN
betas = args.betas

a_bulk = inner(sigma(u), grad(v))
L_bulk = inner(f, v)

a_bdry = (
    -inner(dot(sigma(u), n), v)
    - inner(u, dot(sigma(v), n))
    + inner(betaN / h * (2 * mu + lmbda) * u, v)
)
L_bdry = -inner(g, dot(sigma(v), n)) + inner(betaN / h * (2 * mu + lmbda) * g, v)

a_stab = betas * avg(h) * inner(jump(sigma(u)), jump(sigma(v)))

# Standard measures
dx_uncut = ufl.dx(subdomain_data=celltags, domain=mesh)
dS = ufl.dS(subdomain_data=facetags, domain=mesh)

# Integration using standard assembler (uncut cells, ghost penalty
# faces)
ax = dolfinx.fem.form(
    a_bulk * dx_uncut(uncut_cell_tag) + a_stab * dS(ghost_penalty_tag)
)
t = dolfinx.common.Timer()
Ax = dolfinx.fem.petsc.assemble_matrix(ax)
Ax.assemble()
print("Assemble interior took", t.elapsed()[0])
Lx = dolfinx.fem.form(L_bulk * dx_uncut(uncut_cell_tag))
bx = dolfinx.fem.petsc.assemble_vector(Lx)

# Integration using custom assembler (i.e. integrals over cut cells,
# both cut bulk part and bdry part)
md = {"quadrature_rule": "runtime"}
dx_cut = ufl.dx(metadata=md, domain=mesh)
ds_cut = ufl.dx(subdomain_data=celltags, metadata=md, domain=mesh)

qr_bulk = [(cut_cells, qr_pts, qr_w)]
qr_bdry = [(cut_cells, qr_pts_bdry, qr_w_bdry, qr_n)]

# FIXME make sure we can assemble over many forms
form1 = dolfinx.fem.form(a_bulk * dx_cut)
form2 = dolfinx.fem.form(a_bdry * ds_cut(cut_cell_tag))

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

L2 = dolfinx.fem.form(L_bdry * ds_cut)
bc2 = customquad.assemble_vector(L2, qr_bdry)
b += bc2

customquad.utils.dump("output/A.txt", A)
customquad.utils.dump("output/b.txt", b)
customquad.utils.dump("output/bx.txt", bx)
customquad.utils.dump("output/bc1.txt", bc1)
customquad.utils.dump("output/bc2.txt", bc2)

assert np.isfinite(b.array).all()
assert np.isfinite(A.norm())

# Lock inactive dofs
t = dolfinx.common.Timer()
inactive_dofs = customquad.utils.get_inactive_dofs(V, cut_cells, uncut_cells)
print("Get inactive_dofs took", t.elapsed()[0])
t = dolfinx.common.Timer()
A = customquad.utils.lock_inactive_dofs(inactive_dofs, A)
print("Lock inactive dofs took", t.elapsed()[0])
if args.verbose:
    customquad.utils.dump("output/A_locked.txt", A)
assert np.isfinite(A.norm()).all()

uh = dolfinx.fem.Function(V)

if gdim == 2:

    def ksp_solve(A, b):
        # Direct solver using mumps
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

    vec = ksp_solve(A, b)
    energy = vec.dot(b)
    energy_err = abs(energy - energy_exact)
    print("Energy", "{:e}".format(energy))
    uh.vector.setArray(vec.array)

elif gdim == 3:

    def build_nullspace(V):
        """Build PETSc nullspace for 3D elasticity"""

        # Create list of vectors for building nullspace
        index_map = V.dofmap.index_map
        bs = V.dofmap.index_map_bs
        ns = [dolfinx.la.create_petsc_vector(index_map, bs) for i in range(6)]
        with ExitStack() as stack:
            vec_local = [stack.enter_context(x.localForm()) for x in ns]
            basis = [np.asarray(x) for x in vec_local]

            # Get dof indices for each subspace (x, y and z dofs)
            dofs = [V.sub(i).dofmap.list.array for i in range(3)]

            # Build the three translational rigid body modes
            for i in range(3):
                basis[i][dofs[i]] = 1.0

            # Build the three rotational rigid body modes
            x = V.tabulate_dof_coordinates()
            dofs_block = V.dofmap.list.array
            x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
            basis[3][dofs[0]] = -x1
            basis[3][dofs[1]] = x0
            basis[4][dofs[0]] = x2
            basis[4][dofs[2]] = -x0
            basis[5][dofs[2]] = x1
            basis[5][dofs[1]] = -x2

        # Orthonormalise the six vectors
        dolfinx.la.orthonormalize(ns)
        assert dolfinx.la.is_orthonormal(ns)

        return PETSc.NullSpace().create(vectors=ns)

    # Solve as in demo_elasticity.py in dolfinx
    null_space = build_nullspace(V)
    A.setNearNullSpace(null_space)

    # Set solver options
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-10
    opts["pc_type"] = "gamg"

    # Use Chebyshev smoothing for multigrid
    opts["mg_levels_ksp_type"] = "chebyshev"
    opts["mg_levels_pc_type"] = "jacobi"

    # Improve estimate of eigenvalues for Chebyshev smoothing
    opts["mg_levels_esteig_ksp_type"] = "cg"
    opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

    # Create PETSc Krylov solver and turn convergence monitoring on
    solver = PETSc.KSP().create(mesh.comm)
    solver.setFromOptions()

    # Set matrix operator
    solver.setOperators(A)

    # Set a monitor, solve linear system, and display the solver
    # configuration
    solver.setMonitor(
        lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
    )
    solver.solve(b, uh.vector)
    solver.view()

    # Scatter forward the solution vector to update ghost values
    uh.x.scatter_forward()


write("output/displacement" + str(args.N) + ".xdmf", mesh, uh)


def assemble(integrand):
    m_cut = customquad.assemble_scalar(dolfinx.fem.form(integrand * dx_cut), qr_bulk)
    m_uncut = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(integrand * dx_uncut(uncut_cell_tag))
    )
    return m_cut + m_uncut


# L2 errors: beware of cancellation
L2_integrand = (uh - u_ufl) ** 2
L2_err = np.sqrt(assemble(L2_integrand))

# H10 errors
H10_integrand = (grad(uh) - grad(u_ufl)) ** 2
H10_err = np.sqrt(assemble(H10_integrand))

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
pts = np.reshape(flatten(xyz), (-1, gdim))
pts_bdry = np.reshape(flatten(xyz_bdry), (-1, gdim))
pts_midpt = dolfinx.mesh.compute_midpoints(mesh, gdim, uncut_cells)
pts = np.append(pts, pts_bdry, axis=0)
pts = np.append(pts, pts_midpt[:, 0:gdim], axis=0)
if gdim == 2:
    # Pad with zero column
    pts = np.hstack((pts, np.zeros((pts.shape[0], 1))))

cell_candidates = dolfinx.cpp.geometry.compute_collisions(bb_tree, pts)
cells = dolfinx.cpp.geometry.compute_colliding_cells(mesh, cell_candidates, pts)
uh_vals = uh.eval(pts, cells.array)
uh_x = uh_vals[:, 0]
uh_y = uh_vals[:, 1]
print("uh in range", min(uh_x), max(uh_x), min(uh_y), max(uh_y))

if gdim == 2:
    # Save coordinates and solution for plotting
    filename = "output/uu" + str(args.N) + ".txt"
    uu = np.empty((len(uh_x), 4))
    uu[:, 0:2] = pts[:, 0:2]
    uu[:, 2] = uh_x
    uu[:, 3] = uh_y
    np.savetxt(filename, uu)

    # Save xy and error for plotting
    err = uu
    err[:, 0:2] = pts[:, 0:2]
    xy = [pts[:, 0], pts[:, 1]]
    zz = u_expr(xy)
    err[:, 2] = abs(zz[0] - uh_x)
    err[:, 3] = abs(zz[1] - uh_y)
    filename = "output/err" + str(args.N) + ".txt"
    np.savetxt(filename, err)
    max_err_x = max(err[:, 2])
    max_err_y = max(err[:, 3])
    print("max err x", max_err_x)
    print("max err y", max_err_y)


# Print
h = dolfinx.cpp.mesh.h(mesh, mesh.topology.dim, cut_cells)
conv = np.array(
    [
        max(h),
        L2_err,
        H10_err,
        energy,
        energy_err,
        max_err_x,
        max_err_y,
        volume,
        volume_err,
        area,
        area_err,
        args.N,
    ],
)

print(conv)
np.savetxt("output/conv" + str(args.N) + ".txt", conv.reshape(1, conv.shape[0]))
