import dolfinx
import numba
import numpy as np
from petsc4py import PETSc


def get_num_entities(mesh, tdim):
    # Create all connectivities manually (it used to exist a
    # create_connectivity_all function)
    for d0 in range(tdim):
        for d1 in range(tdim):
            mesh.topology.create_connectivity(d0, d1)
    num_owned_entities = mesh.topology.index_map(tdim).size_local
    num_ghost_entities = mesh.topology.index_map(tdim).num_ghosts
    num_entities = num_owned_entities + num_ghost_entities
    return num_entities


def get_num_cells(mesh):
    tdim = mesh.topology.dim
    return get_num_entities(mesh, tdim)


def get_num_faces(mesh):
    tdim = mesh.topology.dim
    return get_num_entities(mesh, tdim - 1)


def get_num_nodes(mesh):
    return get_num_entities(mesh, 0)


def get_dofs(V):
    """
    customquad.assemble
    (Pdb++) V
    <dolfinx.cpp.fem.FunctionSpace object at 0x7fb5163b5eb0>
    <dolfinx.cpp.fem.DofMap object at 0x7fb521c7ae30>
    have list()

    but if type(V) = <class 'dolfinx.fem.function.FunctionSpace'>
    <dolfinx.fem.dofmap.DofMap object at 0x7fe5f7511360>
    have V.dofmap.list

    """
    num_cells = get_num_cells(V.mesh)
    bs = V.dofmap.index_map_bs
    num_loc_dofs = V.dofmap.dof_layout.num_dofs * bs

    if bs == 1:
        try:
            dofs = V.dofmap.list().array.reshape(num_cells, num_loc_dofs)
        except:
            dofs = V.dofmap.list.array.reshape(num_cells, num_loc_dofs)
    else:
        dofs = np.ndarray((num_cells, num_loc_dofs), np.int32)
        # r = np.arange(num_loc_dofs)
        # FIXME vectorize
        for cell in range(num_cells):
            for i, dof in enumerate(V.dofmap.cell_dofs(cell)):
                for j in range(bs):
                    dofs[cell, i * bs + j] = dof * bs + j
    return dofs, num_loc_dofs


def get_vertices(mesh):
    coords = mesh.geometry.x
    gdim = mesh.geometry.dim
    num_cells = get_num_cells(mesh)
    vertices = mesh.geometry.dofmap.array.reshape(num_cells, -1)
    return vertices, coords, gdim


def get_inactive_dofs(V, cut_cells, uncut_cells):
    dofs, _ = get_dofs(V)
    num_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    all_dofs = np.arange(num_dofs)
    for cells in [cut_cells, uncut_cells]:
        for cell in cells:
            all_dofs[dofs[cell, :]] = -1
    inactive_dofs = np.arange(num_dofs, dtype=np.int32)[all_dofs > -1]
    return inactive_dofs


def lock_inactive_dofs(inactive_dofs, A):
    nnz = np.ones(A.size[0], dtype=np.int32)
    A0 = PETSc.Mat().createAIJ(A.size, nnz=nnz, comm=A.comm)
    diag = A.createVecLeft()
    diag.array[:] = 0.0
    diag.array[inactive_dofs] = 1.0
    A0.setDiagonal(diag)
    A0.assemble()
    A += A0

    # check diagonal
    ad = A.getDiagonal()
    if (ad.array == 0).any():
        zeros = np.where(ad.array == 0)
        print("zero", zeros[0])
        for i in zeros[0]:
            A.setValue(i, i, 1.0)
        A.assemble()
        raise RuntimeError("Zeros on the diagonal should not happen")

    return A


def dump(filename, A, do_print=False):
    print(f"dump to {filename}")

    if isinstance(A, PETSc.Mat):
        assert A.assembled
        f = open(filename, "w")
        for r in range(A.size[0]):
            cols, vals = A.getRow(r)
            for i in range(len(cols)):
                s = str(r) + " " + str(cols[i]) + " " + str(vals[i]) + "\n"
                f.write(s)
                if do_print:
                    print(s, end="")
        f.close()
    else:
        f = open(filename, "w")
        np.savetxt(f, A.array)
        f.close()


def get_celltags(
    mesh,
    cut_cells,
    uncut_cells,
    outside_cells,
    outside_cell_tag=0,
    uncut_cell_tag=1,
    cut_cell_tag=2,
):
    assert outside_cell_tag != uncut_cell_tag
    assert outside_cell_tag != cut_cell_tag
    assert uncut_cell_tag != cut_cell_tag
    init_tag = min(min(outside_cell_tag, uncut_cell_tag), cut_cell_tag) - 1
    tdim = mesh.topology.dim
    num_cells = get_num_cells(mesh)
    cells = np.arange(0, num_cells)

    # Setup cell tags using values
    values = np.full(cells.shape, init_tag, dtype=np.intc)
    values[outside_cells] = outside_cell_tag
    values[uncut_cells] = uncut_cell_tag
    values[cut_cells] = cut_cell_tag
    mt = dolfinx.mesh.meshtags(mesh, tdim, cells, values)
    mt.name = "celltags"

    return mt


def get_facetags(mesh, cut_cells, outside_cells, ghost_penalty_tag=1):
    if ghost_penalty_tag == 0:
        init_tag = ghost_penalty_tag + 1
    else:
        init_tag = ghost_penalty_tag - 1
    tdim = mesh.topology.dim
    num_faces = get_num_faces(mesh)
    faces = np.arange(0, num_faces)

    # Find ghost penalty faces as all faces shared by a cut cell and
    # not an outside cell
    mesh.topology.create_connectivity(tdim - 1, tdim)
    face_2_cells = mesh.topology.connectivity(tdim - 1, tdim)
    gp_faces = []
    for f in faces:
        local_cells = face_2_cells.links(f)
        if len(local_cells) == 2:
            if (
                local_cells[0] in cut_cells and not local_cells[1] in outside_cells
            ) or (local_cells[1] in cut_cells and not local_cells[0] in outside_cells):
                gp_faces.append(f)

    # Setup face tags using values
    values = np.full(faces.shape, init_tag, dtype=np.intc)
    values[gp_faces] = ghost_penalty_tag
    mt = dolfinx.mesh.meshtags(mesh, tdim - 1, faces, values)
    mt.name = "facetags"

    return mt


def print_for_header(
    b_local,
    coeffs,
    constants,
    cell_coords,
    entity_local_index,
    quadrature_permutation,
    num_quadrature_points,
    qr_pts,
    qr_w,
    qr_n,
):
    def print_flat(x):
        print("{", end="")
        for xi in x:
            print(xi, end=",")
        print("};")

    print("printing function params:")
    print("double A[] = ", end="")
    print_flat(b_local)
    print("const double w[] = ", end="")
    print_flat(coeffs)
    print("const double c[] = ", end="")
    print_flat(constants)
    print("const double coordinate_dofs[] = ", end="")
    print_flat(cell_coords.flatten())
    print("const int entity_local_index[] = ", end="")
    print_flat(entity_local_index.flatten())
    print("const uint8_t quadrature_permutation[] = ", end="")
    print_flat(quadrature_permutation.flatten())
    print("const int num_quadrature_points = ", num_quadrature_points, ";")
    print("const double quadrature_points[] = ", end="")
    print_flat(qr_pts)
    print("const double quadrature_weights[] = ", end="")
    print_flat(qr_w)
    print("const double facet_normals[] = ", end="")
    print_flat(qr_n)
    print(
        "tabulate_tensor_integral_custom_otherwise(A,w,c,coordinate_dofs,entity_local_index,quadrature_permutation,num_quadrature_points,quadrature_points,quadrature_weights,facet_normals);"
    )


def volume(xmin, xmax, NN, uncut_cells, qr_w):
    flatten = lambda l: [item for sublist in l for item in sublist]
    gdim = len(NN)
    cellvol = np.prod((xmax - xmin)[0:gdim]) / np.prod(NN)
    cut_vol = sum(flatten(qr_w)) * cellvol
    uncut_vol = cellvol * len(uncut_cells)
    v = cut_vol + uncut_vol
    return v


def area(xmin, xmax, NN, qr_w_bdry):
    flatten = lambda l: [item for sublist in l for item in sublist]
    gdim = len(NN)
    cellvol = np.prod((xmax - xmin)[0:gdim]) / np.prod(NN)
    a = sum(flatten(qr_w_bdry)) * cellvol
    return a
