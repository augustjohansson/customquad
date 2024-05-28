import dolfinx
import numpy as np
from petsc4py import PETSc
import customquad as cq


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

    but if type(V) = <class 'dolfinx.fem.function.FunctionSpace'>`
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

    init_tag = min(outside_cell_tag, uncut_cell_tag, cut_cell_tag) - 1
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

    # Find ghost penalty faces as all faces shared by a cut cell and
    # not an outside cell
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f2c = mesh.topology.connectivity(tdim - 1, tdim)
    diffs = np.diff(f2c.offsets)
    faces = np.where(diffs == 2)[0]
    cells = np.array([f2c.links(f) for f in faces])
    left_cells = cells[:, 0]
    right_cells = cells[:, 1]
    num_cells = get_num_cells(mesh)
    out = np.full(num_cells, False)
    cut = np.full(num_cells, False)
    out[outside_cells] = True
    cut[cut_cells] = True
    gp_faces = []

    for f, left, right in zip(faces, left_cells, right_cells):
        if (cut[left] and not out[right]) or (cut[right] and not out[left]):
            gp_faces.append(f)

    # Setup face tags using values
    num_faces = get_num_faces(mesh)
    faces = np.arange(0, num_faces)
    values = np.full(faces.shape, init_tag, dtype=np.intc)
    values[gp_faces] = ghost_penalty_tag
    mt = dolfinx.mesh.meshtags(mesh, tdim - 1, faces, values)
    mt.name = "facetags"

    return mt


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def volume(xmin, xmax, NN, uncut_cells, qr_w):
    gdim = len(NN)
    cellvol = np.prod((xmax - xmin)[0:gdim]) / np.prod(NN)
    cut_vol = sum(flatten(qr_w)) * cellvol
    uncut_vol = cellvol * len(uncut_cells)
    v = cut_vol + uncut_vol
    return v


def area(xmin, xmax, NN, qr_w_bdry):
    gdim = len(NN)
    cellvol = np.prod((xmax - xmin)[0:gdim]) / np.prod(NN)
    a = sum(flatten(qr_w_bdry)) * cellvol
    return a


def assemble_cut_uncut(integrand, dx_cut, qr_bulk, dx_uncut, uncut_cell_tag):

    # Assemble over cut part
    form = dolfinx.fem.form(integrand * dx_cut)
    m_cut = cq.assemble_scalar(form, qr_bulk)

    # Assemble over interior
    form = dolfinx.fem.form(integrand * dx_uncut(uncut_cell_tag))
    m_uncut = dolfinx.fem.assemble_scalar(form)

    return m_cut + m_uncut


def writeXDMF(filename, mesh, data):
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
            raise RuntimeError("Unsupported data when writing file", filename)


def subdomain(qr_data, num_cells):
    # In the case of integration over subdomains,
    # eg. ds_cut(cut_cell_tag), the coeffs are already defined over
    # the subdomain with id=cut_cell_tag. Hence we need to renumber
    # the cells in the qr_data to match this. Note that the provided
    # quadrature rule must match the number of cells in this
    # subdomain.

    idx = np.arange(num_cells)
    qr_data2 = [(idx,) + qr[1:] for qr in qr_data]

    return qr_data2
