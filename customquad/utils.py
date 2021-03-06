import dolfinx
import numba
import numpy
from petsc4py import PETSc


def get_num_entities(mesh, td):
    mesh.topology.create_connectivity_all()
    num_owned_entities = mesh.topology.index_map(td).size_local
    num_ghost_entities = mesh.topology.index_map(td).num_ghosts
    num_entities = num_owned_entities + num_ghost_entities
    return num_entities


def get_num_cells(mesh):
    tdim = mesh.topology.dim
    return get_num_entities(mesh, tdim)


def get_num_faces(mesh):
    tdim = mesh.topology.dim
    return get_num_entities(mesh, tdim-1)


def get_dofs(V):
    num_cells = get_num_cells(V.mesh)
    num_loc_dofs = V.dofmap.dof_layout.num_dofs \
        * V.dofmap.dof_layout.block_size()
    dofs = V.dofmap.list.array.reshape(num_cells,
                                       num_loc_dofs).astype(numpy.dtype(PETSc.IntType))
    return dofs, num_loc_dofs


def get_vertices(mesh):
    coords = mesh.geometry.x
    gdim = mesh.geometry.dim
    # num_vertices = mesh.topology.index_map(0).size_local
    celltype = mesh.topology.cell_type
    num_loc_vertices = dolfinx.cpp.mesh.cell_num_vertices(celltype)
    num_cells = get_num_cells(mesh)
    vertices = mesh.geometry.dofmap.array.reshape(num_cells, num_loc_vertices)
    return vertices, coords, gdim


def check_qr(qr_pts, qr_w, qr_n, cells):

    if isinstance(qr_pts, list):
        qr_pts = numba.typed.List(qr_pts)
    if isinstance(qr_w, list):
        qr_w = numba.typed.List(qr_w)
    if isinstance(cells, int):
        cells = numba.typed.List([cells])
    if isinstance(cells, list):
        cells = numba.typed.List(cells)

    assert len(qr_pts) == len(qr_w)

    # Find first non-empty
    k = 0
    while True:
        if qr_pts[k].any():
            gdim = float(len(qr_pts[k])) / float(len(qr_w[k]))
            assert gdim.is_integer()
            gdim = int(gdim)
            break
        else:
            k += 1

    # Cells are dense, but qr_pts, qr_w, qr_n not. qr_pts and qr_w
    # (not qr_n) may for convenience be of length 1.
    if len(qr_pts) == 1:
        assert isinstance(qr_pts[0], numpy.ndarray)
        qr_pts = replicate(qr_pts[0], cells)

    if len(qr_w) == 1:
        assert isinstance(qr_w[0], numpy.ndarray)
        qr_w = replicate(qr_w[0], cells)

    for cell in cells:
        assert len(qr_pts[cell]) == gdim*len(qr_w[cell])

    if qr_n:
        if isinstance(qr_n, list):
            qr_n = numba.typed.List(qr_n)
        if len(qr_n) != len(qr_pts):
            # FIXME Doesn't make sense to duplicate, but generated
            # code requires some data
            assert isinstance(qr_n[0], numpy.ndarray)
            qr_n = replicate(qr_n[0], cells)
        for cell in cells:
            assert len(qr_n[cell]) == len(qr_pts[cell])

    if not qr_n:
        # Let qr_n contain dummy data since it shouldn't be used in
        # the form, but the generated code calls
        # e.g. facet_normal[2*ip+1].
        qr_n = qr_pts  # Pointer

    return qr_pts, qr_w, qr_n, cells


def replicate(a, cells):
    b = numba.typed.List()
    N = max(cells)+1  # not necessarily num_cells, but sufficient
    for i in range(N):
        b.append(numpy.array([]))
    for cell in cells:
        b[cell] = a.copy()
    return b


def get_inactive_dofs(V, cut_cells, uncut_cells):
    dofs, _ = get_dofs(V)
    num_vertices = V.mesh.topology.index_map(0).size_local
    assert num_vertices == dofs.max() + 1  # P1 elements
    all_dofs = numpy.arange(0, num_vertices)
    for cells in [cut_cells, uncut_cells]:
        for cell in cells:
            cell_dofs = V.dofmap.cell_dofs(cell)
            all_dofs[cell_dofs] = -1
    inactive_dofs = numpy.arange(0, num_vertices, dtype='int32')[all_dofs > -1]

    # num_cells = get_num_cells(V)
    # for c in range(num_cells):
    #     cell_dofs = V.dofmap.cell_dofs(c)
    #     print(c, cell_dofs)

    # # histogram
    # mesh = V.mesh
    # mesh.topology.create_connectivity_all()
    # num_cells = get_num_cells(mesh)
    # hist = numpy.zeros(num_vertices)
    # #for c in range(num_cells):
    # for c in cut_cells:
    #     cell_dofs = V.dofmap.cell_dofs(c)
    #     for dof in cell_dofs:
    #         hist[dof] += 1
    # x = mesh.geometry.x
    # for i in range(num_vertices):
    #     if hist[i] == 0:
    #         print(x[i][0],x[i][1])
    # import ipdb; ipdb.set_trace()

    return inactive_dofs


def lock_inactive_dofs(inactive_dofs, A):

    # dump("before.txt", A)

    # import ipdb; ipdb.set_trace()

    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    A.zeroRows(inactive_dofs, 1)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)

    # dump("after.txt", A)

    # check diagonal
    ad = A.getDiagonal()
    if (ad.array == 0).any():
        zeros = numpy.where(ad.array == 0)
        print("zero", zeros[0])
        for i in zeros[0]:
            A.setValue(i, i, 1.0)
        A.assemble()
        import ipdb; ipdb.set_trace()

    # import ipdb; ipdb.set_trace()
    # zeroRows(self, rows, diag=1, Vec x=None, Vec b=None)
    return A


@numba.njit
def printer(s, a):
    print(s, a)


def dump(filename, A):
    import petsc4py
    if isinstance(A, petsc4py.PETSc.Mat):
        assert A.assembled
        f = open(filename, 'w')
        for r in range(A.size[0]):
            cols, vals = A.getRow(r)
            for i in range(len(cols)):
                s = str(r+1) + " " + str(cols[i]+1) + " " + str(vals[i]) + "\n"
                f.write(s)
        f.close()
        print(f"A=load('{filename}');A=sparse(A(:,1),A(:,2),A(:,3)); condest(A), spy(A)")
    else:
        f = open(filename, 'w')
        numpy.savetxt(f, A.array)
        f.close()


def get_celltags(mesh, cut_cells, uncut_cells, outside_cells,
                 outside_cell_tag=0, uncut_cell_tag=1, cut_cell_tag=2):
    assert outside_cell_tag != uncut_cell_tag
    assert outside_cell_tag != cut_cell_tag
    assert uncut_cell_tag != cut_cell_tag
    init_tag = min(min(outside_cell_tag, uncut_cell_tag), cut_cell_tag) - 1
    tdim = mesh.topology.dim
    # cell_map = mesh.topology.index_map(tdim)
    num_cells = get_num_cells(mesh)
    cells = numpy.arange(0, num_cells)

    # Setup cell tags using values
    values = numpy.full(cells.shape, init_tag, dtype=numpy.intc)
    values[outside_cells] = outside_cell_tag
    values[uncut_cells] = uncut_cell_tag
    values[cut_cells] = cut_cell_tag
    mt = dolfinx.mesh.MeshTags(mesh, tdim, cells, values)

    return mt


def get_facetags(mesh, cut_cells, outside_cells,
                 ghost_penalty_tag=1):
    if ghost_penalty_tag == 0:
        init_tag = ghost_penalty_tag+1
    else:
        init_tag = ghost_penalty_tag-1
    tdim = mesh.topology.dim
    # face_map = mesh.topology.index_map(tdim-1)
    num_faces = get_num_faces(mesh)
    faces = numpy.arange(0, num_faces)

    # Find ghost penalty faces as all faces shared by a cut cell and
    # not an outside cell
    face_2_cells = mesh.topology.connectivity(tdim-1, tdim)
    gp_faces = []
    for f in faces:
        local_cells = face_2_cells.links(f)
        if len(local_cells) == 2:
            if (local_cells[0] in cut_cells and not local_cells[1] in outside_cells) or \
               (local_cells[1] in cut_cells and not local_cells[0] in outside_cells):
               gp_faces.append(f)

    # Setup face tags using values
    values = numpy.full(faces.shape, init_tag, dtype=numpy.intc)
    values[gp_faces] = ghost_penalty_tag
    mt = dolfinx.mesh.MeshTags(mesh, tdim-1, faces, values)

    return mt

def print_for_header(b_local, coeffs, constants, cell_coords,
                     num_quadrature_points, qr_pts, qr_w, qr_n):
    def print_flat(x):
        print("{", end="")
        for xi in x:
            print(xi, end=",")
        print("};")
    print("double A[] = ", end=""); print_flat(b_local);
    print("const double w[] = ", end=""); print_flat(coeffs);
    print("const double c[] = ", end=""); print_flat(constants)
    print("const double coordinate_dofs[] = ", end=""); print_flat(cell_coords.flatten())
    print("const int num_quadrature_points = ", num_quadrature_points, ";")
    print("const double quadrature_points[] = ", end=""); print_flat(qr_pts)
    print("const double quadrature_weights[] = ", end=""); print_flat(qr_w)
    print("const double facet_normals[] = ", end=""); print_flat(qr_n)
    print("tabulate_tensor_integral_custom_otherwise(A,w,c,coordinate_dofs,num_quadrature_points,quadrature_points,quadrature_weights,facet_normals);")


def volume(xmin, xmax, NN, uncut_cells, qr_w):
    flatten = lambda l: [item for sublist in l for item in sublist]
    gdim = len(NN)
    cellvol = numpy.prod((xmax-xmin)[0:gdim])/numpy.prod(NN)
    cut_vol = sum(flatten(qr_w))*cellvol
    uncut_vol = cellvol*len(uncut_cells)
    volume = cut_vol + uncut_vol
    return volume


def area(xmin, xmax, NN, qr_w_bdry):
    flatten = lambda l: [item for sublist in l for item in sublist]
    gdim = len(NN)
    cellvol = numpy.prod((xmax-xmin)[0:gdim])/numpy.prod(NN)
    area = sum(flatten(qr_w_bdry))*cellvol
    return area
