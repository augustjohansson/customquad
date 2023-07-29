import numpy as np
import cppyy
import cppyy.ll

cppyy.ll.set_signals_as_exception(True)


def check_status(
    mesh,
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
):
    for c in cut_cells:
        assert len(qr_w[c]) > 0
        assert len(qr_w_bdry[c]) > 0
        assert len(qr_pts[c]) > 0
        assert len(qr_pts_bdry[c]) > 0
        assert len(xyz[c]) > 0
        assert len(xyz_bdry[c]) > 0

    for cells in [uncut_cells, outside_cells]:
        for c in cells:
            assert len(qr_w[c]) == 0
            assert len(qr_w_bdry[c]) == 0
            assert len(qr_pts[c]) == 0
            assert len(qr_pts_bdry[c]) == 0
            assert len(xyz[c]) == 0, print(c, len(xyz[c]))
            assert len(xyz_bdry[c]) == 0

    gdim = mesh.geometry.dim
    x = mesh.geometry.x
    xmin = np.min(x, axis=0)
    xmax = np.max(x, axis=0)
    for c in cut_cells:
        for qr in [qr_pts[c], qr_pts_bdry[c]]:
            n = len(qr) // gdim
            p = np.reshape(qr, (n, gdim))
            pmin = np.min(p, axis=0)
            pmax = np.max(p, axis=0)
            for d in range(gdim):
                assert pmin[d] >= xmin[d], print("min fail", c, pmin[d], xmin[d])
                assert pmax[d] <= xmax[d], print("max fail", c, pmax[d], xmax[d])


cppyy.cppdef(
    """
    void* double_array(const std::vector<double>& v) {
      double* pf = (double*)malloc(sizeof(double)*v.size());
      for (std::size_t i = 0; i < v.size(); ++i) pf[i] = v[i];
      return pf;
    }"""
)

cppyy.cppdef(
    """
    void* int_array(const std::vector<int>& v) {
      int* pf = (int*)malloc(sizeof(int)*v.size());
      for (std::size_t i = 0; i < v.size(); ++i) pf[i] = v[i];
      return pf;
    }"""
)


def create_double_array(vec):
    array = cppyy.gbl.double_array(vec)
    v = np.frombuffer(array, dtype=np.float64, count=len(vec))
    return v


def create_int_array(vec):
    array = cppyy.gbl.int_array(vec)
    v = np.frombuffer(array, dtype=np.int32, count=len(vec))
    return v


def create_list_of_arrays(v):
    w = [None] * len(v)
    for i in range(len(v)):
        w[i] = create_double_array(v[i])
    return w


# Main function
def generate_qr(mesh, NN, degree, domain, opts=[]):
    """degree specifies the degree of the underlying one-dimensional
    Gaussian quadrature scheme and must satisfy 1 <= qo && qo <= 10.
    """
    cppyy.add_include_path("/usr/include/algoim/algoim")
    if domain == "square":
        hppfile = "square.hpp"
    elif domain == "circle":
        hppfile = "circle.hpp"
    elif domain == "sphere":
        hppfile = "sphere.hpp"
    else:
        RuntimeError("unknown domain", domain)

    cppyy.include(hppfile)
    do_map = True
    do_scale = True

    if opts == []:
        do_verbose = False
    else:
        do_verbose = opts["verbose"]

    gdim = mesh.geometry.dim
    num_cells = np.prod(NN)
    assert mesh.topology.index_map(gdim).size_local == num_cells
    LLx = np.zeros(num_cells)
    LLy = np.zeros(num_cells)
    LLz = np.zeros(num_cells)
    URx = np.zeros(num_cells)
    URy = np.zeros(num_cells)
    URz = np.zeros(num_cells)

    num_loc_vertices = 2**gdim
    dofmap = mesh.geometry.dofmap
    x = mesh.geometry.x
    cell_coords = np.zeros((num_loc_vertices, gdim))
    for cell in range(num_cells):
        dofs = dofmap.links(cell)
        for j in range(num_loc_vertices):
            cell_coords[j] = x[dofs[j], 0:gdim]
        LLx[cell] = min(cell_coords[:, 0])
        LLy[cell] = min(cell_coords[:, 1])
        URx[cell] = max(cell_coords[:, 0])
        URy[cell] = max(cell_coords[:, 1])
        if gdim == 3:
            LLz[cell] = min(cell_coords[:, 2])
            URz[cell] = max(cell_coords[:, 2])

    cppyy.gbl.run(LLx, LLy, LLz, URx, URy, URz, degree, do_verbose, do_map, do_scale)

    qr_pts = create_list_of_arrays(cppyy.gbl.get_qr_pts())
    qr_w = create_list_of_arrays(cppyy.gbl.get_qr_w())
    qr_pts_bdry = create_list_of_arrays(cppyy.gbl.get_qr_pts_bdry())
    qr_w_bdry = create_list_of_arrays(cppyy.gbl.get_qr_w_bdry())
    qr_n = create_list_of_arrays(cppyy.gbl.get_qr_n())
    xyz = create_list_of_arrays(cppyy.gbl.get_xyz())
    xyz_bdry = create_list_of_arrays(cppyy.gbl.get_xyz_bdry())

    cut_cells = create_int_array(cppyy.gbl.get_cut_cells())
    uncut_cells = create_int_array(cppyy.gbl.get_uncut_cells())
    outside_cells = create_int_array(cppyy.gbl.get_outside_cells())

    check_status(
        mesh,
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
    )

    return (
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
    )
