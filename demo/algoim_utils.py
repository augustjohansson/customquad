import numpy as np
import cppyy
import cppyy.ll
import dolfinx

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
                assert pmin[d] >= xmin[d], f"min fail {c}: {pmin[d]}, {xmin[d]}"
                assert pmax[d] <= xmax[d], f"max fail {c}: {pmax[d]}, {xmax[d]}"


cppyy.cppdef(
    """
    void* double_array(const std::vector<double>& v)
    {
      double* ptr = (double*)malloc(sizeof(double)*v.size());
      for (std::size_t i = 0; i < v.size(); ++i)
        ptr[i] = v[i];
      return ptr;
    }"""
)

cppyy.cppdef(
    """
    void* int_array(const std::vector<int>& v)
    {
      int* ptr = (int*)malloc(sizeof(int)*v.size());
      for (std::size_t i = 0; i < v.size(); ++i)
        ptr[i] = v[i];
      return ptr;
    }"""
)


def create_double_array(vec):
    array = cppyy.gbl.double_array(vec)
    v = np.frombuffer(array, dtype=np.float64, count=len(vec))
    return v


def create_int_array(vec):
    array = cppyy.gbl.int_array(vec)
    v = np.frombuffer(array, dtype=np.int32, count=len(vec)).tolist()
    return v


def create_list_of_arrays(v, cells=[]):
    if len(cells):
        return [create_double_array(v[c]) for c in cells]
    else:
        return [create_double_array(v_i) for v_i in v]


# Main function
def generate_qr(mesh, NN, degree, domain, opts=[]):
    """degree specifies the degree of the underlying one-dimensional
    Gaussian quadrature scheme and must satisfy 1 <= qo && qo <= 10.
    """
    cppyy.add_include_path("/usr/local/include/algoim/algoim")

    if domain == "circle":
        hppfile = "circle.hpp"
    elif domain == "sphere":
        hppfile = "sphere.hpp"
    else:
        raise RuntimeError("Unknown domain", domain)

    cppyy.include(hppfile)
    cppyy.include("algoim_utils.hpp")

    do_map = True
    do_scale = True

    if opts == []:
        do_verbose = False
    else:
        do_verbose = opts["verbose"]

    gdim = mesh.geometry.dim
    num_cells = np.prod(NN)
    assert mesh.topology.index_map(gdim).size_local == num_cells

    t = dolfinx.common.Timer()
    dofs = mesh.geometry.dofmap.array.reshape((num_cells, -1))
    cell_coords = mesh.geometry.x[dofs]
    bbxmin = np.min(cell_coords, axis=1)
    bbxmax = np.max(cell_coords, axis=1)
    LLx = bbxmin[:, 0]
    LLy = bbxmin[:, 1]
    URx = bbxmax[:, 0]
    URy = bbxmax[:, 1]
    if gdim == 3:
        LLz = bbxmin[:, 2]
        URz = bbxmax[:, 2]
    else:
        LLz = np.zeros(num_cells)
        URz = np.zeros(num_cells)
    print("Getting cell sizes for algoim took", t.elapsed()[0])

    t = dolfinx.common.Timer()
    cppyy.gbl.run(
        LLx.copy(),
        LLy.copy(),
        LLz.copy(),
        URx.copy(),
        URy.copy(),
        URz.copy(),
        degree,
        do_verbose,
        do_map,
        do_scale,
    )
    print("Algoim call took", t.elapsed()[0])

    t = dolfinx.common.Timer()

    cut_cells = create_int_array(cppyy.gbl.algoim_utils.cut_cells)
    uncut_cells = create_int_array(cppyy.gbl.algoim_utils.uncut_cells)
    outside_cells = create_int_array(cppyy.gbl.algoim_utils.outside_cells)

    qr_pts = create_list_of_arrays(cppyy.gbl.algoim_utils.qr_pts, cut_cells)
    qr_w = create_list_of_arrays(cppyy.gbl.algoim_utils.qr_w, cut_cells)
    qr_pts_bdry = create_list_of_arrays(cppyy.gbl.algoim_utils.qr_pts_bdry, cut_cells)
    qr_w_bdry = create_list_of_arrays(cppyy.gbl.algoim_utils.qr_w_bdry, cut_cells)
    qr_n = create_list_of_arrays(cppyy.gbl.algoim_utils.qr_n, cut_cells)

    xyz = create_list_of_arrays(cppyy.gbl.algoim_utils.xyz)
    xyz_bdry = create_list_of_arrays(cppyy.gbl.algoim_utils.xyz_bdry)

    print("Converting arrays took", t.elapsed()[0])

    # t = dolfinx.common.Timer()
    # check_status(
    #     mesh,
    #     cut_cells,
    #     uncut_cells,
    #     outside_cells,
    #     qr_pts,
    #     qr_w,
    #     qr_pts_bdry,
    #     qr_w_bdry,
    #     qr_n,
    #     xyz,
    #     xyz_bdry,
    # )
    # print("Checking status of qr took", t.elapsed()[0])

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
