#include <cassert>
#include <fstream>
#include "quadrature_general.hpp"

template <int gdim>
struct Sphere
{
  const double xc = 0;
  const double yc = 0;
  const double zc = 0;
  const double R = 1;

  template<typename T>
  T operator()(const algoim::uvector<T, gdim>& x) const
  {
    if constexpr (gdim == 2)
      return (x(0) - xc) * (x(0) - xc) + (x(1) - yc) * (x(1) - yc) - R * R;
    else
      return (x(0) - xc) * (x(0) - xc) \
	+ (x(1) - yc) * (x(1) - yc) \
	+ (x(2) - zc) * (x(2) - zc) - R * R;
  }

  template<typename T>
  algoim::uvector<T, gdim> grad(const algoim::uvector<T, gdim>& x) const
  {
    if constexpr (gdim == 2)
      return algoim::uvector<T, gdim>(2.0 * (x(0) - xc), 2.0 * (x(1) - yc));
    else
      return algoim::uvector<T, gdim>(2.0 * (x(0) - xc),
				      2.0 * (x(1) - yc),
				      2.0 * (x(2) - zc));
  }
};

std::vector<int> cut_cells;
std::vector<int> uncut_cells;
std::vector<int> outside_cells;
std::vector<std::vector<double>> qr_pts;
std::vector<std::vector<double>> qr_w;
std::vector<std::vector<double>> qr_pts_bdry;
std::vector<std::vector<double>> qr_w_bdry;
std::vector<std::vector<double>> qr_n;
std::vector<std::vector<double>> xyz, xyz_bdry;

std::vector<int> get_cut_cells() { return cut_cells; }
std::vector<int> get_uncut_cells() { return uncut_cells; }
std::vector<int> get_outside_cells() { return outside_cells; }
std::vector<std::vector<double>> get_qr_pts() { return qr_pts; }
std::vector<std::vector<double>> get_qr_w() { return qr_w; }
std::vector<std::vector<double>> get_qr_pts_bdry() { return qr_pts_bdry; }
std::vector<std::vector<double>> get_qr_w_bdry() { return qr_w_bdry; }
std::vector<std::vector<double>> get_qr_n() { return qr_n; }
std::vector<std::vector<double>> get_xyz() { return xyz; }
std::vector<std::vector<double>> get_xyz_bdry() { return xyz_bdry; }

template <class T>
void convert(const T& qr, const int gdim, std::vector<double>& pts,
             std::vector<double>& w)
{
  const std::size_t N = qr.nodes.size();
  pts.resize(gdim * N);
  w.resize(N);
  for (std::size_t i = 0; i < N; ++i)
  {
    for (int d = 0; d < gdim; ++d)
      pts[gdim * i + d] = qr.nodes[i].x(d);
    w[i] = qr.nodes[i].w;
  }
}

void run(const std::vector<double>& LLx, const std::vector<double>& LLy,
         const std::vector<double>& LLz, const std::vector<double>& URx,
         const std::vector<double>& URy, const std::vector<double>& URz,
         int degree, bool verbose, bool map, bool scale)
{
  std::cout << "algoim " << __FILE__ << std::endl;

  if (verbose)
  {
    std::cout << "LLx " << LLx.size() << '\n'
              << "LLy " << LLy.size() << '\n'
              << "URx " << URx.size() << '\n'
              << "URy " << URy.size() << '\n'
              << "degree " << degree << '\n'
              << "verbose " << verbose << '\n'
              << "map " << map << '\n'
              << "scale " << scale << std::endl;
  }

  const int gdim = 2;
  Sphere<2> phi;

  const int side = -1;
  const int dim_bulk = -1;
  const int dim_bdry = gdim;

  const std::size_t num_cells = LLx.size();
  assert(num_cells == LLy.size());
  assert(num_cells == URx.size());
  assert(num_cells == URy.size());
  qr_pts.resize(num_cells);
  qr_w.resize(num_cells);
  qr_pts_bdry.resize(num_cells);
  qr_w_bdry.resize(num_cells);
  qr_n.resize(num_cells);
  xyz.resize(num_cells);
  xyz_bdry.resize(num_cells);

  for (std::size_t cell_no = 0; cell_no < num_cells; ++cell_no)
  {
    const algoim::uvector<double, 2> min(LLx[cell_no], LLy[cell_no]);
    const algoim::uvector<double, 2> max(URx[cell_no], URy[cell_no]);

    const auto q_bdry = algoim::quadGen<2>(
        phi, algoim::HyperRectangle<double, 2>(min, max), dim_bdry, side, degree);
    if (q_bdry.nodes.size())
    {
      // Cut cell
      cut_cells.push_back(cell_no);

      // Boundary qr
      convert(q_bdry, gdim, qr_pts_bdry[cell_no], qr_w_bdry[cell_no]);
      xyz_bdry[cell_no] = qr_pts_bdry[cell_no];

      // Normal
      for (std::size_t k = 0; k < qr_pts_bdry[cell_no].size() / gdim; ++k)
      {
        const double x = qr_pts_bdry[cell_no][gdim * k];
        const double y = qr_pts_bdry[cell_no][gdim * k + 1];
        const auto grad = phi.grad(algoim::uvector<double, 2>(x, y));
        for (std::size_t d = 0; d < gdim; ++d)
          qr_n[cell_no].push_back(grad(d));
      }

      // Bulk
      const auto q_bulk = algoim::quadGen<2>(
          phi, algoim::HyperRectangle<double, 2>(min, max), dim_bulk, side, degree);
      convert(q_bulk, gdim, qr_pts[cell_no], qr_w[cell_no]);
      xyz[cell_no] = qr_pts[cell_no];

      if (scale)
      {
        // fenics scales with volume, need to compensate for this
        const double dx = URx[cell_no] - LLx[cell_no];
        const double dy = URy[cell_no] - LLy[cell_no];
        const double vol = dx * dy;
        for (double& w : qr_w[cell_no])
          w /= vol;
        for (double& w : qr_w_bdry[cell_no])
          w /= vol;
      }

      if (map)
      {
        // Map to [0,1]^2
        const double dx = URx[cell_no] - LLx[cell_no];
        const double dy = URy[cell_no] - LLy[cell_no];
        auto map = [&](std::vector<double>& qr) {
          for (std::size_t i = 0; i < qr.size(); i += 2)
          {
            qr[i] = (qr[i] - LLx[cell_no]) / dx;
            qr[i + 1] = (qr[i + 1] - LLy[cell_no]) / dy;
          }
        };

        map(qr_pts[cell_no]);
        map(qr_pts_bdry[cell_no]);
      }
    }
    else
    {
      // Either outside or inside
      const algoim::uvector<double, 2> midp = 0.5 * (min + max);
      if (phi(midp) < 0)
        uncut_cells.push_back(cell_no);
      else
        outside_cells.push_back(cell_no);
    }
  }

  for (std::size_t i = 0; i < num_cells; ++i)
  {
    if (qr_w_bdry[i].size())
    {
      assert(qr_pts[i].size());
      assert(qr_w[i].size());
      assert(qr_pts[i].size() == gdim * qr_w[i].size());

      assert(qr_pts_bdry[i].size());
      assert(qr_w_bdry[i].size());
      assert(qr_n[i].size());
      assert(qr_pts_bdry[i].size() == gdim * qr_w_bdry[i].size());
      assert(qr_pts_bdry[i].size() == qr_n[i].size());

      assert(xyz[i].size());
      assert(xyz_bdry[i].size());
    }
    else
    {
      assert(qr_pts[i].size() == 0);
      assert(qr_w[i].size() == 0);
      assert(qr_pts_bdry[i].size() == 0);
      assert(qr_w_bdry[i].size() == 0);
      assert(qr_n[i].size() == 0);
      assert(xyz[i].size() == 0);
      assert(xyz_bdry[i].size() == 0);
    }
  }

  if (verbose)
  {
    auto print = [](const std::vector<std::vector<double>>& p) {
      for (std::size_t i = 0; i < p.size(); ++i)
        for (std::size_t j = 0; j < p[i].size(); j += 2)
          std::cout << p[i][j] << ' ' << p[i][j + 1] << '\n';
    };

    std::cout << "bulk [0,1]^2\n";
    print(qr_pts);
    std::cout << "bdry [0,1]^2\n";
    print(qr_pts_bdry);
    std::cout << "xyz\n";
    print(xyz);
    std::cout << "xyz bdry\n";
    print(xyz_bdry);
  }

  std::cout << __FILE__ << " done" << std::endl;
}
