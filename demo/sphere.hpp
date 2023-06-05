#include "algoim_quad.hpp"
#include <blitz/array.h>
#include <cassert>
#include <fstream>

struct Sphere
{
  const double xc = 0;
  const double yc = 0;
  const double zc = 0;
  const double R = 1;

  template <typename T>
  T operator()(const blitz::TinyVector<T, 3>& x) const
  {
    return (x(0) - xc) * (x(0) - xc) + (x(1) - yc) * (x(1) - yc)
           + (x(2) - zc) * (x(2) - zc) - R * R * R;
  }

  template <typename T>
  blitz::TinyVector<T, 3> grad(const blitz::TinyVector<T, 3>& x) const
  {
    return blitz::TinyVector<T, 3>(2.0 * (x(0) - xc), 2.0 * (x(1) - yc),
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
std::vector<std::vector<double>> xyz;
std::vector<std::vector<double>> xyz_bdry;

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

void check_inside(double x0, double y0, double z0, double x1, double y1,
                  double z1, const std::vector<double>& pts)
{
  for (std::size_t i = 0; i < pts.size() / 3; ++i)
  {
    const double x = pts[3 * i];
    const double y = pts[3 * i + 1];
    const double z = pts[3 * i + 2];
    assert(x >= x0 and x <= x1);
    assert(y >= y0 and y <= y1);
    assert(z >= z0 and z <= z1);
  }
}

template <class T>
void convert(const T& qr, const int gdim, std::vector<double>& pts,
             std::vector<double>& w)
{
  const std::size_t N = qr.nodes.size();
  pts.resize(3 * N);
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
              << "LLz " << LLz.size() << '\n'
              << "URx " << URx.size() << '\n'
              << "URy " << URy.size() << '\n'
              << "URz " << URz.size() << '\n'
              << "degree " << degree << '\n'
              << "verbose " << verbose << '\n'
              << "map " << map << '\n'
              << "scale " << scale << std::endl;
  }

  Sphere phi;
  const int gdim = 3;
  const int side = -1;
  const int dim_bulk = -1;
  const int dim_bdry = gdim;

  const std::size_t num_cells = LLx.size();
  assert(num_cells == LLy.size());
  assert(num_cells == LLz.size());
  assert(num_cells == URx.size());
  assert(num_cells == URy.size());
  assert(num_cells == URz.size());
  qr_pts.resize(num_cells);
  qr_w.resize(num_cells);
  qr_pts_bdry.resize(num_cells);
  qr_w_bdry.resize(num_cells);
  qr_n.resize(num_cells);
  xyz.resize(num_cells);
  xyz_bdry.resize(num_cells);

  for (std::size_t cell_no = 0; cell_no < num_cells; ++cell_no)
  {
    const blitz::TinyVector<double, 3> a
        = {LLx[cell_no], LLy[cell_no], LLz[cell_no]};
    const blitz::TinyVector<double, 3> b
        = {URx[cell_no], URy[cell_no], URz[cell_no]};

    if (verbose)
    {
      std::cout << cell_no << ' ' << a << ' ' << b << '\n';
    }

    const auto q_bdry = Algoim::quadGen<3>(
        phi, Algoim::BoundingBox<double, 3>(a, b), dim_bdry, side, degree);

    if (q_bdry.nodes.size())
    {
      // Cut cell
      cut_cells.push_back(cell_no);

      // Boundary qr
      convert(q_bdry, gdim, qr_pts_bdry[cell_no], qr_w_bdry[cell_no]);
      xyz_bdry[cell_no] = qr_pts_bdry[cell_no];

      check_inside(LLx[cell_no], LLy[cell_no], LLz[cell_no], URx[cell_no],
                   URy[cell_no], URz[cell_no], qr_pts_bdry[cell_no]);

      // Normal
      for (std::size_t k = 0; k < qr_pts_bdry[cell_no].size() / gdim; ++k)
      {
        const double x = qr_pts_bdry[cell_no][gdim * k];
        const double y = qr_pts_bdry[cell_no][gdim * k + 1];
        const double z = qr_pts_bdry[cell_no][gdim * k + 2];
        const auto grad = phi.grad(blitz::TinyVector<double, 3>(x, y, z));
        const double invnorm = 1. / Algoim::mag<double, 3>(grad);
        for (std::size_t d = 0; d < gdim; ++d)
          qr_n[cell_no].push_back(grad(d) * invnorm);
      }

      // Bulk
      const auto q_bulk = Algoim::quadGen<3>(
          phi, Algoim::BoundingBox<double, 3>(a, b), dim_bulk, side, degree);
      convert(q_bulk, gdim, qr_pts[cell_no], qr_w[cell_no]);
      xyz[cell_no] = qr_pts[cell_no];

      check_inside(LLx[cell_no], LLy[cell_no], LLz[cell_no], URx[cell_no],
                   URy[cell_no], URz[cell_no], qr_pts[cell_no]);

      if (scale)
      {
        // fenics scales with volume, need to compensate for this
        std::vector<double> dx
            = {URx[cell_no] - LLx[cell_no], URy[cell_no] - LLy[cell_no],
               URz[cell_no] - LLz[cell_no]};
        const double vol = std::accumulate(dx.begin(), dx.end(), 1.0,
                                           std::multiplies<double>());
        for (double& w : qr_w[cell_no])
          w /= vol;
        for (double& w : qr_w_bdry[cell_no])
          w /= vol;
      }

      if (map)
      {
        // Map to [0,1]^3
        // const std::vector<double> dx = {URx[cell_no] - LLx[cell_no],
        //   URy[cell_no] - LLy[cell_no],
        //   URz[cell_no] - LLz[cell_no]};
        // const std::vector<double> LLxyz = {LLx[cell_no], LLy[cell_no],
        // LLz[cell_no]};

        const double dx = URx[cell_no] - LLx[cell_no];
        const double dy = URy[cell_no] - LLy[cell_no];
        const double dz = URz[cell_no] - LLz[cell_no];
        auto map = [&](std::vector<double>& qr) {
          // for (std::size_t i = 0; i < qr.size(); i += gdim)
          //   for (std::size_t d = 0; d < gdim; ++d) {
          //     qr[i+d] = (qr[i+d] - LLxyz[d]) / dx[d];
          //     //assert(qr[i+d] >= 0.0 and qr[i+d] <= 1.0);
          //     // std::cout << "qr[i+d] "<<cell_no<<' '<<qr[i+d]<<'\n';
          //   }

          for (std::size_t i = 0; i < qr.size() / 3; ++i)
          {
            qr[3 * i] = (qr[3 * i] - LLx[cell_no]) / dx;
            qr[3 * i + 1] = (qr[3 * i + 1] - LLy[cell_no]) / dy;
            qr[3 * i + 2] = (qr[3 * i + 2] - LLz[cell_no]) / dz;
          }
        };

        map(qr_pts[cell_no]);
        check_inside(0, 0, 0, 1, 1, 1, qr_pts[cell_no]);
        map(qr_pts_bdry[cell_no]);
        check_inside(0, 0, 0, 1, 1, 1, qr_pts_bdry[cell_no]);
      }
    }
    else
    {
      // Either outside or inside
      const blitz::TinyVector<double, 3> midp = 0.5 * (a + b);
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
    auto print = [](const std::vector<std::vector<double>>& p,
                    const std::vector<std::vector<double>>& w,
                    const std::string tag, std::ostream& out = std::cout) {
      out << tag << '\n';
      for (std::size_t i = 0; i < w.size(); ++i)
      {
        // for (std::size_t j = 0; j < p[i].size(); j += 3)
        //   std::cout << p[i][j] << ' ' << p[i][j+1] << ' ' << p[i][j+2] << "
        //   " << w[i][j] << '\n';
        for (std::size_t j = 0; j < w[i].size(); ++j)
          out << p[i][3 * j] << ' ' << p[i][3 * j + 1] << ' ' << p[i][3 * j + 2]
              << "      " << w[i][j] << '\n';
      }
    };

    print(qr_pts, qr_w, "bulk [0,1]^3");
    print(qr_pts_bdry, qr_w_bdry, "bdry [0,1]^3");
    print(xyz, qr_w, "xyz");
    print(xyz_bdry, qr_w_bdry, "xyz_bdry");

    // static std::ofstream file;
    // if (!file.is_open())
    //     file.open("qr.txt");
    // print(qr_pts, qr_w, "bulk [0,1]^3", file);
    // print(qr_pts_bdry, qr_w_bdry, "bdry [0,1]^3", file);
    // print(xyz, qr_w, "xyz", file);
    // print(xyz_bdry, qr_w_bdry, "xyz bdry", file);
    // file.close();
  }

  std::cout << __FILE__ << " done" << std::endl;
}
