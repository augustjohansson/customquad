// Examples to demonstrate Algoim's methods for computing high-order accurate quadrature schemes
// for implicitly defined domains in hyperrectangles. The file contains a single main() routine;
// compile it as you would for any .cpp file with a main() entry point.

#include <fstream>
#include "algoim_quad.hpp"
#include <cassert>
#include <blitz/array.h>
#include <cmath>
#include <climits>

template<int N>
struct Square
{
  int sgn(double val) const
  {
    return (double(0) < val) - (val < double(0));
  }

  template<typename T>
  int sgn(T val) const
  {
    return val.sign();
  }
  
  template<typename T>
  T operator() (const blitz::TinyVector<T,N>& x) const
  {
    const T dx = x(0)-0.5, dy = x(1)-0.5;
    return dx*sgn(dx) + dy*sgn(dy);
  }

  double operator() (const blitz::TinyVector<double,N>& x) const
  {
    const double dx = x(0)-0.5, dy = x(1)-0.5;
    std::cout << dx<<' '<<dy << std::endl;
    return dx*sgn(dx) + dy*sgn(dy);
  }


  template<typename T>
  blitz::TinyVector<T,N> grad(const blitz::TinyVector<T,N>& x) const
  {
    const T dx = x(0)-0.5, dy = x(1)-0.5;
    return blitz::TinyVector<T,N>(T(sgn(dx)), T(sgn(dy)));
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

std::vector<int> get_cut_cells() { return cut_cells; }
std::vector<int> get_uncut_cells() { return uncut_cells; }
std::vector<int> get_outside_cells() { return outside_cells; }
std::vector<std::vector<double>> get_qr_pts() { return qr_pts; }
std::vector<std::vector<double>> get_qr_w() { return qr_w; }
std::vector<std::vector<double>> get_qr_pts_bdry() { return qr_pts_bdry; }
std::vector<std::vector<double>> get_qr_w_bdry() { return qr_w_bdry; }
std::vector<std::vector<double>> get_qr_n() { return qr_n; }

void run(double xmin, double xmax, double ymin, double ymax, int Nx, int Ny, int degree,
         bool map=true,
         bool scale=true)
{
  const int gdim = 2;
  Square<2> phi;

  const double dx = (xmax-xmin) / Nx;
  const double dy = (ymax-ymin) / Ny;
  const int side = -1;
  const int dim_bulk = -1;
  const int dim_bdry = gdim;
  const int num_cells = Nx*Ny;
  qr_pts.resize(num_cells);
  qr_w.resize(num_cells);
  qr_pts_bdry.resize(num_cells);
  qr_w_bdry.resize(num_cells);
  qr_n.resize(num_cells);

  for (int i = 0, cell_no = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j, ++cell_no) {
      const blitz::TinyVector<double,2> a = {xmin + i*dx, ymin + j*dy};
      const blitz::TinyVector<double,2> b = {xmin + i*dx + dx, ymin + j*dy + dy};

      auto q_bdry = Algoim::quadGen<2>(phi, Algoim::BoundingBox<double,2>(a, b), dim_bdry, side, degree);
      std::vector<double> weights_bdry = q_bdry.weights();

      std::cout << cell_no<<' '<<weights_bdry.size() << ' '<<phi(0.5*(a+b))<<std::endl;
      
      if (weights_bdry.size())
      {
        cut_cells.push_back(cell_no);

        qr_pts_bdry[cell_no] = q_bdry.points();
        qr_w_bdry[cell_no] = weights_bdry;

        // Normal
        for (std::size_t k = 0; k < qr_pts_bdry[cell_no].size()/gdim; ++k)
        {
          const double x = qr_pts_bdry[cell_no][gdim*k];
          const double y = qr_pts_bdry[cell_no][gdim*k+1];
          auto grad = phi.grad(blitz::TinyVector<double,2>(x, y));
          for (std::size_t d = 0; d < gdim; ++d)
            qr_n[cell_no].push_back(grad(d));
        }

        // Bulk
        auto q_bulk = Algoim::quadGen<2>(phi, Algoim::BoundingBox<double,2>(a, b), dim_bulk, side, degree);
        qr_pts[cell_no] = q_bulk.points();
        qr_w[cell_no] = q_bulk.weights();
      }
      else
      {
        // Either outside or inside
        blitz::TinyVector<double,2> midp = 0.5*(a + b);
        double phival = phi(midp);
        if (phival < 0)
          uncut_cells.push_back(cell_no);
        else
          outside_cells.push_back(cell_no);
      }

      if (map)
      {
        // Map to [0,1]^2
        for (std::size_t i = 0; i < qr_pts[cell_no].size()/2; ++i)
        {
          qr_pts[cell_no][2*i] = (qr_pts[cell_no][2*i]-a(0))/dx;
          qr_pts[cell_no][2*i+1] = (qr_pts[cell_no][2*i+1]-a(1))/dy;
        }
      }

      if (scale)
      {
        // fenics scales with volume, need to compensate for this
        const double vol = dx*dy;
        for (double& w: qr_w[cell_no])
          w /= vol;
        for (double& w: qr_w_bdry[cell_no])
          w /= vol;
      }
    }
  }

  std::cout << qr_pts.size() << ' ' << qr_w.size() << ' ' << qr_pts_bdry.size() << ' '<< qr_w_bdry.size() << ' ' << qr_n.size() << std::endl;



  double w = 0.;
  for (std::size_t i = 0; i < qr_w.size(); ++i)
    w += std::accumulate(qr_w[i].begin(), qr_w[i].end(), 0.0);
  std::cout << __FILE__<< " w "<< w << std::endl;
  double w_bdry = 0.;
  for (std::size_t i = 0; i < qr_w_bdry.size(); ++i)
    w_bdry += std::accumulate(qr_w_bdry[i].begin(), qr_w_bdry[i].end(), 0.0);
  std::cout << __FILE__<<" w bdry "<< w_bdry << std::endl;

  // check
  for (int c = 0; c < num_cells; ++c)
  {
    assert(qr_pts_bdry[c].size() == qr_n[c].size());
    assert(gdim*qr_w_bdry[c].size() == qr_pts_bdry[c].size());
    assert(gdim*qr_w[c].size() == qr_pts[c].size());
  }

}
