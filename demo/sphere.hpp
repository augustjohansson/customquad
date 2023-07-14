#ifndef CUSTOMQUAD_SPHERE_HPP
#define CUSTOMQUAD_SPHERE_HPP

#include "algoim_utils.hpp"

template<int gdim>
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

void run(const std::vector<double>& LLx,
	 const std::vector<double>& LLy,
         const std::vector<double>& LLz,
	 const std::vector<double>& URx,
         const std::vector<double>& URy,
	 const std::vector<double>& URz,
         int degree, bool do_verbose, bool do_map, bool do_scale)
{
  std::cout << "algoim " << __FILE__ << std::endl;

  Sphere<3> phi;
  
  algoim_utils::run_template<Sphere<3>, 3>
    (phi,
     LLx, LLy, LLz, URx, URy, URz, degree,
     do_verbose, do_map, do_scale,
     cut_cells, uncut_cells, outside_cells,
     qr_pts, qr_w,
     qr_pts_bdry, qr_w_bdry, qr_n,
     xyz, xyz_bdry);
}

#endif
