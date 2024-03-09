#ifndef CUSTOMQUAD_CIRCLE_HPP
#define CUSTOMQUAD_CIRCLE_HPP

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

void run(const std::vector<double>& LLx,
	 const std::vector<double>& LLy,
         const std::vector<double>& LLz,
	 const std::vector<double>& URx,
         const std::vector<double>& URy,
	 const std::vector<double>& URz,
         int degree, bool do_verbose, bool do_map, bool do_scale)
{
  std::cout << "algoim " << __FILE__ << std::endl;

  Sphere<2> phi;

  algoim_utils::run_template<Sphere<2>, 2>
    (phi,
     LLx, LLy, LLz, URx, URy, URz, degree,
     do_verbose, do_map, do_scale);
}

#endif
