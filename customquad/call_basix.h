// Cell types (cell.h)
// enum class type
// {
//   point = 0,
//   interval = 1,
//   triangle = 2,
//   tetrahedron = 3,
//   quadrilateral = 4,
//   hexahedron = 5,
//   prism = 6,
//   pyramid = 7
// };
// /// Available element families (element_families.h)
// enum class family
// {
//   custom = 0,
//   P = 1,
//   RT = 2,
//   N1E = 3,
//   BDM = 4,
//   N2E = 5,
//   CR = 6,
//   Regge = 7,
//   DPC = 8,
//   bubble = 9,
//   serendipity = 10,
//   HHJ = 11,
//   Hermite = 12
// };
// /// Variants of a Lagrange space that can be created
// enum class lagrange_variant
// {
//   unset = -1,
//   equispaced = 0,
//   gll_warped = 1,
//   gll_isaac = 2,
//   gll_centroid = 3,
//   chebyshev_warped = 4,
//   chebyshev_isaac = 5,
//   chebyshev_centroid = 6,
//   gl_warped = 7,
//   gl_isaac = 8,
//   gl_centroid = 9,
//   legendre = 10,
//   bernstein = 11,
//   vtk = 20,
// };

#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <iostream>
#include <sstream>
#include <fstream>

namespace stdex = std::experimental;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;

void call_basix(double***** FE,
                int num_quadrature_points,
                const double* quadrature_points,
                int basix_derivative,
                int family,
                int cell_type,
                int degree,
                int /* lattice_type */,
                int gdim)
{

  
  // Create the lagrange element
  basix::FiniteElement lagrange = basix::create_element(basix::element::family(family),
							basix::cell::type(cell_type),
							degree,
							basix::element::lagrange_variant::unset);

  // Number of derivatives to obtain (0 or first order for now)
  // FIXME Handle derivatives of higher order.
  const int nd = basix_derivative == 0 ? 0 : 1;

  // Compute basis and derivatives. The shape is (derivative, point,
  // basis fn index, value index).
  auto [tab_data, shape] = lagrange.tabulate(nd, std::vector<double>(quadrature_points, quadrature_points+gdim*num_quadrature_points), {num_quadrature_points, gdim});

  // Convenient format
  cmdspan4_t tab(tab_data.data(), shape);
  const int num_basis_functions = tab.extent(2);

  // Check size
  assert(tab.extent(1) == num_quadrature_points);

  // Allocate
  *FE = new double***[1];
  for (int i = 0; i < 1; ++i)    {
    (*FE)[i] = new double**[1];
    for (int j = 0; j < 1; ++j)	{
      (*FE)[i][j] = new double*[num_quadrature_points];
      for (int k = 0; k < num_quadrature_points; ++k)
	(*FE)[i][j][k] = new double[num_basis_functions];
    }
  }

  // Copy
  for (int i = 0; i < num_quadrature_points; ++i)
    for (int j = 0; j < num_basis_functions; ++j) 
      (*FE)[0][0][i][j] = tab(basix_derivative, i, j, 0);

  // Debug output
  std::ofstream f;
  std::stringstream ss;
  ss << "/tmp/call_basix" << reinterpret_cast<void*>(FE) << ".txt";
  f.open(ss.str());
  f << "basix_derivative=" << basix_derivative << " family " << family << " cell_type " << cell_type << " degree " << degree << " gdim " << gdim << "\n";
  f << "tab table copied to FE[0][0][i][j]:\n";
  for (int i = 0; i < num_quadrature_points; ++i)
    for (int j = 0; j < num_basis_functions; ++j) 
      f << i << ' ' << j << ' ' << (*FE)[0][0][i][j] << '\n';
  f << "I.e. the i j matrix looks like\n";
  for (int i = 0; i < num_quadrature_points; ++i) {
    for (int j = 0; j < num_basis_functions; ++j) 
      f << (*FE)[0][0][i][j] << ' ';
    f << '\n';
  }
  f << "quadrature points:\n";
  for (int i = 0; i < num_quadrature_points; ++i)
    for (int d = 0; d < gdim; ++d)
      f << quadrature_points[gdim*i+d] << ' ';
  f << '\n';
  f << "tab shape: ";
  for (auto s: shape)
    f << s <<' ';
  f << '\n';

  f << "tab:\n";
  for (std::size_t i = 0; i != tab.extent(0); ++i){
    f << "i "<<i << '\n';
    for (std::size_t j = 0; j != tab.extent(1); ++j){
      for (std::size_t k = 0; k != tab.extent(2); ++k){
	for (std::size_t l = 0; l != tab.extent(3); ++l)
	  f << tab(i,j,k,l)<<' ';
	f<< '\n';
      }
      f << '\n';
    }
    f << '\n';
  }
  
  f.close();

}
