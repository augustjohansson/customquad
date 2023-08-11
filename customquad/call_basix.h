// For all cell types, see basix cell.h
// For all element families, see element_families.h
// For all variants of Lagrange spaces, see element_families.h

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
  std::vector<double> qr_pts(quadrature_points, quadrature_points+gdim*num_quadrature_points);
  auto [tab_data, shape] = lagrange.tabulate(nd, qr_pts, {num_quadrature_points, gdim});

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

  // Permutation vector
  std::vector<int> perm;
  if (cell_type == 4) {
    // Quad
    if (degree == 1)
      perm = {0, 2, 1, 3};
    else if (degree == 2)
      perm = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  }
  else if (cell_type == 5) {
    // Hex
    if (degree == 1)
      perm = {0,1,2,3,4,5,6,7};
    else if (degree == 2)
      perm = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
  }
  
  assert(perm.size() == num_basis_functions);

  // Copy with permutation  
  for (int i = 0; i < num_quadrature_points; ++i)
    for (int j = 0; j < num_basis_functions; ++j) 
      (*FE)[0][0][i][j] = tab(basix_derivative, i, perm[j], 0);

  bool debug_output = false;
  
  if (debug_output) {
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
  
}
