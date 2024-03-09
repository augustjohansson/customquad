#ifndef CUSTOMQUAD_ALGOIM_UTILS_HPP
#define CUSTOMQUAD_ALGOIM_UTILS_HPP

#include "quadrature_general.hpp"

namespace algoim_utils
{

  std::vector<int> cut_cells;
  std::vector<int> uncut_cells;
  std::vector<int> outside_cells;
  std::vector<std::vector<double>> qr_pts;
  std::vector<std::vector<double>> qr_w;
  std::vector<std::vector<double>> qr_pts_bdry;
  std::vector<std::vector<double>> qr_w_bdry;
  std::vector<std::vector<double>> qr_n;
  std::vector<std::vector<double>> xyz, xyz_bdry;

  template <class T>
  void convert(const T& qr,
	       const int gdim,
	       std::vector<double>& pts,
	       std::vector<double>& w)
  {
    const std::size_t N = qr.nodes.size();
    pts.resize(gdim * N);
    w.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
      for (int d = 0; d < gdim; ++d)
	pts[gdim * i + d] = qr.nodes[i].x(d);
      w[i] = qr.nodes[i].w;
    }
  }

  template<class Y, int gdim>
  void run_template(const Y& phi,
		    const std::vector<double>& LLx,
		    const std::vector<double>& LLy,
		    const std::vector<double>& LLz,
		    const std::vector<double>& URx,
		    const std::vector<double>& URy,
		    const std::vector<double>& URz,
		    int degree,
		    bool do_verbose,
		    bool do_map,
		    bool do_scale)
  {
    std::cout << __FILE__ << ' ' << __FUNCTION__ << std::endl;

    if (do_verbose) {
      std::cout << "LLx " << LLx.size() << '\n'
		<< "LLy " << LLy.size() << '\n';
      if (gdim == 3)
	std::cout << "LLz " << LLz.size() << '\n';
      std::cout << "URx " << URx.size() << '\n'
		<< "URy " << URy.size() << '\n';
      if (gdim == 3)
	std::cout << "URz " << URz.size() << '\n';
      std::cout << "degree " << degree << '\n'
		<< "do_verbose " << do_verbose << '\n'
		<< "map " << do_map << '\n'
		<< "do_scale " << do_scale << std::endl;
    }

    const int side = -1;
    const int dim_bulk = -1;

    const std::size_t num_cells = LLx.size();
    assert(num_cells == LLy.size());
    if (gdim == 3)
      assert(num_cells == LLz.size());
    assert(num_cells == URx.size());
    assert(num_cells == URy.size());
    if (gdim == 3)
      assert(num_cells == URz.size());
    qr_pts.resize(num_cells);
    qr_w.resize(num_cells);
    qr_pts_bdry.resize(num_cells);
    qr_w_bdry.resize(num_cells);
    qr_n.resize(num_cells);
    xyz.resize(num_cells);
    xyz_bdry.resize(num_cells);

    for (std::size_t cell_no = 0; cell_no < num_cells; ++cell_no) {
      algoim::uvector<double, gdim> xmin, xmax;
      xmin(0) = LLx[cell_no];
      xmin(1) = LLy[cell_no];
      xmax(0) = URx[cell_no];
      xmax(1) = URy[cell_no];
      if (gdim == 3) {
	xmin(2) = LLz[cell_no];
	xmax(2) = URz[cell_no];
      }

      const auto q_bdry = algoim::quadGen<gdim>
	(phi, algoim::HyperRectangle<double, gdim>(xmin, xmax), gdim, side, degree);

      if (q_bdry.nodes.size()) {
	// Cut cell
	cut_cells.push_back(cell_no);

	// Boundary qr
	convert(q_bdry, gdim, qr_pts_bdry[cell_no], qr_w_bdry[cell_no]);
	xyz_bdry[cell_no] = qr_pts_bdry[cell_no];

	// Normal
	for (std::size_t k = 0; k < qr_pts_bdry[cell_no].size() / gdim; ++k) {
	  algoim::uvector<double, gdim> pt;
	  for (std::size_t d = 0; d < gdim; ++d)
	    pt(d) = qr_pts_bdry[cell_no][gdim * k + d];
	  const algoim::uvector<double, gdim> grad = phi.grad(pt);
	  for (std::size_t d = 0; d < gdim; ++d)
	    qr_n[cell_no].push_back(grad(d)); // FIXME
	}

	// Bulk
	const auto q_bulk = algoim::quadGen<gdim>
	  (phi, algoim::HyperRectangle<double, gdim>(xmin, xmax), dim_bulk, side, degree);
	convert(q_bulk, gdim, qr_pts[cell_no], qr_w[cell_no]);
	xyz[cell_no] = qr_pts[cell_no];

	const algoim::uvector<double, gdim> dx(xmax - xmin);
	double vol = 1;
	for (std::size_t d = 0; d < gdim; ++d)
	  vol *= dx(d);

	if (do_scale) {
	  for (double& w: qr_w[cell_no])
	    w /= vol;
	  for (double& w: qr_w_bdry[cell_no])
	    w /= vol;
	}

	if (do_map) {
	  auto map = [&](std::vector<double>& qr) {
	    for (std::size_t i = 0; i < qr.size(); i += gdim) {
	      for (std::size_t d = 0; d < gdim; ++d)
		qr[i + d] = (qr[i + d] - xmin(d)) / dx(d);
	    }
	  };
	  map(qr_pts[cell_no]);
	  map(qr_pts_bdry[cell_no]);
	}
      }
      else {
	// Either outside or inside
	const algoim::uvector<double, gdim> midp(0.5 * (xmin + xmax));
	if (phi(midp) < 0)
	  uncut_cells.push_back(cell_no);
	else
	  outside_cells.push_back(cell_no);
      }
    }

    for (std::size_t i = 0; i < num_cells; ++i) {
      if (qr_w_bdry[i].size()) {
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
      else {
	assert(qr_pts[i].size() == 0);
	assert(qr_w[i].size() == 0);
	assert(qr_pts_bdry[i].size() == 0);
	assert(qr_w_bdry[i].size() == 0);
	assert(qr_n[i].size() == 0);
	assert(xyz[i].size() == 0);
	assert(xyz_bdry[i].size() == 0);
      }
    }

    if (do_verbose) {
      auto print = [](const std::vector<std::vector<double>>& p) {
	for (std::size_t i = 0; i < p.size(); ++i)
	  for (std::size_t j = 0; j < p[i].size(); j += gdim) {
	    for (std::size_t d = 0; d < gdim; ++d)
	      std::cout << p[i][j+d] << ' ';
	    std::cout << '\n';
	  }
      };

      std::cout << "% bulk [0,1]^" << gdim << "\n"
		<<"qr_pts=[";
      print(qr_pts);
      std::cout << "];\n";

      std::cout << "% bdry [0,1]^" << gdim << "\n"
		<< "qr_pts_bdry=[";
      print(qr_pts_bdry);
      std::cout << "];\n";

      std::cout << "xyz=[";
      print(xyz);
      std::cout << "];\n";

      std::cout << "xyz_bdry=[";
      print(xyz_bdry);
      std::cout << "];\n";

      std::cout << "qr_n=[";
      print(qr_n);
      std::cout << "];\n";

      std::cout << "drawarrow(xyz_bdry, xyz_bdry+qr_n);\n";
    }

    std::cout << __FILE__ << " done" << std::endl;
  }
}

#endif
