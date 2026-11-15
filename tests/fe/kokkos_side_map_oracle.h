// The libMesh Finite Element Library.
// Copyright (C) 2002-2026 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

// Checks fe_side_reference_point(), run in a Kokkos kernel, against
// FE::side_map() on the host: every side of every element type
// Elem::master_point() knows, at the points of several Gauss rules on
// the side.  The kernel must give the host's reference points to
// within rounding.

#ifndef KOKKOS_SIDE_MAP_ORACLE_H
#define KOKKOS_SIDE_MAP_ORACLE_H

#include "numerics/kokkos_numerics_oracle_test_utils.h"

#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_map_point.h"
#include "libmesh/int_range.h"
#include "libmesh/node.h"
#include "libmesh/quadrature_gauss.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace libMeshTest
{
namespace KokkosSideMapOracle
{

using namespace libMesh;

// A point on a side of an element type, for the kernel to map to the
// element's reference coordinates
struct Input
{
  ElemType type;
  unsigned int side;

  // The point on the side's reference element
  Point qp;
};

inline int
run_all_oracles()
{
  const ElemType types[] = {EDGE2, EDGE3, EDGE4,
                            TRI3, TRISHELL3, TRI6, TRI7,
                            QUAD4, QUADSHELL4, QUAD8, QUADSHELL8, QUAD9, QUADSHELL9,
                            TET4, TET10, TET14, HEX8, HEX20, HEX27,
                            PRISM6, PRISM15, PRISM18, PRISM20, PRISM21,
                            PYRAMID5, PYRAMID13, PYRAMID14, PYRAMID18};
  // Odd orders only: QGauss gives orders 2n and 2n+1 the same rule on
  // edges and quads, so these are their rules of 1 to 5 points per
  // direction
  const Order orders[] = {FIRST, THIRD, FIFTH, SEVENTH, NINTH};

  std::vector<Input> inputs;
  // What FE::side_map() mapped each input to
  std::vector<Point> refs;

  for (const ElemType t : types)
    {
      // side_map() reads only the element's type and topology, so its
      // nodes may as well be the reference element's
      std::unique_ptr<Elem> elem = Elem::build(t);
      std::vector<std::unique_ptr<Node>> nodes;
      for (const auto n : make_range(elem->n_nodes()))
        {
          nodes.push_back(std::make_unique<Node>(Elem::master_point(t, n), n));
          elem->set_node(n, nodes.back().get());
        }
      const unsigned int dim = elem->dim();

      for (const auto s : make_range(elem->n_sides()))
        for (const Order o : orders)
          {
            std::unique_ptr<const Elem> side = elem->build_side_ptr(s);
            // Request the map before side_map(), as a FE::reinit()
            // caller would
            std::unique_ptr<FEBase> fe = FEBase::build(dim, FEType(FIRST, LAGRANGE));
            fe->get_xyz();
            QGauss qside(dim - 1, o);
            qside.init(*side);
            std::vector<Point> ref;
            fe->side_map(elem.get(), side.get(), s, qside.get_points(), ref);

            for (const auto q : index_range(ref))
              {
                inputs.push_back({t, s, qside.qp(q)});
                refs.push_back(ref[q]);
              }
          }
    }

  const auto d_in = KokkosOracle::upload_objects(inputs, "side_map_in");
  const std::size_t n_cases = inputs.size();
  ::Kokkos::View<Point *> d_out("side_map_out", n_cases);

  ::Kokkos::parallel_for(
    "side_map_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const Input in = d_in(k);
      d_out(k) = fe_side_reference_point(in.type, in.side, in.qp);
    });

  auto h_out = ::Kokkos::create_mirror_view(d_out);
  ::Kokkos::deep_copy(h_out, d_out);

  int n_fail = 0;
  for (const auto k : make_range(n_cases))
    {
      const Input & in = inputs[k];
      const Point & host = refs[k];
      const Point & kernel = h_out(k);
      // 1e-12 relative to 1 + |host|: a reference point is a sum of
      // a few node coordinates times side shapes, so a contracted sum
      // changes it by a few ulp
      if (!KokkosOracle::same_bits(host, kernel) &&
          (kernel - host).norm() > 1e-12 * (1 + host.norm()))
        {
          if (n_fail < 10)
            std::printf("[side_map_oracle] mismatch: type %d side %u qp (%g %g %g): "
                        "host (%.17g %.17g %.17g) kernel (%.17g %.17g %.17g)\n",
                        int(in.type), in.side, double(in.qp(0)), double(in.qp(1)), double(in.qp(2)),
                        double(host(0)), double(host(1)), double(host(2)),
                        double(kernel(0)), double(kernel(1)), double(kernel(2)));
          ++n_fail;
        }
    }

  std::printf("[side_map_oracle] %s  (%d failures; %zu points)\n",
              n_fail ? "FAIL" : "PASS",
              n_fail, n_cases);
  return n_fail;
}

} // namespace KokkosSideMapOracle
} // namespace libMeshTest

#endif // KOKKOS_SIDE_MAP_ORACLE_H
