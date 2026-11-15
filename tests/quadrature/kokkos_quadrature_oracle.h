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

// Checks QGauss::n_points_elem() and QGauss::rule_elem(), run in a
// Kokkos kernel, against QGauss::init() on the host: every element
// type QGauss has a fixed rule for, every order up to FORTYTHIRD, with
// and without rules with negative weights.  The kernel must give the
// host's number of points, and its points and weights to within
// rounding.

#ifndef KOKKOS_QUADRATURE_ORACLE_H
#define KOKKOS_QUADRATURE_ORACLE_H

#include "numerics/kokkos_numerics_oracle_test_utils.h"

#include "libmesh/elem.h"
#include "libmesh/int_range.h"
#include "libmesh/quadrature_gauss_3D.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace libMeshTest
{
namespace KokkosQuadratureOracle
{

using namespace libMesh;

// One rule: its element type and order, whether it may have negative
// weights, and the host's number of points and where they start
struct Input
{
  ElemType type;
  Order order;
  bool negative_weights;

  // How many points the host's rule has, and where they start in the
  // host's points and weights
  unsigned int n_points, first;
};

inline int
run_all_oracles()
{
  const ElemType types[] = {NODEELEM,
                            EDGE2, EDGE3, EDGE4,
                            TRI3, TRISHELL3, TRI3SUBDIVISION, TRI6, TRI7,
                            QUAD4, QUADSHELL4, QUAD8, QUADSHELL8, QUAD9, QUADSHELL9,
                            TET4, TET10, TET14, HEX8, HEX20, HEX27,
                            PRISM6, PRISM15, PRISM18, PRISM20, PRISM21,
                            PYRAMID5, PYRAMID13, PYRAMID14, PYRAMID18};

  // The host's rules, one after another
  std::vector<Input> inputs;
  std::vector<Point> host_points;
  std::vector<Real> host_weights;
  for (const ElemType t : types)
    for (const auto o : make_range(unsigned(FORTYTHIRD) + 1))
      for (const bool negative : {false, true})
        {
          QGauss qrule(Elem::type_to_dim_map[t], Order(o));
          qrule.allow_rules_with_negative_weights = negative;
          qrule.init(t);
          inputs.push_back({t, Order(o), negative, qrule.n_points(),
                            cast_int<unsigned int>(host_points.size())});
          host_points.insert(host_points.end(), qrule.get_points().begin(), qrule.get_points().end());
          host_weights.insert(host_weights.end(), qrule.get_weights().begin(), qrule.get_weights().end());
        }

  const std::size_t n_cases = inputs.size(), n_total = host_points.size();
  const auto d_in = KokkosOracle::upload_objects(inputs, "quadrature_in");
  ::Kokkos::View<unsigned int *> d_n("quadrature_n", n_cases);
  ::Kokkos::View<Point *> d_points("quadrature_points", n_total);
  ::Kokkos::View<Real *> d_weights("quadrature_weights", n_total);

  // One thread per rule, which goes in the host rule's place only if
  // it has the same number of points
  ::Kokkos::parallel_for(
    "quadrature_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const Input in = d_in(k);
      d_n(k) = QGauss::n_points_elem(in.type, in.order, in.negative_weights);
      if (d_n(k) == in.n_points)
        QGauss::rule_elem(in.type, in.order, in.negative_weights,
                          &d_points(in.first), &d_weights(in.first));
    });

  auto h_n = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_n);
  auto h_points = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_points);
  auto h_weights = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_weights);

  int n_fail = 0;
  for (const auto k : make_range(n_cases))
    {
      const Input & in = inputs[k];
      bool ok = h_n(k) == in.n_points;
      // A weight goes through same_bits() as a Point's x.  1e-12: the
      // rules are tables, or tensor products of a table's values, so a
      // contracted product changes them by a few ulp
      for (unsigned int q = in.first; ok && q != in.first + in.n_points; ++q)
        ok = (KokkosOracle::same_bits(host_points[q], h_points(q)) &&
              KokkosOracle::same_bits(Point(host_weights[q]), Point(h_weights(q)))) ||
             ((host_points[q] - h_points(q)).norm() <= 1e-12 &&
              std::abs(host_weights[q] - h_weights(q)) <= 1e-12 * (1 + std::abs(host_weights[q])));
      if (!ok)
        {
          if (n_fail < 10)
            std::printf("[quadrature_oracle] mismatch: type %u order %u negative %u: "
                        "host %u points, kernel %u\n", unsigned(in.type), unsigned(in.order),
                        unsigned(in.negative_weights), in.n_points, h_n(k));
          ++n_fail;
        }
    }

  std::printf("[quadrature_oracle] %s  (%d failures; %zu rules, %zu points)\n",
              n_fail ? "FAIL" : "PASS",
              n_fail, n_cases, n_total);
  return n_fail;
}

} // namespace KokkosQuadratureOracle
} // namespace libMeshTest

#endif // KOKKOS_QUADRATURE_ORACLE_H
