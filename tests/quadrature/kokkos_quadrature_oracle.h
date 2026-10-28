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
// host's number of points, and its points and weights exactly on a
// host execution space.

#ifndef KOKKOS_QUADRATURE_ORACLE_H
#define KOKKOS_QUADRATURE_ORACLE_H

#include "numerics/kokkos_numerics_oracle_test_utils.h"
#include "fe/kokkos_inverse_map_oracle.h"

#include "libmesh/elem.h"
#include "libmesh/quadrature_gauss_3D.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace libMeshTest
{
namespace KokkosQuadratureOracle
{

using namespace libMesh;

inline int
run_all_oracles()
{
  const ElemType types[] = {EDGE2, EDGE3, EDGE4,
                            TRI3, TRISHELL3, TRI3SUBDIVISION, TRI6, TRI7,
                            QUAD4, QUADSHELL4, QUAD8, QUADSHELL8, QUAD9, QUADSHELL9,
                            TET4, TET10, TET14, HEX8, HEX20, HEX27,
                            PRISM6, PRISM15, PRISM18, PRISM20, PRISM21,
                            PYRAMID5, PYRAMID13, PYRAMID14, PYRAMID18};

  // The host's rules, one after another, each case's (type, order,
  // allow negative weights, number of points, offset) in info
  std::vector<unsigned int> info;
  std::vector<Point> host_points;
  std::vector<Real> host_weights;
  for (const ElemType t : types)
    for (unsigned int o = 0; o <= FORTYTHIRD; ++o)
      for (const bool negative : {false, true})
        {
          QGauss qrule(Elem::type_to_dim_map[t], Order(o));
          qrule.allow_rules_with_negative_weights = negative;
          qrule.init(t);
          info.insert(info.end(), {unsigned(t), o, negative, qrule.n_points(),
                                   cast_int<unsigned int>(host_points.size())});
          host_points.insert(host_points.end(), qrule.get_points().begin(), qrule.get_points().end());
          host_weights.insert(host_weights.end(), qrule.get_weights().begin(), qrule.get_weights().end());
        }

  const std::size_t n_cases = info.size() / 5, n_total = host_points.size();
  ::Kokkos::View<unsigned int *> d_info("quadrature_info", info.size());
  ::Kokkos::View<unsigned int *> d_n("quadrature_n", n_cases);
  ::Kokkos::View<Point *> d_points("quadrature_points", n_total);
  ::Kokkos::View<Real *> d_weights("quadrature_weights", n_total);
  ::Kokkos::deep_copy(d_info, ::Kokkos::View<unsigned int *, ::Kokkos::HostSpace>(info.data(), info.size()));

  // One thread per rule, which goes in the host rule's place only if
  // it has the same number of points
  ::Kokkos::parallel_for(
    "quadrature_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const unsigned int * c = &d_info(5*k);
      d_n(k) = QGauss::n_points_elem(ElemType(c[0]), Order(c[1]), c[2]);
      if (d_n(k) == c[3])
        QGauss::rule_elem(ElemType(c[0]), Order(c[1]), c[2], &d_points(c[4]), &d_weights(c[4]));
    });

  auto h_n = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_n);
  auto h_points = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_points);
  auto h_weights = ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), d_weights);

  // Compare: exactly on a host execution space, where the kernel does the
  // host's arithmetic; a GPU may contract it differently.
  using ExecSpace = ::Kokkos::DefaultExecutionSpace;
  constexpr bool exact =
    ::Kokkos::SpaceAccessibility<::Kokkos::HostSpace, ExecSpace::memory_space>::accessible;

  int n_fail = 0;
  for (std::size_t k = 0; k != n_cases; ++k)
    {
      const unsigned int * c = &info[5*k];
      bool ok = h_n(k) == c[3];
      for (unsigned int q = c[4]; ok && q != c[4] + c[3]; ++q)
        ok = (KokkosInverseMapOracle::same_bits(host_points[q], h_points(q)) &&
              KokkosInverseMapOracle::same_bits(Point(host_weights[q]), Point(h_weights(q)))) ||
             (!exact && (host_points[q] - h_points(q)).norm() <= 1e-12 &&
              std::abs(host_weights[q] - h_weights(q)) <= 1e-12 * (1 + std::abs(host_weights[q])));
      if (!ok)
        {
          if (n_fail < 10)
            std::printf("[quadrature_oracle] mismatch: type %u order %u negative %u: "
                        "host %u points, kernel %u\n", c[0], c[1], c[2], c[3], h_n(k));
          ++n_fail;
        }
    }

  std::printf("[quadrature_oracle] [%s] %s  (%d failures; %zu rules, %zu points)\n",
              exact ? "exact" : "tolerance", n_fail ? "FAIL" : "PASS",
              n_fail, n_cases, n_total);
  return n_fail;
}

} // namespace KokkosQuadratureOracle
} // namespace libMeshTest

#endif // KOKKOS_QUADRATURE_ORACLE_H
