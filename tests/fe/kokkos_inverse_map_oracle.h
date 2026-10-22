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

// Checks fe_inverse_map(), run in a Kokkos kernel with LagrangeMapEval
// and NothrowInverseMapPolicy, against FEMap::inverse_map() on the host:
// every Lagrange element type, undistorted, distorted, tilted into 3D
// and collapsed, at points inside, on, outside and far from it, secure
// or not.  Where FEMap returns a point the kernel must return the same
// one, exactly on a host execution space; where FEMap throws, or
// returns a singular node, the kernel must report Newton hasn't converged.

#ifndef KOKKOS_INVERSE_MAP_ORACLE_H
#define KOKKOS_INVERSE_MAP_ORACLE_H

#include "numerics/kokkos_numerics_oracle_test_utils.h"

#include "libmesh/elem.h"
#include "libmesh/fe_map.h"
#include "libmesh/fe_map_point.h"
#include "libmesh/libmesh_exceptions.h"
#include "libmesh/node.h"
#include "libmesh/reference_elem.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

namespace libMeshTest
{
namespace KokkosInverseMapOracle
{

using namespace libMesh;

struct Case
{
  unsigned int dim, n_nodes, first_node;
  ElemType type;
  Order order;
  Point x;
  Real tol;
  bool secure;

  // What FEMap::inverse_map() did
  bool host_threw;
  Point host_p;
  bool host_at_singular_node;
};

// Node coordinates of one element of type t: the reference nodes, lightly
// distorted (variant 0), tilted out of plane for 1D/2D (1), or collapsed
// onto one point (2).
inline std::vector<Point>
element_nodes(const ElemType t, const int variant)
{
  const Elem & ref = ReferenceElem::get(t);
  const unsigned int dim = ref.dim();
  std::vector<Point> pts;
  for (unsigned int n = 0; n != ref.n_nodes(); ++n)
    {
      Point x = ref.point(n);
      if (variant == 2)
        x = Point(0.25, 0.5, 0.75);
      else
        {
          for (unsigned int d = 0; d != dim; ++d)
            x(d) += 0.03 * std::sin(1.7 * n + 0.9 * d + 0.3);
          if (variant == 1)
            {
              x(2) += 0.3 * x(0) - 0.2 * (dim > 1 ? x(1) : 0.) + 0.04 * std::cos(2.1 * n);
              if (dim == 1)
                x(1) += 0.5 * x(0);
            }
        }
      pts.push_back(x);
    }
  return pts;
}

// Bit for bit equal, so that a kernel reproducing the host's NaN, as
// from a collapsed element, matches it too.
inline bool
same_bits(const Point & a, const Point & b)
{
  for (unsigned int d = 0; d != 3; ++d)
    {
      const Real x = a(d), y = b(d);
      if (std::memcmp(&x, &y, sizeof(Real)))
        return false;
    }
  return true;
}

inline int
run_all_oracles()
{
  const ElemType types[] = {EDGE2, EDGE3, EDGE4,
                            TRI3, TRI6, TRI7, QUAD4, QUAD8, QUAD9,
                            TET4, TET10, TET14, HEX8, HEX20, HEX27,
                            PRISM6, PRISM15, PRISM18, PRISM20, PRISM21,
                            PYRAMID5, PYRAMID13, PYRAMID14, PYRAMID18};
  const Real tols[] = {TOLERANCE * TOLERANCE, 1.e-6};

  std::vector<Case> cases;
  std::vector<Point> all_nodes;

  // FEMap warns on libMesh::err when Newton struggles; keep that out of
  // the test output.
  std::ostringstream quiet;
  libMesh::err.reset(quiet);

  for (const ElemType t : types)
    for (int variant = 0; variant != 3; ++variant)
      {
        const Elem & ref = ReferenceElem::get(t);
        const unsigned int dim = ref.dim();
        if (variant == 1 && dim == 3)
          continue;

        const std::vector<Point> pts = element_nodes(t, variant);
        std::vector<std::unique_ptr<Node>> nodes;
        std::unique_ptr<Elem> elem = Elem::build(t);
        for (unsigned int n = 0; n != pts.size(); ++n)
          {
            nodes.push_back(std::make_unique<Node>(pts[n], n));
            elem->set_node(n, nodes.back().get());
          }

        const unsigned int first_node = all_nodes.size();
        all_nodes.insert(all_nodes.end(), pts.begin(), pts.end());

        std::vector<Point> targets;
        for (const Point & r : {Point(0.2, 0.15, 0.1), Point(1./3, 1./3, 0.25),
                                Point(-0.4, 0.3, 0.2), Point(1.4, -0.3, 0.6)})
          targets.push_back(FEMap::map(dim, elem.get(), r));
        for (unsigned int n = 0; n != ref.n_vertices(); ++n)
          targets.push_back(pts[n]);
        targets.push_back(Point(100., -50., 25.));

        for (const Point & x : targets)
          for (const Real tol : tols)
            for (const bool secure : {false, true})
              {
                Case c{dim, cast_int<unsigned int>(pts.size()), first_node, t,
                       elem->default_order(), x, tol, secure, false, Point(), false};
                try
                  {
                    c.host_p = FEMap::inverse_map(dim, elem.get(), x, tol, secure);
                    for (unsigned int n = 0; n != ref.n_nodes(); ++n)
                      if (elem->is_singular_node(n) && c.host_p == elem->master_point(n))
                        c.host_at_singular_node = true;
                  }
                catch (const DegenerateMap &)
                  {
                    c.host_threw = true;
                  }
                cases.push_back(c);
              }
      }

  libMesh::err.reset(std::cerr);

  // Upload the cases and node coordinates
  const std::size_t n_cases = cases.size();
  ::Kokkos::View<Real *> d_nodes("inverse_map_nodes", 3 * all_nodes.size());
  ::Kokkos::View<Real *> d_in("inverse_map_in", 4 * n_cases);
  ::Kokkos::View<unsigned int *> d_info("inverse_map_info", 6 * n_cases);
  ::Kokkos::View<Real *> d_out("inverse_map_out", 3 * n_cases);
  ::Kokkos::View<unsigned int *> d_converged("inverse_map_converged", n_cases);
  {
    auto h_nodes = ::Kokkos::create_mirror_view(d_nodes);
    for (std::size_t i = 0; i != all_nodes.size(); ++i)
      for (unsigned int d = 0; d != 3; ++d)
        h_nodes(3*i + d) = all_nodes[i](d);
    auto h_in = ::Kokkos::create_mirror_view(d_in);
    auto h_info = ::Kokkos::create_mirror_view(d_info);
    for (std::size_t k = 0; k != n_cases; ++k)
      {
        const Case & c = cases[k];
        for (unsigned int d = 0; d != 3; ++d)
          h_in(4*k + d) = c.x(d);
        h_in(4*k + 3) = c.tol;
        h_info(6*k + 0) = c.dim;
        h_info(6*k + 1) = c.type;
        h_info(6*k + 2) = c.order;
        h_info(6*k + 3) = c.n_nodes;
        h_info(6*k + 4) = c.first_node;
        h_info(6*k + 5) = c.secure;
      }
    ::Kokkos::deep_copy(d_nodes, h_nodes);
    ::Kokkos::deep_copy(d_in, h_in);
    ::Kokkos::deep_copy(d_info, h_info);
  }

  ::Kokkos::parallel_for(
    "inverse_map_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const unsigned int first = d_info(6*k + 4);
      const auto nodes = [=](const unsigned int i) {
        return Point(d_nodes(3*(first+i)), d_nodes(3*(first+i) + 1), d_nodes(3*(first+i) + 2));
      };
      const unsigned int dim = d_info(6*k + 0);
      const LagrangeMapEval<decltype(nodes)> eval
        {dim, ElemType(d_info(6*k + 1)), Order(d_info(6*k + 2)), d_info(6*k + 3), nodes};
      const NothrowInverseMapPolicy policy{dim};
      bool converged = false;
      const Point x(d_in(4*k), d_in(4*k + 1), d_in(4*k + 2));
      const Point p = fe_inverse_map(dim, x, d_in(4*k + 3), d_info(6*k + 5),
                                     eval, policy, converged);
      for (unsigned int d = 0; d != 3; ++d)
        d_out(3*k + d) = p(d);
      d_converged(k) = converged;
    });

  auto h_out = ::Kokkos::create_mirror_view(d_out);
  auto h_converged = ::Kokkos::create_mirror_view(d_converged);
  ::Kokkos::deep_copy(h_out, d_out);
  ::Kokkos::deep_copy(h_converged, d_converged);

  // Compare: exactly on a host execution space, where the kernel does the
  // host's arithmetic; a GPU may contract it differently.
  using ExecSpace = ::Kokkos::DefaultExecutionSpace;
  constexpr bool exact =
    ::Kokkos::SpaceAccessibility<::Kokkos::HostSpace, ExecSpace::memory_space>::accessible;

  int n_fail = 0, n_point = 0, n_threw = 0, n_singular_node = 0;
  for (std::size_t k = 0; k != n_cases; ++k)
    {
      const Case & c = cases[k];
      const bool converged = h_converged(k);
      const Point p(h_out(3*k), h_out(3*k + 1), h_out(3*k + 2));
      bool ok;
      if (c.host_threw)
        {
          // FEMap threw DegenerateMap: a singular Jacobian or a failed Newton
          ok = !converged;
          ++n_threw;
        }
      else if (c.host_at_singular_node)
        {
          // FEMap returns the singular node; the kernel doesn't look for one
          ok = !converged;
          ++n_singular_node;
        }
      else
        {
          ok = same_bits(p, c.host_p) ||
               (!exact && (p - c.host_p).norm() <= 1e-10);
          ++n_point;
        }
      if (!ok)
        {
          if (n_fail < 10)
            std::printf("[inverse_map_oracle] mismatch: type %d secure %d x (%g %g %g): "
                        "host %s (%.17g %.17g %.17g), kernel converged %d (%.17g %.17g %.17g)\n",
                        int(c.type), int(c.secure), double(c.x(0)), double(c.x(1)), double(c.x(2)),
                        c.host_threw ? "threw" : "returned",
                        double(c.host_p(0)), double(c.host_p(1)), double(c.host_p(2)),
                        int(converged), double(p(0)), double(p(1)), double(p(2)));
          ++n_fail;
        }
    }

  std::printf("[inverse_map_oracle] [%s] %s  (%d failures; %d points, %d throws, %d singular nodes)\n",
              exact ? "exact" : "tolerance", n_fail ? "FAIL" : "PASS",
              n_fail, n_point, n_threw, n_singular_node);
  return n_fail;
}

} // namespace KokkosInverseMapOracle
} // namespace libMeshTest

#endif // KOKKOS_INVERSE_MAP_ORACLE_H
