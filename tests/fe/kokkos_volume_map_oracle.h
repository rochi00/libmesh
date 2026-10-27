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

// Checks fe_map_*D() and fe_map_second_derivs_*(), run in a Kokkos
// kernel on map derivatives from LagrangeMapEval, against
// FEMap::compute_map() on the host: every Lagrange element type,
// distorted and, for 1D and 2D, tilted into 3D, at each quadrature
// point.  The kernel must give JxW and the inverse map's first and
// second derivatives exactly on a host execution space.  On elements
// with an affine map FEMap computes neither JxW, which it rescales from
// the first point's, nor second derivatives at the other points, so
// those aren't checked there.

#ifndef KOKKOS_VOLUME_MAP_ORACLE_H
#define KOKKOS_VOLUME_MAP_ORACLE_H

#include "numerics/kokkos_numerics_oracle_test_utils.h"
#include "fe/kokkos_inverse_map_oracle.h"

#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_map_point.h"
#include "libmesh/node.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/reference_elem.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace libMeshTest
{
namespace KokkosVolumeMapOracle
{

using namespace libMesh;

// Per quadrature point: JxW, the gradients of xi, eta and zeta, and
// their second derivatives
constexpr unsigned int n_out = 1 + 9 + 18;

struct Case
{
  unsigned int dim;
  ElemType type;
  Order order;
  unsigned int n_nodes, first_node;
  Point qp;
  Real w;

  // What FEMap computed, and a bit for each value it computed
  Real host[n_out];
  std::uint32_t computed;
};

inline int
run_all_oracles()
{
  const ElemType types[] = {EDGE2, EDGE3, EDGE4,
                            TRI3, TRI6, TRI7, QUAD4, QUAD8, QUAD9,
                            TET4, TET10, TET14, HEX8, HEX20, HEX27,
                            PRISM6, PRISM15, PRISM18, PRISM20, PRISM21,
                            PYRAMID5, PYRAMID13, PYRAMID14, PYRAMID18};

  std::vector<Case> cases;
  std::vector<Point> all_nodes;

  for (const ElemType t : types)
    for (int variant = 0; variant != 2; ++variant)
      {
        const Elem & ref = ReferenceElem::get(t);
        const unsigned int dim = ref.dim();
        if (variant == 1 && dim == 3)
          continue;

        const std::vector<Point> pts = KokkosInverseMapOracle::element_nodes(t, variant);
        std::vector<std::unique_ptr<Node>> nodes;
        std::unique_ptr<Elem> elem = Elem::build(t);
        for (unsigned int n = 0; n != pts.size(); ++n)
          {
            nodes.push_back(std::make_unique<Node>(pts[n], n));
            elem->set_node(n, nodes.back().get());
          }

        const unsigned int first_node = all_nodes.size();
        all_nodes.insert(all_nodes.end(), pts.begin(), pts.end());

        std::unique_ptr<FEBase> fe = FEBase::build(dim, FEType(elem->default_order(), LAGRANGE));
        QGauss qrule(dim, FIFTH);
        fe->attach_quadrature_rule(&qrule);
        const FEMap & map = fe->get_fe_map();
        const std::vector<Real> * first[] =
          {&map.get_dxidx(), &map.get_dxidy(), &map.get_dxidz(),
           &map.get_detadx(), &map.get_detady(), &map.get_detadz(),
           &map.get_dzetadx(), &map.get_dzetady(), &map.get_dzetadz()};
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        fe->get_d2phi();
        const std::vector<std::vector<Real>> * second[] =
          {&map.get_d2xidxyz2(), &map.get_d2etadxyz2(), &map.get_d2zetadxyz2()};
#endif
        const auto & JxW = fe->get_JxW();
        fe->reinit(elem.get());

        for (unsigned int q = 0; q != JxW.size(); ++q)
          {
            const bool affine = elem->has_affine_map();
            Case c{dim, t, elem->default_order(), cast_int<unsigned int>(pts.size()),
                   first_node, qrule.qp(q), qrule.w(q), {}, !affine || q == 0};
            c.host[0] = JxW[q];
            for (unsigned int i = 0; i != 9; ++i)
              if (q < first[i]->size())
                {
                  c.host[1 + i] = (*first[i])[q];
                  c.computed |= std::uint32_t(1) << (1 + i);
                }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            if (!affine)
              for (unsigned int i = 0; i != 3; ++i)
                if (q < second[i]->size() && (*second[i])[q].size() == 6)
                  for (unsigned int k = 0; k != 6; ++k)
                    {
                      c.host[10 + 6*i + k] = (*second[i])[q][k];
                      c.computed |= std::uint32_t(1) << (10 + 6*i + k);
                    }
#endif
            cases.push_back(c);
          }
      }

  // Upload the cases and node coordinates
  const std::size_t n_cases = cases.size();
  ::Kokkos::View<Real *> d_nodes("volume_map_nodes", 3 * all_nodes.size());
  ::Kokkos::View<Real *> d_in("volume_map_in", 4 * n_cases);
  ::Kokkos::View<unsigned int *> d_info("volume_map_info", 5 * n_cases);
  ::Kokkos::View<Real *> d_out("volume_map_out", n_out * n_cases);
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
          h_in(4*k + d) = c.qp(d);
        h_in(4*k + 3) = c.w;
        h_info(5*k + 0) = c.dim;
        h_info(5*k + 1) = c.type;
        h_info(5*k + 2) = c.order;
        h_info(5*k + 3) = c.n_nodes;
        h_info(5*k + 4) = c.first_node;
      }
    ::Kokkos::deep_copy(d_nodes, h_nodes);
    ::Kokkos::deep_copy(d_in, h_in);
    ::Kokkos::deep_copy(d_info, h_info);
  }

  ::Kokkos::parallel_for(
    "volume_map_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const unsigned int dim = d_info(5*k), first = d_info(5*k + 4);
      const auto nodes = [=](const unsigned int i) {
        return Point(d_nodes(3*(first+i)), d_nodes(3*(first+i) + 1), d_nodes(3*(first+i) + 2));
      };
      const LagrangeMapEval<decltype(nodes)> eval
        {dim, ElemType(d_info(5*k + 1)), Order(d_info(5*k + 2)), d_info(5*k + 3), nodes};

      const Point qp(d_in(4*k), d_in(4*k + 1), d_in(4*k + 2));
      Point dxyz[3];
      for (unsigned int j = 0; j != dim; ++j)
        dxyz[j] = eval.map_deriv(j, qp);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
      RealVectorValue d2xyz[6];
      for (unsigned int j = 0; j != dim*(dim+1)/2; ++j)
        d2xyz[j] = eval.map_second_deriv(j, qp);
#endif
      const auto no_check = [](const Real) {};

      FEMapPoint<> m;
      switch (dim)
        {
        case 1:
          fe_map_1D(dxyz[0], m, no_check);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          fe_map_second_derivs_1D(dxyz[0], d2xyz, m);
#endif
          break;
        case 2:
          fe_map_2D(dxyz[0], dxyz[1], m, no_check);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          fe_map_second_derivs_2D(dxyz[0], dxyz[1], d2xyz, m);
#endif
          break;
        case 3:
          fe_map_3D(dxyz[0], dxyz[1], dxyz[2], m, no_check);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
          fe_map_second_derivs_square(d2xyz, m);
#endif
          break;
        default:
          break;
        }

      Real * out = &d_out(n_out*k);
      out[0] = m.jac * d_in(4*k + 3);
      for (unsigned int d = 0; d != 3; ++d)
        {
          out[1 + d] = m.dxi(d);
          out[4 + d] = m.deta(d);
          out[7 + d] = m.dzeta(d);
        }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
      for (unsigned int i = 0; i != 6; ++i)
        {
          out[10 + i] = m.d2xi[i];
          out[16 + i] = m.d2eta[i];
          out[22 + i] = m.d2zeta[i];
        }
#endif
    });

  auto h_out = ::Kokkos::create_mirror_view(d_out);
  ::Kokkos::deep_copy(h_out, d_out);

  // Compare: exactly on a host execution space, where the kernel does the
  // host's arithmetic; a GPU may contract it differently.
  using ExecSpace = ::Kokkos::DefaultExecutionSpace;
  constexpr bool exact =
    ::Kokkos::SpaceAccessibility<::Kokkos::HostSpace, ExecSpace::memory_space>::accessible;

  int n_fail = 0, n_values = 0;
  for (std::size_t k = 0; k != n_cases; ++k)
    {
      const Case & c = cases[k];
      for (unsigned int i = 0; i != n_out; ++i)
        {
          if (!(c.computed >> i & 1))
            continue;
          ++n_values;
          const Real host = c.host[i], kernel = h_out(n_out*k + i);
          if (!KokkosInverseMapOracle::same_bits(Point(host), Point(kernel)) &&
              (exact || std::abs(kernel - host) > 1e-10 * (1 + std::abs(host))))
            {
              if (n_fail < 10)
                std::printf("[volume_map_oracle] mismatch: type %d qp (%g %g %g) value %u: "
                            "host %.17g kernel %.17g\n",
                            int(c.type), double(c.qp(0)), double(c.qp(1)), double(c.qp(2)),
                            i, double(host), double(kernel));
              ++n_fail;
            }
        }
    }

  std::printf("[volume_map_oracle] [%s] %s  (%d failures; %zu points, %d values)\n",
              exact ? "exact" : "tolerance", n_fail ? "FAIL" : "PASS",
              n_fail, n_cases, n_values);
  return n_fail;
}

} // namespace KokkosVolumeMapOracle
} // namespace libMeshTest

#endif // KOKKOS_VOLUME_MAP_ORACLE_H
