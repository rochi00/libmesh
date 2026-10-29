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

// Checks fe_face_map_*D() and fe_face_curvature_*D(), run in a Kokkos
// kernel on side map derivatives from LagrangeMapEval, against
// FEMap::compute_face_map() on the host: every side of every 2D and 3D
// Lagrange element type, distorted and, for 2D, tilted into 3D, at each
// face quadrature point.  A 2D element's side in 3D finds the element's
// own map derivatives by fe_inverse_map(), as FEMap does by
// FEMap::inverse_map().  The kernel must give the side's JxW, normals,
// tangents and curvatures exactly on a host execution space.  (Not its
// points: FE::reinit() replaces those with the element's own map at
// the side's points.)

#ifndef KOKKOS_FACE_MAP_ORACLE_H
#define KOKKOS_FACE_MAP_ORACLE_H

#include "numerics/kokkos_numerics_oracle_test_utils.h"
#include "fe/kokkos_inverse_map_oracle.h"

#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_map_point.h"
#include "libmesh/node.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/reference_elem.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace libMeshTest
{
namespace KokkosFaceMapOracle
{

using namespace libMesh;

// Per face quadrature point: JxW, the normal, two tangents and the
// curvature
constexpr unsigned int n_out = 1 + 3 + 6 + 1;

struct Case
{
  unsigned int dim, side;
  ElemType type, side_type;
  Order order, side_order;
  unsigned int n_nodes, first_node, n_side_nodes, first_side_node;
  Point qp;
  Real w;

  // What FEMap::compute_face_map() computed, and how many of the
  // tangents and whether the curvature is set
  Real host[n_out];
  unsigned int n_tangents;
  bool has_curvature;
};

inline int
run_all_oracles()
{
  const ElemType types[] = {TRI3, TRI6, TRI7, QUAD4, QUAD8, QUAD9,
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

        for (unsigned int s = 0; s != elem->n_sides(); ++s)
          {
            std::unique_ptr<const Elem> side = elem->build_side_ptr(s);
            const unsigned int first_side_node = all_nodes.size();
            for (const Point & x : side->node_ref_range())
              all_nodes.push_back(x);

            std::unique_ptr<FEBase> fe = FEBase::build(dim, FEType(FIRST, LAGRANGE));
            QGauss qface(dim - 1, FIFTH);
            fe->attach_quadrature_rule(&qface);
            const auto & JxW = fe->get_JxW();
            const auto & normals = fe->get_normals();
            const auto & tangents = fe->get_tangents();
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            const auto & curvatures = fe->get_curvatures();
#endif
            fe->reinit(elem.get(), s);

            for (unsigned int q = 0; q != JxW.size(); ++q)
              {
                Case c{dim, s, t, side->type(), elem->default_order(), side->default_order(),
                       cast_int<unsigned int>(pts.size()), first_node,
                       side->n_nodes(), first_side_node,
                       qface.qp(q), qface.w(q), {}, 0, false};
                c.host[0] = JxW[q];
                for (unsigned int d = 0; d != 3; ++d)
                  c.host[1 + d] = normals[q](d);
                if (q < tangents.size())
                  {
                    c.n_tangents = tangents[q].size();
                    for (unsigned int k = 0; k != c.n_tangents; ++k)
                      for (unsigned int d = 0; d != 3; ++d)
                        c.host[4 + 3*k + d] = tangents[q][k](d);
                  }
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
                if (q < curvatures.size())
                  {
                    c.has_curvature = true;
                    c.host[10] = curvatures[q];
                  }
#endif
                cases.push_back(c);
              }
          }
      }

  // Upload the cases and node coordinates
  const std::size_t n_cases = cases.size();
  ::Kokkos::View<Real *> d_nodes("face_map_nodes", 3 * all_nodes.size());
  ::Kokkos::View<Real *> d_in("face_map_in", 4 * n_cases);
  ::Kokkos::View<unsigned int *> d_info("face_map_info", 10 * n_cases);
  ::Kokkos::View<Real *> d_out("face_map_out", n_out * n_cases);
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
        const unsigned int info[10] = {c.dim, c.side, unsigned(c.type), unsigned(c.order),
                                       c.n_nodes, c.first_node, unsigned(c.side_type),
                                       unsigned(c.side_order), c.n_side_nodes, c.first_side_node};
        for (unsigned int i = 0; i != 10; ++i)
          h_info(10*k + i) = info[i];
      }
    ::Kokkos::deep_copy(d_nodes, h_nodes);
    ::Kokkos::deep_copy(d_in, h_in);
    ::Kokkos::deep_copy(d_info, h_info);
  }

  ::Kokkos::parallel_for(
    "face_map_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const auto info = [=](const unsigned int i) { return d_info(10*k + i); };
      const unsigned int dim = info(0), first = info(5), side_first = info(9);
      const auto elem_nodes = [=](const unsigned int i) {
        return Point(d_nodes(3*(first+i)), d_nodes(3*(first+i) + 1), d_nodes(3*(first+i) + 2));
      };
      const auto side_nodes = [=](const unsigned int i) {
        return Point(d_nodes(3*(side_first+i)), d_nodes(3*(side_first+i) + 1), d_nodes(3*(side_first+i) + 2));
      };
      const LagrangeMapEval<decltype(elem_nodes)> elem_eval
        {dim, ElemType(info(2)), Order(info(3)), info(4), elem_nodes};
      const LagrangeMapEval<decltype(side_nodes)> side_eval
        {dim - 1, ElemType(info(6)), Order(info(7)), info(8), side_nodes};

      const Point qp(d_in(4*k), d_in(4*k + 1), d_in(4*k + 2));
      const auto no_check = [](const Real) {};

      FEFaceMapPoint<> f;
      Real curvature = 0;
      switch (dim)
        {
        case 2:
          {
            const Point xyz = side_eval.map(qp);
            const auto parent_derivs = [&](RealVectorValue & dx_dxi, RealVectorValue & dx_deta)
              {
                const NothrowInverseMapPolicy policy;
                bool converged = false;
                const Point p = fe_inverse_map(dim, xyz, TOLERANCE, true, elem_eval, policy, converged);
                dx_dxi  = elem_eval.map_deriv(0, p);
                dx_deta = elem_eval.map_deriv(1, p);
              };
            const Point dxyzdxi = side_eval.map_deriv(0, qp);
            fe_face_map_2D(dxyzdxi, parent_derivs, f, no_check);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            curvature = fe_face_curvature_2D(dxyzdxi, side_eval.map_second_deriv(0, qp), f.normal);
#endif
            break;
          }
        case 3:
          {
            const Point dxyzdxi = side_eval.map_deriv(0, qp);
            const Point dxyzdeta = side_eval.map_deriv(1, qp);
            fe_face_map_3D(dxyzdxi, dxyzdeta, f, no_check);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
            curvature = fe_face_curvature_3D(dxyzdxi, dxyzdeta,
                                             side_eval.map_second_deriv(0, qp),
                                             side_eval.map_second_deriv(1, qp),
                                             side_eval.map_second_deriv(2, qp),
                                             f.normal);
#endif
            break;
          }
        default:
          break;
        }

      Real * out = &d_out(n_out*k);
      out[0] = f.jac * d_in(4*k + 3);
      for (unsigned int d = 0; d != 3; ++d)
        {
          out[1 + d] = f.normal(d);
          out[4 + d] = f.tangents[0](d);
          out[7 + d] = f.tangents[1](d);
        }
      out[10] = curvature;
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
          const bool set = (i < 4) ? true :
                           (i < 10) ? (i - 4) / 3 < c.n_tangents :
                           c.has_curvature;
          if (!set)
            continue;
          ++n_values;
          const Real host = c.host[i], kernel = h_out(n_out*k + i);
          if (!KokkosInverseMapOracle::same_bits(Point(host), Point(kernel)) &&
              (exact || std::abs(kernel - host) > 1e-10 * (1 + std::abs(host))))
            {
              if (n_fail < 10)
                std::printf("[face_map_oracle] mismatch: type %d side %u qp (%g %g %g) value %u: "
                            "host %.17g kernel %.17g\n",
                            int(c.type), c.side, double(c.qp(0)), double(c.qp(1)), double(c.qp(2)),
                            i, double(host), double(kernel));
              ++n_fail;
            }
        }
    }

  std::printf("[face_map_oracle] [%s] %s  (%d failures; %zu points, %d values)\n",
              exact ? "exact" : "tolerance", n_fail ? "FAIL" : "PASS",
              n_fail, n_cases, n_values);
  return n_fail;
}

} // namespace KokkosFaceMapOracle
} // namespace libMeshTest

#endif // KOKKOS_FACE_MAP_ORACLE_H
