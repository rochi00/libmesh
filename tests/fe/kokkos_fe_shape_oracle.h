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

// Checks fe_shape(), fe_shape_deriv() and fe_shape_second_deriv(), run
// in a Kokkos kernel, against FEInterface on the host: every shape
// function, first and second derivative of every LAGRANGE, L2_LAGRANGE
// and MONOMIAL element the host evaluates, at reference points inside
// and outside the element.  The kernel must return the host's
// value, exactly on a host execution space.

#ifndef KOKKOS_FE_SHAPE_ORACLE_H
#define KOKKOS_FE_SHAPE_ORACLE_H

#include "numerics/kokkos_numerics_oracle_test_utils.h"

#include "libmesh/elem.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_shape.h"
#include "libmesh/fe_type.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace libMeshTest
{
namespace KokkosFEShapeOracle
{

using namespace libMesh;

// Which function a case evaluates
enum Kind : unsigned int { VALUE, DERIV, SECOND_DERIV };

struct Case
{
  unsigned int dim;
  FEFamily family;
  ElemType type;
  Order order;
  unsigned int i, j;
  Kind kind;
  Point p;
  Real host;
};

// The host's value for c, through FEInterface on elem
inline Real
host_value(const Case & c, const Elem & elem)
{
  const FEType fe_t(c.order, c.family);
  switch (c.kind)
    {
    case VALUE:
      return FEInterface::shape(fe_t, &elem, c.i, c.p);
    case DERIV:
      return FEInterface::shape_deriv(fe_t, &elem, c.i, c.j, c.p);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
    case SECOND_DERIV:
      return FEInterface::shape_second_deriv(fe_t, &elem, c.i, c.j, c.p);
#endif
    default:
      libmesh_error();
    }
}

inline int
run_all_oracles()
{
  const ElemType types[] = {EDGE2, EDGE3, EDGE4,
                            TRI3, TRI6, TRI7, QUAD4, QUAD8, QUAD9,
                            TRISHELL3, QUADSHELL4, QUADSHELL8, QUADSHELL9,
                            TET4, TET10, TET14, HEX8, HEX20, HEX27,
                            PRISM6, PRISM15, PRISM18, PRISM20, PRISM21,
                            PYRAMID5, PYRAMID13, PYRAMID14, PYRAMID18};
  const FEFamily families[] = {LAGRANGE, L2_LAGRANGE, MONOMIAL};
  const Point points[] = {Point(0.2, 0.15, 0.1), Point(1./3, 1./3, 0.25),
                          Point(-0.4, 0.3, 0.2), Point(1.4, -0.3, 0.6)};

  std::vector<Case> cases;
  unsigned int n_elements = 0;

  for (const ElemType t : types)
    for (const FEFamily family : families)
      for (int o = CONSTANT; o <= FIFTH; ++o)
        {
          // Not ReferenceElem::get(t), which gives a TRI3 for a
          // TRISHELL3 and so on; Lagrange and monomial shapes only
          // need the element's type, so it needs no nodes.
          const std::unique_ptr<Elem> elem = Elem::build(t);
          const unsigned int dim = elem->dim();
          const FEType fe_t(Order(o), family);
          if (o > int(FEInterface::max_order(fe_t, t)))
            continue;

          // Every function of this element, or none if the host
          // rejects it
          std::vector<Case> elem_cases;
          try
            {
              const unsigned int n_shapes = FEInterface::n_shape_functions(fe_t, elem.get());
              for (const Point & p : points)
                for (unsigned int i = 0; i != n_shapes; ++i)
                  {
                    Case c{dim, family, t, Order(o), i, 0, VALUE, p, 0};
                    elem_cases.push_back(c);
                    c.kind = DERIV;
                    for (c.j = 0; c.j != dim; ++c.j)
                      elem_cases.push_back(c);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
                    c.kind = SECOND_DERIV;
                    for (c.j = 0; c.j != dim*(dim+1)/2; ++c.j)
                      elem_cases.push_back(c);
#endif
                  }
              for (Case & c : elem_cases)
                c.host = host_value(c, *elem);
            }
          catch (const std::exception &)
            {
              continue;
            }

          cases.insert(cases.end(), elem_cases.begin(), elem_cases.end());
          ++n_elements;
        }

  // Upload the cases
  const std::size_t n_cases = cases.size();
  ::Kokkos::View<Real *> d_p("fe_shape_p", 3 * n_cases);
  ::Kokkos::View<unsigned int *> d_info("fe_shape_info", 7 * n_cases);
  ::Kokkos::View<Real *> d_out("fe_shape_out", n_cases);
  {
    auto h_p = ::Kokkos::create_mirror_view(d_p);
    auto h_info = ::Kokkos::create_mirror_view(d_info);
    for (std::size_t k = 0; k != n_cases; ++k)
      {
        const Case & c = cases[k];
        for (unsigned int d = 0; d != 3; ++d)
          h_p(3*k + d) = c.p(d);
        h_info(7*k + 0) = c.dim;
        h_info(7*k + 1) = c.family;
        h_info(7*k + 2) = c.type;
        h_info(7*k + 3) = c.order;
        h_info(7*k + 4) = c.i;
        h_info(7*k + 5) = c.j;
        h_info(7*k + 6) = c.kind;
      }
    ::Kokkos::deep_copy(d_p, h_p);
    ::Kokkos::deep_copy(d_info, h_info);
  }

  ::Kokkos::parallel_for(
    "fe_shape_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const unsigned int dim = d_info(7*k + 0);
      const FEFamily family = FEFamily(d_info(7*k + 1));
      const ElemType type = ElemType(d_info(7*k + 2));
      const Order order = Order(d_info(7*k + 3));
      const unsigned int i = d_info(7*k + 4), j = d_info(7*k + 5);
      const Point p(d_p(3*k), d_p(3*k + 1), d_p(3*k + 2));
      switch (d_info(7*k + 6))
        {
        case VALUE:
          d_out(k) = fe_shape(dim, family, type, order, i, p);
          break;
        case DERIV:
          d_out(k) = fe_shape_deriv(dim, family, type, order, i, j, p);
          break;
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        case SECOND_DERIV:
          d_out(k) = fe_shape_second_deriv(dim, family, type, order, i, j, p);
          break;
#endif
        }
    });

  auto h_out = ::Kokkos::create_mirror_view(d_out);
  ::Kokkos::deep_copy(h_out, d_out);

  // Compare: exactly on a host execution space, where the kernel does the
  // host's arithmetic; a GPU may contract it differently.
  using ExecSpace = ::Kokkos::DefaultExecutionSpace;
  constexpr bool exact =
    ::Kokkos::SpaceAccessibility<::Kokkos::HostSpace, ExecSpace::memory_space>::accessible;

  int n_fail = 0;
  for (std::size_t k = 0; k != n_cases; ++k)
    {
      const Case & c = cases[k];
      const Real x = h_out(k);
      const bool ok = !std::memcmp(&x, &c.host, sizeof(Real)) ||
        (!exact && std::abs(x - c.host) <= 1e-10 * (1 + std::abs(c.host)));
      if (!ok)
        {
          if (n_fail < 10)
            std::printf("[fe_shape_oracle] mismatch: family %d type %d order %d kind %d i %u j %u "
                        "p (%g %g %g): host %.17g, kernel %.17g\n",
                        int(c.family), int(c.type), int(c.order), int(c.kind), c.i, c.j,
                        double(c.p(0)), double(c.p(1)), double(c.p(2)),
                        double(c.host), double(x));
          ++n_fail;
        }
    }

  std::printf("[fe_shape_oracle] [%s] %s  (%d failures; %d elements, %zu values)\n",
              exact ? "exact" : "tolerance", n_fail ? "FAIL" : "PASS",
              n_fail, int(n_elements), n_cases);
  return n_fail;
}

} // namespace KokkosFEShapeOracle
} // namespace libMeshTest

#endif // KOKKOS_FE_SHAPE_ORACLE_H
