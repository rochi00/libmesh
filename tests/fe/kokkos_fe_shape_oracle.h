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
// value to within rounding.

#ifndef KOKKOS_FE_SHAPE_ORACLE_H
#define KOKKOS_FE_SHAPE_ORACLE_H

#include "numerics/kokkos_numerics_oracle_test_utils.h"

#include "libmesh/elem.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_shape.h"
#include "libmesh/fe_type.h"
#include "libmesh/int_range.h"

#include <cstdio>
#include <memory>
#include <sstream>
#include <vector>

namespace libMeshTest
{
namespace KokkosFEShapeOracle
{

using namespace libMesh;

// Which function a case evaluates
enum Kind : unsigned int { VALUE, DERIV, SECOND_DERIV };

// What the kernel evaluates for one case: the function, of shape i of
// the element type's family and order, in direction j for a
// derivative, at reference point p
struct Input
{
  unsigned int dim;
  FEFamily family;
  ElemType type;
  Order order;
  // The shape, and the derivative direction, 0 for a value; for a
  // second derivative, j numbers xixi, xieta, etaeta, xizeta, etazeta,
  // zetazeta, as fe_shape_second_deriv() does
  unsigned int i, j;
  Kind kind;
  Point p;
};

// The host's value for in, through FEInterface on elem
inline Real
host_value(const Input & in, const Elem & elem)
{
  const FEType fe_t(in.order, in.family);
  switch (in.kind)
    {
    case VALUE:
      return FEInterface::shape(fe_t, &elem, in.i, in.p);
    case DERIV:
      return FEInterface::shape_deriv(fe_t, &elem, in.i, in.j, in.p);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
    case SECOND_DERIV:
      return FEInterface::shape_second_deriv(fe_t, &elem, in.i, in.j, in.p);
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
  // Two points inside every element, the second a triangle's centroid;
  // one outside the elements whose reference coordinates start at 0,
  // simplices and prisms, and inside the others; and one outside every
  // element, as an inverse map's Newton iterate may be.  Their
  // coordinates differ, so a shape mixing up xi, eta and zeta differs
  // too.
  const Point points[] = {Point(0.2, 0.15, 0.1), Point(1./3, 1./3, 0.25),
                          Point(-0.4, 0.3, 0.2), Point(1.4, -0.3, 0.6)};

  std::vector<Input> inputs;
  // What FEInterface gave for each input
  std::vector<Real> host;
  unsigned int n_elements = 0;

  for (const ElemType t : types)
    for (const FEFamily family : families)
      // Orders CONSTANT to FIFTH: past THIRD, the highest LAGRANGE order on
      // these types, and past FOURTH, the highest MONOMIAL order with shapes
      // of their own, so MONOMIAL's general case is checked too
      for (const auto o : make_range(unsigned(FIFTH) + 1))
        {
          // Not ReferenceElem::get(t), which gives a TRI3 for a
          // TRISHELL3 and so on; Lagrange and monomial shapes only
          // need the element's type, so it needs no nodes.
          const std::unique_ptr<Elem> elem = Elem::build(t);
          const unsigned int dim = elem->dim();
          const FEType fe_t(Order(o), family);
          if (o > unsigned(FEInterface::max_order(fe_t, t)))
            continue;

          // Every function of this element, or none if the host
          // rejects it
          std::vector<Input> elem_inputs;
          std::vector<Real> elem_host;
          try
            {
              const unsigned int n_shapes = FEInterface::n_shape_functions(fe_t, elem.get());
              for (const Point & p : points)
                for (const auto i : make_range(n_shapes))
                  {
                    elem_inputs.push_back({dim, family, t, Order(o), i, 0, VALUE, p});
                    for (const auto j : make_range(dim))
                      elem_inputs.push_back({dim, family, t, Order(o), i, j, DERIV, p});
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
                    for (const auto j : make_range(dim*(dim+1)/2))
                      elem_inputs.push_back({dim, family, t, Order(o), i, j, SECOND_DERIV, p});
#endif
                  }
              for (const Input & in : elem_inputs)
                elem_host.push_back(host_value(in, *elem));
            }
          catch (const std::exception &)
            {
              continue;
            }

          inputs.insert(inputs.end(), elem_inputs.begin(), elem_inputs.end());
          host.insert(host.end(), elem_host.begin(), elem_host.end());
          ++n_elements;
        }

  const auto d_in = KokkosOracle::upload_objects(inputs, "fe_shape_in");
  const std::size_t n_cases = inputs.size();
  ::Kokkos::View<Real *> d_out("fe_shape_out", n_cases);

  ::Kokkos::parallel_for(
    "fe_shape_oracle",
    n_cases,
    KOKKOS_LAMBDA(const std::size_t k) {
      const Input in = d_in(k);
      switch (in.kind)
        {
        case VALUE:
          d_out(k) = fe_shape(in.dim, in.family, in.type, in.order, in.i, in.p);
          break;
        case DERIV:
          d_out(k) = fe_shape_deriv(in.dim, in.family, in.type, in.order, in.i, in.j, in.p);
          break;
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        case SECOND_DERIV:
          d_out(k) = fe_shape_second_deriv(in.dim, in.family, in.type, in.order, in.i, in.j, in.p);
          break;
#endif
        }
    });

  auto h_out = ::Kokkos::create_mirror_view(d_out);
  ::Kokkos::deep_copy(h_out, d_out);

  // 1e-10 relative to 1 + |host|: far above the few ulp a contracted
  // evaluation of a shape polynomial changes, far below the change from
  // any wrong term
  KokkosOracle::ValueCheck check{"fe_shape_oracle", 1e-10};
  for (const auto k : make_range(n_cases))
    {
      const Input & in = inputs[k];
      check(host[k], h_out(k), [&]()
        {
          std::ostringstream os;
          os << "family " << int(in.family) << " type " << int(in.type) << " order " << int(in.order)
             << " kind " << int(in.kind) << " i " << in.i << " j " << in.j << " p " << in.p;
          return os.str();
        });
    }

  std::printf("[fe_shape_oracle] %s  (%d failures; %d elements, %d values)\n",
              check.n_fail ? "FAIL" : "PASS",
              check.n_fail, int(n_elements), check.n_values);
  return check.n_fail;
}

} // namespace KokkosFEShapeOracle
} // namespace libMeshTest

#endif // KOKKOS_FE_SHAPE_ORACLE_H
