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
// value to within rounding.  Each (type, family,
// order) runs in its own kernel instantiation with these as constants,
// as bucketed library kernels take them.

#ifndef KOKKOS_FE_SHAPE_ORACLE_H
#define KOKKOS_FE_SHAPE_ORACLE_H

#include "numerics/kokkos_numerics_oracle_test_utils.h"

#include "libmesh/elem.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_shape.h"
#include "libmesh/fe_type.h"
#include "libmesh/int_range.h"

#include <cstdio>
#include <iterator>
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

// The buckets: every (type, family, order) a case may have
constexpr ElemType types[] = {EDGE2, EDGE3, EDGE4,
                              TRI3, TRI6, TRI7, QUAD4, QUAD8, QUAD9,
                              TRISHELL3, QUADSHELL4, QUADSHELL8, QUADSHELL9,
                              TET4, TET10, TET14, HEX8, HEX20, HEX27,
                              PRISM6, PRISM15, PRISM18, PRISM20, PRISM21,
                              PYRAMID5, PYRAMID13, PYRAMID14, PYRAMID18};
constexpr FEFamily families[] = {LAGRANGE, L2_LAGRANGE, MONOMIAL};
constexpr std::size_t n_families = std::size(families);

// Orders CONSTANT to FIFTH: past THIRD, the highest LAGRANGE order on
// these types, and past FOURTH, the highest MONOMIAL order with shapes
// of their own, so MONOMIAL's general case is checked too
constexpr std::size_t n_orders = std::size_t(FIFTH) + 1;
constexpr std::size_t n_buckets = std::size(types) * n_families * n_orders;

// A bucket's key: the indices of its type in types and its family in
// families, and its order.  with_constant_index() takes it as the one
// index bucket_index() gives.
struct BucketKey
{
  std::size_t type, family, order;
};

constexpr std::size_t
bucket_index(const BucketKey & key)
{
  return (key.type * n_families + key.family) * n_orders + key.order;
}

// The key of bucket index b, the inverse of bucket_index()
constexpr BucketKey
bucket_key(const std::size_t b)
{
  return {b / (n_families * n_orders), b / n_orders % n_families, b % n_orders};
}

// What the kernel evaluates for one case: the function, of shape i,
// in direction j for a derivative, at reference point p
struct Input
{
  // The shape, and the derivative direction, 0 for a value; for a
  // second derivative, j numbers xixi, xieta, etaeta, xizeta, etazeta,
  // zetazeta, as fe_shape_second_deriv() does
  unsigned int i, j;
  Kind kind;
  Point p;
};

// Evaluates cases [begin, end), all of bucket B, with the bucket's key
// as compile-time constants
template <std::size_t B>
void
run_bucket(const ::Kokkos::View<Input *> & d_in,
           const ::Kokkos::View<Real *> & d_out,
           const std::size_t begin,
           const std::size_t end)
{
  constexpr BucketKey key = bucket_key(B);
  constexpr ElemType type = types[key.type];
  constexpr unsigned int dim = Elem::type_to_dim_map[type];
  constexpr FEFamily family = families[key.family];
  constexpr Order order = Order(key.order);

  ::Kokkos::parallel_for(
    "fe_shape_oracle",
    ::Kokkos::RangePolicy<>(begin, end),
    KOKKOS_LAMBDA(const std::size_t k) {
      const Input in = d_in(k);
      switch (in.kind)
        {
        case VALUE:
          d_out(k) = fe_shape(dim, family, type, order, in.i, in.p);
          break;
        case DERIV:
          d_out(k) = fe_shape_deriv(dim, family, type, order, in.i, in.j, in.p);
          break;
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
        case SECOND_DERIV:
          d_out(k) = fe_shape_second_deriv(dim, family, type, order, in.i, in.j, in.p);
          break;
#endif
        }
    });
}

// A case's element and FE type, and the value FEInterface gave for it
struct Reference
{
  FEFamily family;
  ElemType type;
  Order order;
  Real host;
};

// The host's value for in, through FEInterface on elem
inline Real
host_value(const FEType & fe_t, const Input & in, const Elem & elem)
{
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
  // Two points inside every element, the second a triangle's centroid;
  // one outside the elements whose reference coordinates start at 0,
  // simplices and prisms, and inside the others; and one outside every
  // element, as an inverse map's Newton iterate may be.  Their
  // coordinates differ, so a shape mixing up xi, eta and zeta differs
  // too.
  const Point points[] = {Point(0.2, 0.15, 0.1), Point(1./3, 1./3, 0.25),
                          Point(-0.4, 0.3, 0.2), Point(1.4, -0.3, 0.6)};

  std::vector<Input> inputs;
  std::vector<Reference> refs;

  // Each nonempty bucket and the end of its cases, which are contiguous
  std::vector<std::pair<std::size_t, std::size_t>> buckets;

  for (const auto t_i : make_range(std::size(types)))
    for (const auto f_i : make_range(std::size(families)))
      for (const auto o : make_range(n_orders))
        {
          const ElemType t = types[t_i];
          const FEFamily family = families[f_i];
          // Not ReferenceElem::get(t), which gives a TRI3 for a
          // TRISHELL3 and so on; Lagrange and monomial shapes only
          // need the element's type, so it needs no nodes.
          const std::unique_ptr<Elem> elem = Elem::build(t);
          const unsigned int dim = elem->dim();
          const FEType fe_t(Order(o), family);
          if (o > std::size_t(FEInterface::max_order(fe_t, t)))
            continue;

          // Every function of this element, or none if the host
          // rejects it
          std::vector<Input> elem_inputs;
          std::vector<Reference> elem_refs;
          try
            {
              const unsigned int n_shapes = FEInterface::n_shape_functions(fe_t, elem.get());
              for (const Point & p : points)
                for (const auto i : make_range(n_shapes))
                  {
                    elem_inputs.push_back({i, 0, VALUE, p});
                    for (const auto j : make_range(dim))
                      elem_inputs.push_back({i, j, DERIV, p});
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
                    for (const auto j : make_range(dim*(dim+1)/2))
                      elem_inputs.push_back({i, j, SECOND_DERIV, p});
#endif
                  }
              for (const Input & in : elem_inputs)
                elem_refs.push_back({family, t, Order(o), host_value(fe_t, in, *elem)});
            }
          catch (const std::exception &)
            {
              continue;
            }

          inputs.insert(inputs.end(), elem_inputs.begin(), elem_inputs.end());
          refs.insert(refs.end(), elem_refs.begin(), elem_refs.end());
          buckets.emplace_back(bucket_index({t_i, f_i, o}), inputs.size());
        }

  const auto d_in = KokkosOracle::upload_objects(inputs, "fe_shape_in");
  const std::size_t n_cases = inputs.size();
  ::Kokkos::View<Real *> d_out("fe_shape_out", n_cases);

  std::size_t begin = 0;
  for (const auto & [b, end] : buckets)
    {
      KokkosOracle::with_constant_index<n_buckets>
        (b, [&, end = end](auto bucket)
            { run_bucket<decltype(bucket)::value>(d_in, d_out, begin, end); });
      begin = end;
    }

  auto h_out = ::Kokkos::create_mirror_view(d_out);
  ::Kokkos::deep_copy(h_out, d_out);

  // 1e-10 relative to 1 + |host|: far above the few ulp a contracted
  // evaluation of a shape polynomial changes, far below the change from
  // any wrong term
  KokkosOracle::ValueCheck check{"fe_shape_oracle", 1e-10};
  for (const auto k : make_range(n_cases))
    {
      const Input & in = inputs[k];
      const Reference & r = refs[k];
      check(r.host, h_out(k), [&]()
        {
          std::ostringstream os;
          os << "family " << int(r.family) << " type " << int(r.type) << " order " << int(r.order)
             << " kind " << int(in.kind) << " i " << in.i << " j " << in.j << " p " << in.p;
          return os.str();
        });
    }

  std::printf("[fe_shape_oracle] %s  (%d failures; %d elements, %d values)\n",
              check.n_fail ? "FAIL" : "PASS",
              check.n_fail, int(buckets.size()), check.n_values);
  return check.n_fail;
}

} // namespace KokkosFEShapeOracle
} // namespace libMeshTest

#endif // KOKKOS_FE_SHAPE_ORACLE_H
